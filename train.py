#!/usr/bin/env python3
"""Train a PlantVillage classifier with MobileNetV2 transfer learning.

Pipeline (follows the standard transfer-learning recipe):
  1. Discover images in class-named folders and remove exact duplicates.
  2. Split FILE LISTS into train / validation / test (stratified, fixed seed)
     BEFORE any augmentation, so no augmented copy can leak across splits.
  3. Phase 1 - feature extraction: frozen ImageNet MobileNetV2 + new classifier.
  4. Phase 2 - fine-tuning: unfreeze the top layers with a very small learning rate.
  5. Choose the better phase using the VALIDATION set only.
  6. Evaluate once on the untouched TEST set and export model + reports.

Usage:
    python train.py --data_dir data/plantvillage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from utils import IMG_SIZE, parse_class_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Folder containing one sub-folder per class (e.g. data/plantvillage/anthracnose/).")
    p.add_argument("--output_dir", default="models", help="Where the model and reports are written.")
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=8, help="Epochs for feature extraction.")
    p.add_argument("--epochs_finetune", type=int, default=10, help="Epochs for fine-tuning (0 to skip).")
    p.add_argument("--fine_tune_at", type=int, default=100,
                   help="Index of the first MobileNetV2 layer to unfreeze (154 layers in total).")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="Use 'none' only for offline smoke tests.")
    p.add_argument("--max_images_per_class", type=int, default=0,
                   help="Subsample each class (0 = use all). Handy for quick trial runs.")
    p.add_argument("--skip_dedup", action="store_true", help="Skip exact-duplicate removal.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------
# Data discovery, de-duplication and splitting
# ----------------------------------------------------------------------------
def resolve_root(data_dir: Path) -> Path:
    """Accept either the class folder root or a parent that contains a 'color' folder."""
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "  -> Download the PlantVillage images and point --data_dir at the folder that "
            "holds one sub-folder per class (README, step 2), e.g.\n"
            "     python train.py --data_dir data\\plantvillage\n       (the folder must contain class sub-folders like anthracnose/, healthy/ etc.)"
        )
    if (data_dir / "color").is_dir():
        return data_dir / "color"
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(subdirs) == 1 and any(d.is_dir() for d in subdirs[0].iterdir()):
        return subdirs[0]  # e.g. an extra wrapper folder from unzipping
    return data_dir


def discover_images(root: Path, max_per_class: int, seed: int):
    """Discover images in class-named folders and return class names cleaned for Blackgram."""
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    rng = random.Random(seed)
    paths, labels, class_names = [], [], []
    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            print(f"  [warn] no images in '{class_dir.name}', skipping")
            continue
        if max_per_class and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        idx = len(class_names)
        # Clean folder name to get the canonical class name:
        #   "Anthracnose 230" -> "anthracnose", "Healthy 220" -> "healthy",
        #   "Leaf Crinckle 150" -> "leaf_crinkle", "Powdery Mildew 180" -> "powdery_mildew",
        #   "Yellow Mosaic 220" -> "yellow_mosaic"
        raw = class_dir.name
        # Remove trailing digit(s) and space, then lowercase and replace spaces with underscores
        cleaned = " ".join(raw.split()).strip()
        # Strip trailing digits
        import re
        cleaned = re.sub(r"\s*\d+\s*$", "", cleaned)
        cleaned = cleaned.lower().replace(" ", "_")
        class_names.append(cleaned)
        paths.extend(str(f) for f in files)
        labels.extend([idx] * len(files))
    if len(class_names) < 2:
        raise SystemExit(
            f"Found {len(class_names)} class folder(s) with images under '{root}'.\n"
            "  -> Expected at least two sub-folders, one per class, e.g.\n"
            "     data/plantvillage/anthracnose/  data/plantvillage/healthy/\n"
            "  -> Use the original (non-augmented) images and make sure each folder actually "
            "contains image files."
        )
    return paths, np.array(labels), class_names


def drop_duplicates(paths, labels):
    """Remove byte-identical files. If identical bytes carry different labels, drop all copies."""
    seen: dict[str, int] = {}
    conflicting: set[str] = set()
    hashes = []
    for path, label in zip(paths, labels):
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
        hashes.append(digest)
        if digest in seen and seen[digest] != label:
            conflicting.add(digest)
        seen.setdefault(digest, label)

    kept_paths, kept_labels, used = [], [], set()
    for path, label, digest in zip(paths, labels, hashes):
        if digest in conflicting or digest in used:
            continue
        used.add(digest)
        kept_paths.append(path)
        kept_labels.append(label)
    removed = len(paths) - len(kept_paths)
    return kept_paths, np.array(kept_labels), removed


def make_splits(paths, labels, val_size, test_size, seed):
    counts = Counter(labels.tolist())
    too_small = [c for c, n in counts.items() if n < 10]
    if too_small:
        raise SystemExit(
            "Every class needs at least 10 images for a stratified split, but "
            f"{len(too_small)} class(es) have fewer (smallest: {min(counts[c] for c in too_small)}).\n"
            "  -> Add more images to those classes, or use --max_images_per_class only on "
            "classes that are large enough."
        )
    idx = np.arange(len(paths))
    trainval_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, stratify=labels[trainval_idx], random_state=seed
    )
    parts = {}
    for name, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        parts[name] = ([paths[i] for i in ids], labels[ids])
    # Hard guarantee: no file appears in more than one split.
    sets = {k: set(v[0]) for k, v in parts.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), \
        "Data leakage detected between splits!"
    return parts


# ----------------------------------------------------------------------------
# tf.data pipelines
# ----------------------------------------------------------------------------
def build_augmenter(seed: int) -> keras.Sequential:
    """Training-only augmentation. Applied AFTER the split, never to val/test."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.15, seed=seed),
            layers.RandomZoom((-0.2, 0.1), seed=seed),
            layers.RandomContrast(0.15, seed=seed),
            layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        ],
        name="augmentation",
    )


def make_dataset(paths, labels, img_size, batch_size, training, seed, augmenter=None):
    ds = tf.data.Dataset.from_tensor_slices((np.array(paths), labels.astype("int32")))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size), method="bilinear")
        image.set_shape((img_size, img_size, 3))
        return image, label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).ignore_errors()
    # ignore_errors() marks the dataset cardinality as INFINITE (-2), which makes
    # Keras 3 unable to size an epoch ("input ran out of data" warning). Re-assert
    # the real element count so every epoch sees exactly one full pass.
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(paths)))
    ds = ds.batch(batch_size)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes: int, img_size: int, dropout: float, weights: str | None):
    """MobileNetV2 backbone + new classifier. Raw 0-255 pixels go in; scaling is inside the model."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)  # -> [-1, 1]
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="plant_disease_mobilenetv2"), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def callbacks(patience=3):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
    ]


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def plot_history(histories, out_path: Path, boundary: int | None):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tr, va, title in ((axes[0], acc, val_acc, "Accuracy"), (axes[1], loss, val_loss, "Loss")):
        ax.plot(tr, label="train")
        ax.plot(va, label="validation")
        if boundary:
            ax.axvline(boundary - 1, color="gray", linestyle="--", label="start fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, class_names, out_path: Path):
    short = [parse_class_name(c)["label"] for c in class_names]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, len(class_names) * 0.32)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def predict_dataset(model, ds):
    """Return (y_true, y_pred) by iterating once, so labels and predictions always align."""
    y_true, y_pred = [], []
    for x, y in ds:
        probs = model(x, training=False).numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(probs.argmax(axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("== 1/6  Discovering images ==")
    root = resolve_root(Path(args.data_dir))
    paths, labels, class_names = discover_images(root, args.max_images_per_class, args.seed)
    print(f"  {len(paths):,} images, {len(class_names)} classes (root: {root})")

    removed = 0
    if not args.skip_dedup:
        print("== 2/6  Removing exact duplicates ==")
        paths, labels, removed = drop_duplicates(paths, labels)
        print(f"  removed {removed:,} duplicate/conflicting files")

    print("== 3/6  Splitting BEFORE augmentation (stratified) ==")
    parts = make_splits(paths, labels, args.val_size, args.test_size, args.seed)
    sizes = {k: len(v[0]) for k, v in parts.items()}
    print("  " + ", ".join(f"{k}: {n:,}" for k, n in sizes.items()))

    augmenter = build_augmenter(args.seed)
    train_ds = make_dataset(*parts["train"], args.img_size, args.batch_size, True, args.seed, augmenter)
    val_ds = make_dataset(*parts["val"], args.img_size, args.batch_size, False, args.seed)
    test_ds = make_dataset(*parts["test"], args.img_size, args.batch_size, False, args.seed)

    present = np.unique(parts["train"][1])
    weights_arr = compute_class_weight("balanced", classes=present, y=parts["train"][1])
    class_weight = {int(c): float(w) for c, w in zip(present, weights_arr)}

    print("== 4/6  Phase 1: feature extraction (frozen backbone) ==")
    model, base = build_model(len(class_names), args.img_size, args.dropout,
                              None if args.weights == "none" else "imagenet")
    compile_model(model, args.lr_head)
    histories = [model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                           class_weight=class_weight, callbacks=callbacks(), shuffle=False, verbose=2)]
    phase1_val = model.evaluate(val_ds, verbose=0)[1]
    phase1_weights = model.get_weights()
    print(f"  phase 1 validation accuracy: {phase1_val:.4f}")
    boundary = None
    phase = "feature_extraction"

    if args.epochs_finetune > 0:
        print(f"== 5/6  Phase 2: fine-tuning from layer {args.fine_tune_at} ==")
        base.trainable = True
        for layer in base.layers[: args.fine_tune_at]:
            layer.trainable = False
        for layer in base.layers:  # never update BatchNorm statistics on a small LR schedule
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, args.lr_finetune)  # re-compile after changing trainable flags
        boundary = len(histories[0].history["loss"]) + 1
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                                   class_weight=class_weight, callbacks=callbacks(patience=3),
                                   shuffle=False, verbose=2))
        phase2_val = model.evaluate(val_ds, verbose=0)[1]
        print(f"  phase 2 validation accuracy: {phase2_val:.4f}")
        if phase2_val >= phase1_val:
            phase = "fine_tuned"
        else:
            print("  fine-tuning did not improve validation accuracy; keeping phase-1 weights")
            model.set_weights(phase1_weights)
    else:
        print("== 5/6  Fine-tuning skipped ==")

    print("== 6/6  Final evaluation on the held-out TEST set ==")
    y_true, y_pred = predict_dataset(model, test_ds)
    labels_idx = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels_idx, zero_division=0,
                                   target_names=[parse_class_name(c)["label"] for c in class_names],
                                   output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=labels_idx, zero_division=0,
                                        target_names=[parse_class_name(c)["label"] for c in class_names])
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    test_acc = float((y_true == y_pred).mean())
    print(f"  TEST accuracy: {test_acc:.4f}   macro-F1: {report['macro avg']['f1-score']:.4f}")

    # Export a clean copy (no optimizer state) so the committed file stays small.
    export, _ = build_model(len(class_names), args.img_size, args.dropout, None)
    export.set_weights(model.get_weights())
    model_path = out_dir / "plant_disease_mobilenetv2.keras"
    export.save(model_path)

    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (out_dir / "classification_report.txt").write_text(report_text)
    plot_history(histories, out_dir / "training_curves.png", boundary)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    metrics = {
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "phase1_val_accuracy": float(phase1_val),
        "selected_phase": phase,
        "num_classes": len(class_names),
        "split_sizes": sizes,
        "duplicates_removed": removed,
        "img_size": args.img_size,
        "epochs_run": [len(h.history["loss"]) for h in histories],
        "seed": args.seed,
        "weights": args.weights,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model ({model_path.stat().st_size / 1e6:.1f} MB) and reports to '{out_dir}/'.")


if __name__ == "__main__":
    main()
