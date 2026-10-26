# 🌿 Plant Disease Detection with MobileNetV2 Transfer Learning

A Streamlit web app that classifies a photo of a plant leaf into a PlantVillage
disease or *healthy* class. A MobileNetV2 network pre-trained on ImageNet is reused
through **transfer learning**: first as a frozen feature extractor, then fine-tuned.

> **Status:** the code is complete and has been verified end-to-end (training run,
> model loading, image upload, prediction). The repo ships **without** a trained
> model — train it once (locally or on Colab) and commit `models/plant_disease_mobilenetv2.keras`
> + `models/class_names.json` before deploying. No accuracy is claimed here, because
> no real training run has been done yet.

> **Dataset:** Blackgram Plant Leaf Disease Classification (BPLD) with 5 classes:

## Features

- Pre-trained **MobileNetV2** + new classifier head, two-phase training (feature extraction → fine-tuning)
- Leakage-aware pipeline: exact-duplicate removal, stratified **train / validation / test split made before augmentation**
- Augmentation (flip, rotation, zoom, contrast, brightness) on the training set only
- Overfitting controls: dropout, early stopping, learning-rate reduction, class weights, small fine-tuning LR, frozen BatchNorm
- Model selection uses the **validation** set; the **test** set is evaluated once at the end
- UI: upload or camera input, confidence bar, top-N predictions, low-confidence warnings
- Invalid-file handling: empty, corrupt, non-image, unsupported format, oversized, too-small images; prediction errors are caught and shown as messages, not stack traces
- Training dependencies are completely separate from the deployed app

## Project structure

```
plant-disease-detection/
├── app.py                  # Streamlit app (no training code, no training imports)
├── train.py                # Training + evaluation script
├── utils.py                # Shared helpers: class-name parsing, image validation
├── requirements.txt        # APP / deployment dependencies  <- Streamlit Cloud installs only this
├── requirements-train.txt  # + scikit-learn, matplotlib    <- install only to train
├── .streamlit/config.toml  # Theme, 10 MB upload limit, no telemetry
├── .gitignore              # Keeps data/, caches and scratch files out of Git
├── models/                 # Trained model + reports (empty until you train)
├── data/                   # Dataset goes here (git-ignored, never committed)
├── .venv/                  # Python environment (git-ignored, created in step 1)
└── .cache/                 # Local pip/temp/Keras/matplotlib caches (git-ignored)
```

The virtual environment lives **inside** the repo at `.venv/` (git-ignored, so it can never
be pushed to GitHub) — see "Why long-path support had to be enabled first" below.

## Prerequisites

- Windows 10 (commands below are Windows **Command Prompt**; the project also works on Linux/macOS)
- **Python 3.10–3.12** (verified with Python 3.12.10)
- About **2 GB of free disk space** for the Python packages (E: drive is used for that here)
- A GitHub account and a [Streamlit Community Cloud](https://share.streamlit.io) account for deployment
- Optional: a Google account for the free Colab training alternative

---

## 1. Setup (copy-paste, in order)

Open **Command Prompt** and run:

```cmd
cd /d "E:\Downloads\DL\DL project\plant-disease-detection\plant-disease-detection"

mkdir .cache\tmp .cache\pip .cache\keras .cache\mpl 2>nul

rem --- keep pip, temp files, Keras weights and matplotlib on the E: drive ---
set "TMP=%CD%\.cache\tmp"
set "TEMP=%CD%\.cache\tmp"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
set "KERAS_HOME=%CD%\.cache\keras"
set "MPLCONFIGDIR=%CD%\.cache\mpl"

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

> **Already set up on this computer:** `.venv` exists and is fully installed
> (`tensorflow-cpu 2.21.0`, `streamlit 1.66.1`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`).
> Re-running the block above is harmless (idempotent).
>
> An older environment at `E:\plant-disease-venv` also exists and works. Either is fine —
> just never activate both in the same window.
>
> **Every new Command Prompt window** needs the `cd /d ...` line and the five `set`
> lines again before you use the venv, because environment variables do not survive
> between windows. To verify your setup:
>
> ```cmd
> python -c "import tensorflow, streamlit; print(tensorflow.__version__, streamlit.__version__)"
> ```
>
> (Must print `2.21.0 1.66.1`.)

### Why long-path support had to be enabled first

The very first `python -m venv .venv` + `pip install` on this machine failed with:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
...\.venv\Lib\site-packages\tensorflow\include\external\envoy_api\...\...upb_minitable.h
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

That was **not** a disk-space error: Windows' 260-character path limit (MAX_PATH) was
exceeded. The longest file inside the TensorFlow wheel needs 262 characters when installed
in this project's `.venv` — 2 over the limit, so the install stopped halfway and left a
broken environment behind.

**Fix (already applied on this computer):** enable Win32 long paths. Open an *elevated*
(Administrator) Command Prompt and run:

```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Then re-run step 1. On any other machine this one line must run **before** TensorFlow can be
installed into this folder. If you cannot use an Administrator prompt, create the environment
at a short path instead (`python -m venv E:\venv`) — that needs no system change.

The pip cache, pip's temporary folders, the Keras weight cache and matplotlib's config are
redirected into `.cache\` for the same reason (short paths) **and** because C: is full.

### What this fixes — and what it does not

Keeping the environment, pip cache and temp folders on E: means **installing packages and
training no longer need any free space on C:**. That is what made the TensorFlow install work.

It does **not** fix a full C: drive. Windows itself, browsers, antivirus and other programs
still write to C:, and with only a few MB free those programs will keep failing. Free at least
**2–5 GB on C:** for normal use (Disk Cleanup → "Temporary files", clear the browser cache,
empty `C:\Users\<you>\AppData\Local\Temp`). Treat the E: redirection as "the project works",
not as "the C: problem is solved".

---

## 2. Get the Blackgram BPLD dataset

Nothing is downloaded automatically — check free space first (the dataset is about **1–2 GB**
unzipped; E: has plenty).

The BPLD dataset contains five classes: **anthracnose, healthy, leaf_crinkle, powdery_mildew, yellow_mosaic**.

1. Download the BPLD dataset (extracted ZIP already contains the class folders).
2. Arrange the class folders under `data/plantvillage/` as follows:

```
data\plantvillage\
├── anthracnose\
├── healthy\
├── leaf_crinkle\
├── powdery_mildew\
└── yellow_mosaic\
```

The folder names are the canonical class names used for training and prediction.

`--data_dir` may point at the `data/plantvillage/` folder directly.

> ⚠️ Do **not** use pre-augmented copies. Augmenting before splitting
> puts near-identical images in both train and test and inflates accuracy.

The `data\` folder is git-ignored, so the dataset can never be pushed to GitHub.

---

## 3. Train

With the Command Prompt from step 1 active (venv activated, `set` lines applied):

```cmd
set "MPLCONFIGDIR=%CD%\.cache\mpl"
python train.py --data_dir data\\plantvillage
```

A GPU is strongly recommended — see the Colab section if you don't have one
(TensorFlow on native Windows is **CPU-only** since TF 2.11).

> **Every class folder needs at least 10 usable images** (after de-duplication), otherwise the
> stratified split cannot be built and `train.py` stops with a clear message. For a fast first
> trial add `--max_images_per_class 200`.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--epochs_head` | 8 | Feature-extraction epochs (frozen backbone) |
| `--epochs_finetune` | 10 | Fine-tuning epochs; `0` skips fine-tuning |
| `--fine_tune_at` | 100 | First MobileNetV2 layer to unfreeze (154 layers total) |
| `--lr_head` / `--lr_finetune` | 1e-3 / 1e-5 | Learning rates per phase |
| `--batch_size` | 32 | Lower it if you run out of memory |
| `--max_images_per_class` | 0 (all) | Subsample for a quick trial, e.g. `200` |
| `--weights` | `imagenet` | `none` only for offline smoke tests |

What the script does:

1. Finds images in class folders (anthracnose, healthy, leaf_crinkle, powdery_mildew, yellow_mosaic),
   removes byte-identical duplicates, and uses the cleaned folder name as the canonical class name.
2. Splits file lists 70 / 15 / 15 (stratified, seed 42) **before** any augmentation and
   asserts that no file appears in two splits.
3. Phase 1: trains only the new classifier on the frozen backbone.
4. Phase 2: unfreezes the top MobileNetV2 layers at a very small learning rate.
5. Keeps whichever phase has the higher **validation** accuracy.
6. Evaluates once on the untouched test set and writes everything to `models\`:

```
models\plant_disease_mobilenetv2.keras   # model without optimizer state (~10 MB)
models\class_names.json
models\metrics.json                      # test accuracy, macro-F1, split sizes, ...
models\classification_report.txt
models\confusion_matrix.png
models\training_curves.png
```

### About leakage

The pipeline prevents the avoidable leaks (augmenting before splitting, shared validation/test
use, exact duplicates). The BPLD dataset has five classes with varying numbers of images.
A random split cannot fully separate photos of the same leaf — treat the test accuracy as an
optimistic estimate for photos taken in real fields.

---

## 4. Run the app locally

```cmd
streamlit run app.py
```

Open http://localhost:8501. Without a trained model the app shows setup instructions
instead of crashing.

---

## 5. Push to GitHub

There is no Git repository in this folder yet, so create one first (one-time):

```cmd
cd /d "E:\Downloads\DL\DL project\plant-disease-detection\plant-disease-detection"
git init
git status
git add .
git commit -m "Plant disease detection with MobileNetV2 transfer learning"
git branch -M main
git remote add origin https://github.com/<your-username>/plant-disease-detection.git
git push -u origin main
```

Before pushing, `git status` must **not** list anything under `data\` or `.cache\`.
After training it must list `models\plant_disease_mobilenetv2.keras` and
`models\class_names.json` (about 10 MB total — comfortably under GitHub's 100 MB limit).

### What is committed and what is not

| Path | In Git? | Why |
|---|---|---|
| `*.py`, `requirements*.txt`, `.streamlit/`, `README.md` | ✅ | Source |
| `models\*.keras`, `models\*.json`, `models\*.png`, `models\*.txt` | ✅ after training | The deployed app needs them (~10 MB) |
| `data\` (dataset) | ❌ | 1.5–2 GB, never committed |
| `.cache\` (pip/temp/Keras/matplotlib caches) | ❌ | Machine-local, large |
| `.venv\`, `venv\` | ❌ | Machine-local environments |

---

## 6. Deploy on Streamlit Community Cloud (free)

1. Push the repository to GitHub (step 5), **including the trained `models\` files**.
2. Sign in at https://share.streamlit.io with GitHub.
3. **Create app** → choose your repository, branch `main`, main file `app.py`.
4. Open **Advanced settings** → Python version **3.12**.
5. **Deploy.** The first build installs `tensorflow-cpu` and takes a few minutes.

### Which dependencies go where

| File | Packages | Installed by | When |
|---|---|---|---|
| `requirements.txt` | streamlit, tensorflow-cpu, numpy, pillow | **Streamlit Cloud** | Every deploy |
| `requirements-train.txt` | everything above **+** scikit-learn, matplotlib | You, locally / on Colab | Only when training |

Streamlit Cloud installs `requirements.txt` only — which is why training packages are kept in a
separate file. The app loads the model once per server process (`st.cache_resource`).

---

## 7. Alternative: train on Google Colab (free GPU, no local disk needed)

Use this when the local machine cannot hold TensorFlow or has no usable GPU. Colab runs on
Google's servers, so **your C: drive is not involved at all**.

1. Push the project to GitHub first (step 5), or zip the folder and upload it manually.
2. Open https://colab.research.google.com → **File → New notebook**.
3. Choose a GPU: **Runtime → Change runtime type → T4 GPU → Save**.
4. Get the code into the notebook:

   ```python
   !git clone https://github.com/<your-username>/plant-disease-detection.git
   %cd plant-disease-detection
   ```

   (Or use **Files → Upload** and then `!unzip plant-disease-detection.zip && %cd plant-disease-detection`.)
5. Upload the dataset to Colab (Files → Upload, or `!kaggle datasets download -d abdallahalidev/plantvillage-dataset`
   after configuring a Kaggle API token) and unzip it into `data/`.
6. Install only the training extras — Colab already ships TensorFlow, Streamlit and Pillow,
   so installing `requirements.txt` there would try to replace the preinstalled GPU TensorFlow:

   ```python
   !pip install -q scikit-learn matplotlib
   ```
7. Train:

   ```python
   !python train.py --data_dir data/plantvillage
   ```
   (The BPLD dataset images should already be arranged under `data/plantvillage/anthracnose/`, `
    data/plantvillage/healthy/`, etc., as described in step 2.)
8. Download the results: in the left panel right-click `models/` → **Download**, or run:

   ```python
   from google.colab import files
   import os
   for f in os.listdir("models"):
       files.download(os.path.join("models", f))
   ```
9. Copy the downloaded files into your local `models\` folder, commit, push — the deployed
   app on Streamlit Cloud does not care where the model was trained.

> Colab's TensorFlow version may differ slightly from the pinned `tensorflow-cpu==2.21.0`.
> The `.keras` format is stable across Keras 3.x versions, so the model will load on Streamlit
> Cloud. If it ever does not, re-export the model with the same major version the app uses.

---

## 8. Verification status

What was actually run while building this project (so you know what is proven):

- `pip check` → "No broken requirements found" in both `E:\plant-disease-venv` and the project `.venv`
  (`tensorflow-cpu 2.21.0`, `streamlit 1.66.1`, `keras 3.15.1`, `numpy 2.5.4`, `pillow 13.0.0`,
  `scikit-learn 1.9.1`, `matplotlib 3.11.2`)
- `python -m py_compile app.py train.py utils.py` → no syntax errors
- `python train.py` on a tiny **synthetic throwaway dataset** → all 6 steps complete and write
  `plant_disease_mobilenetv2.keras` (~9.6 MB), `class_names.json`, `metrics.json`,
  `classification_report.txt`, `confusion_matrix.png`, `training_curves.png`.
  The accuracy numbers from that run are **meaningless** (random-noise images). Those throwaway
  outputs live only in `.cache\tmp\` (git-ignored) and were never written into `models\` —
  they claim nothing about real plant photos
- The Streamlit app was executed with `streamlit.testing.v1.AppTest` in three scenarios,
  **0 exceptions** in all of them:
  1. **no model present** → shows the "no trained model" warning and the setup steps, no crash
  2. **model present + uploaded leaf image** → image displayed, result card with a confidence
     percentage and the top-N prediction bars
  3. **TensorFlow / Pillow missing** → friendly "Missing Python package" message with the exact
     install command, no stack trace
- Invalid uploads (empty, corrupt, non-image bytes) → friendly error message, no crash
- `streamlit run app.py` → HTTP 200 on `/`, `/healthz` and the static bundle, empty error log,
  with **and** without a trained model present
- Windows long paths enabled (`LongPathsEnabled = 1`), `.venv` rebuilt and TensorFlow installed into it

What has **not** been run: training on the real PlantVillage dataset (it is not downloaded yet).
No accuracy figure for real plant photos exists in this repository, and none is claimed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `[Errno 28] No space left on device` while installing | Make sure the five `set "..."` lines from step 1 are set in *this* window, then re-run the install. Also free space on C: for Windows itself |
| `[Errno 2] No such file or directory: ...site-packages\tensorflow\include\...` | Windows long paths are disabled — run the `reg add` line from step 1 in an **elevated** prompt, delete `.venv`, then re-run step 1 |
| App says "Missing Python package `tensorflow`" | The environment is not active: run `.venv\Scripts\activate`, or reinstall with `python -m pip install -r requirements.txt` |
| `'python' is not recognized` | Install Python 3.12 and tick "Add python.exe to PATH", or use `py -3.12` |
| App says no model found | Train `train.py` and commit the files in `models\` |
| `Model has N outputs but class_names.json lists M` | The two files come from different runs; commit them together |
| Model fails to load after deploy | Train and deploy with the same major TensorFlow/Keras version (pinned to 2.21 in `requirements.txt`) |
| Out of memory while training | `--batch_size 16`, or train on Colab |
| Everything is predicted with low confidence | Use a close, sharp photo of a single leaf |
| `PermissionError` writing to `~/.keras` or `~/.matplotlib` | Set `KERAS_HOME` and `MPLCONFIGDIR` to `.cache\` (step 1) |

## Limitations

- Trained on Blackgram Plant Leaf Disease images (lab-style); field photos with cluttered backgrounds may perform worse.
- The model only knows its training classes and always outputs one of them. Use the confidence
  score and the low-confidence warning; do not treat results as a professional diagnosis.
- This tool supports, but does not replace, advice from an agricultural expert.
