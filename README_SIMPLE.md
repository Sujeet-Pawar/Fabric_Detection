# FABRIC IQ — Simple Track

Photograph a fabric → get the name, a confidence score, what else it could have been,
a **fabric-condition score (0–100)** and **care/preservation tips**.

Six fabrics, each recognised confidently by **two models**: **Cotton, Linen, Silk, Wool,
cord, denim**. Other fabric photos are caught by the unknown-fabric guard instead of guessed.

Three pieces, deliberately simple:

| Piece | Location | What it is |
|---|---|---|
| Training notebook | `ml/notebooks/fabric_classifier_simple.ipynb` | Dataset → augmentation → ResNet18 transfer learning → K-means guard → evaluation, one step per cell |
| Second-model notebook | `ml/notebooks/randomforest_training.ipynb` | Random Forest trained on the CNN's features; the more confident model wins at prediction time |
| Backend | `simple-backend/server.py` | Tiny Flask API (`/api/predict`) — both models answer, test-time augmentation + unknown-fabric guard + quality/care |
| Frontend | `frontend-simple/` | React dropzone UI showing which model won, confidence meter, condition score, care tips, "confused with" alternatives and honest "not sure" cards |

The complex existing `backend/` / `frontend/` / `fabriciq_ml` package are untouched — this is a separate, self-contained track.

---

## Step 1 — Train (the notebook)

Open `ml/notebooks/fabric_classifier_simple.ipynb` and run the cells top to bottom.

### Step 1b — Second model (Random Forest)

After the main notebook has run at least once, open `ml/notebooks/randomforest_training.ipynb`
(Run All). It reuses the trained CNN as a feature extractor, trains a **Random Forest** on the
same 512-d features and the same split, prints a CNN-vs-RF accuracy comparison + confusion
matrix, and saves `model/randomforest.joblib`. At prediction time **both models answer** and
the one with the **higher confidence wins** — the app shows which model produced the answer
and the overall accuracy winner (CNN vs RF test accuracy from `labels.json`).

> ✅ **Kernel:** the **.venv (3.10.11)** kernel works — torch 2.2.2+cu121 (CUDA), sklearn,
> matplotlib, flask. The `torch_cuda (3.11)` kernel works too. TensorFlow is not used:
> the notebook is PyTorch and trains on the RTX 4070 when `USE_GPU = True` (cell 1),
> falling back to CPU automatically.

- **Dataset:** cell 1 points at `E:/auysha_clone/ml/datasets/raw/my_fabric_dataset`.
  The `LOAD` cell keeps only the six trained fabrics via `KEEP_CLASSES`
  (`cotton, linen, silk, wool, cord, denim`) — extra class folders in the raw dataset are
  ignored, so the model is specialised and never confuses the trained fabrics. Folder names
  become the class labels.
- **Data augmentation (cell 4):** tuned for real-world photos — every training image gets
  `AUG_PER_IMAGE` extra copies **saved into the dataset itself**
  (`ml/datasets/fabric_simple/train_aug`): zoom crops onto the weave (up to ~2.2×),
  rotations ±25°, h/v flips, brightness / contrast / saturation / sharpness jitter, hue
  shifts (white balance), mild sensor noise and slight perspective warp. Train split only;
  val/test stay clean for honest metrics.
- **Model (cell 7):** FabricNet = **pretrained ResNet18** backbone (ImageNet features know
  edges, threads and textures already) + a new classification head sized to the six trained
  fabrics. **Staged training:** epochs 1–3
  train only the new head with the backbone frozen, then everything fine-tunes at lr 1e-4.
  This is the main accuracy fix for real-world photos.
- **Live training output (cell 7):** one printed line per epoch with seconds, loss, accuracy,
  val loss, val accuracy and ETA:

  ```
  Epoch  12/25 |############---------|   41.3s | loss   0.5123 | acc  82.14% | val_loss   0.6011 | val_acc  79.38% | ETA   4.8 min
  ```
- **Evaluation (cell 9):** test-set accuracy, full classification report, **per-class
  accuracy** (anything under ~90% is still getting misidentified) and the confusion matrix.
- **K-means (cell 12):** K-means (K=40) over the model's 512-dim learned features:
  1. **Outlier scan** — the farthest training photos are shown in a grid; if one is
     mislabelled/junk, delete it from the dataset and retrain.
  2. **Unknown guard** — the cluster centroids are saved with the model; a photo that lands
     farther than the 99th-percentile distance from *every* centroid is reported as
     **"not one of the trained fabrics"** instead of a confident wrong name.
- **Quality + care (cell 2):** per-fabric **care/preservation tips** (`CARE`) and a
  `score_quality()` heuristic that reads how well-maintained the fabric is (0–100 + issue
  list). The thresholds are **per fabric** so a fabric's own weave — corduroy ribs, denim
  twill — is never mistaken for damage, while smooth fabrics (cotton, silk…) get full
  sensitivity. Tip: (cell 15) prints a demo prediction with its quality + care.
- **Output:** `simple-backend/model/` gets `fabricnet_simple.ts` (classifier),
  `fabricnet_embed.ts` (512-d embedding model for the K-means guard),
  `kmeans_centroids.npy` and `labels.json` (classes, thresholds, test accuracy, care tips).

## Step 2 — Backend

```bash
cd simple-backend
python server.py          # any python with flask + torch; serves http://127.0.0.1:5001
```

The server runs on CPU by design — single-image inference is fast even without a GPU.
It restarts safe: if port 5001 is already taken it exits with the `netstat`/`taskkill` hint.

`POST /api/predict` runs **test-time augmentation** (centre view + zoomed-onto-weave view +
letterboxed view, probabilities averaged) — much steadier on real photos than one squashed
resize — and returns:

```json
{
  "fabric": "denim", "confidence": 0.94,
  "unsure": false,          // best guess below 60% confidence
  "unknown": false,         // K-means: photo far from all trained clusters
  "kmeans_distance": 4.21,
  "alternatives": [{"fabric": "cord", "confidence": 0.04}, ...],
  "quality": {"score": 88, "label": "Excellent",
              "issues": [], "photo": "clear"},
  "care": {"fabric": "denim", "tips": ["Wash cold, inside-out, ...", ...]}
}
```

`quality` is the condition of the fabric in the photo (0–100, label, detected issues such as
creases, discoloration, pilling or fading); `care` lists washing/drying/storing tips for the
reported fabric. Restart the server after re-exporting it from the notebook (`Run All` or
`.venv/Scripts/python.exe ml/tools/export_server.py`).

## Step 3 — Frontend

```bash
cd frontend-simple
npm install        # once
npm run dev        # opens the UI, normally http://localhost:5173
```

Drag a fabric photo in, hit **Identify fabric**. The UI shows the predicted fabric, a
confidence meter, a **fabric-condition score** with any detected issues, **care/preservation
tips** for that fabric, the top-2 alternatives, plus three honest states instead of a
confident lie:

| State | Meaning |
|---|---|
| ✅ Confident match | sure of the answer |
| 🤔 **Not sure** | below 60% confidence — unclear photo or ambiguous fabric |
| 🚫 **Not one of the trained fabrics** | K-means guard: looks unlike the six trained types; add photos of it to the dataset and retrain |

---

## Notes

- **Headless retrain (no notebook):** `ml/tools/train_cnn6.py` trains the CNN on the six
  fabrics (same recipe), evaluates with confusion matrix + per-class recall, builds the
  K-means guard and writes `labels.json` (with care tips); `ml/tools/train_randomforest.py`
  then retrains the Random Forest on the new `fabricnet_embed.ts` and updates
  `labels.json`. Useful for a quick command-line retrain without Jupyter.
- **Timing:** the six-fabric set (1,654 images; ~9.9k with 6× augmentation) trains fast on
  the RTX 4070 — about **15–25 min** for the full 25 epochs (epochs 1–3 are faster — backbone
  frozen). Short on time? Set `EPOCHS = 8` in cell 1. CPU-only would take hours; keep
  `USE_GPU = True`.
- **First run** downloads the ResNet18 ImageNet weights (~45 MB, cached afterwards).
- **Retraining later** (new fabric type, or after deleting outlier photos) = Run All again.
  The augmentation dataset (`ml/datasets/fabric_simple`) is rebuilt from scratch each run.
- The first time you run the notebook in a fresh VS Code session, if it asks about content
  being newer on disk, choose **Revert** (the disk file is the current one).
