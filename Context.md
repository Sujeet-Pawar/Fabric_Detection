# FABRIC IQ — Project Context (memory file)

Last updated: 2026-10-13. Maintained by Buffy (Codebuff). Read this file first to resume work.

---

## 1. WHAT THE APP IS

**FABRIC IQ** — fabric classifier. Upload a photo of fabric → the app returns the fabric
name + confidence + top alternatives, using **TWO models that both answer** (a CNN and a
Random Forest); the more confident one wins and the UI shows which model won. It also
returns a **fabric-condition score (0–100 + issues)** and **care/preservation tips**.

Trained fabrics (6): **Cotton, Linen, Silk, Wool, cord, denim** (`KEEP_CLASSES` filter;
other raw-dataset classes are ignored and caught by the unknown guard).

Honesty layers (so it never confidently lies):
- confidence < 60% → "not sure" (`unsure: true`)
- K-means distance > calibrated threshold → "unknown fabric" (`unknown: true`)

Stack: PyTorch (torch 2.2.2+cu121, RTX 4070 CUDA) + Flask backend (port 5001, numpy-free
by design) + React frontend (`frontend-simple`). Python 3.10.11 in `.venv`.

---

## 2. FULL HISTORY — EVERY USER REQUEST, FROM SCRATCH

| # | User request | What was done |
|---|--------------|---------------|
| 1 | Build a fabric classifier (photo in → name + confidence + alternatives) | ResNet18 transfer-learning CNN ("FabricNet", kept untouched ever since per user), notebook pipeline, Flask server, React frontend |
| 2 | Guard against unknown fabrics | K-means over 512-d embeddings (K=40), calibrated p99 unknown threshold; outlier scan in notebook |
| 3 | Keep CNN as-is, ADD a Random Forest second model; display the higher-accuracy one | RF (400 trees) on the same 512-d features; `better_model` in labels.json + API; frontend winner banner "🏆 Higher confidence on this photo" + overall winner badge |
| 4 | Dedicated .ipynb for the Random Forest too | `ml/notebooks/randomforest_training.ipynb` (9 steps) via generator `ml/tools/make_randomforest_notebook.py` |
| 5 | "Is training done with same data augmentation and epochs?" | Answered: NO — CNN trains on 6× augmented copies for 25 epochs; RF fits once on raw-original embeddings (RF has no epochs). Same split/seed, so accuracies are comparable. RF still won: RF test 99.77% vs CNN 99.55% |
| 6 | "There is the error fix it" (first time, no text) | Diagnosed from system state: backend server was DOWN (port 5001 no listener, old PID dead). Restarted; verified /api/health + /api/predict end-to-end |
| 7 | "There is error in this file" + pasted traceback: `NameError: name 'te_x' is not defined` at `predict_both(te_x[0])` in randomforest_training.ipynb | Root cause: user ran cells out of order (trace showed `In[3]`; `te_x` is defined in Step 3). FIXED the Step 9 demo cell: no longer requires split vars (falls back to any dataset photo, prefers te_x[0] if split ran), true-class read from folder name, friendly RuntimeError guard telling user to Run All. Verified: regenerated, compiled, executed the actual cell against real artifacts |
| 8 | **DONE**: "include few Classes like cotton, lenin(=Linen), Silk, Denim, cord, wool — this much — and these must be identified very confidently without any confusion, made sure with BOTH models. Also add a quality score feature: how good the fabric is maintained + how to preserve/maintain it" | Retrained BOTH models on exactly 6 classes (`KEEP_CLASSES`). CNN test **100.00% (166 imgs), 0 confusions**, mean conf 95%, per-class recall 100%; RF test **100.00%, 0 confusions**, "wrong by BOTH: 0/166". Added a fabric-aware **quality scorer** (`score_quality`: creases/discoloration/pilling/fading/photo-blur → 0–100 + label + issues) and per-fabric **care tips** (`CARE`) → `labels.json['care']` + `/api/predict` (`quality`, `care`) + frontend panels. See §5 for the full record |
| 9 | "it is showing me Not one of the trained fabrics 🚫 but it is trained for denim as well and lenin, wool, silk, cotton is classifying wrong" | **BUG: unknown-guard miscalibrated.** The K-means `unknown_threshold` (18.0066) was calibrated on the *full-resize* embedding, but the server measures the distance on the *80% centre-crop* view — so ordinary trained photos exceeded it and got flagged "not a trained fabric". Diagnosed with `ml/_diag_unknown.py` (trained crop80 7–20, realistic variants up to 24.2; untrained canvas/fleece/rfd/twill overlap 8.7–16.8). FIX: recalibrated in the server's own view — `train_cnn6.py` + notebook KMEANS cell now fit K-means AND set the threshold on the centre-crop view + realistic variants (bg-margin/rotate/bright), `labels.json unknown_threshold` → **27.0**. Verified: bg-margin cotton/denim photos now `unknown:false` |
| 10 | A Linen photo (a low-res screenshot, 646×502) classified as **cord** by both models | Diagnosed: dataset CLEAN (0 Linen↔cord overlap by embedding; cord→Linen nearest 22.0 vs cord-centroid 9.6); the photo is low-detail (edge density 0.019 vs Linen training 0.09–0.11) and its embedding lies nearest a cord photo (10.2) vs Linen (14.2) — an OOD/trained-robustness gap, not label noise. FIX attempted: added **`degrade()`** low-res→upscale + JPEG-screenshot augmentation (notebook generator + train_cnn6.py), regenerated both notebooks, retrained CNN+RF (test **100%/100%, 0 confusions**, unknown_thr→**16.48**). Realworld eval 98.22% (Linen synthetic variants 96→99%). **BUT the actual screenshot STILL classifies as cord (CNN 0.949)** — user declined adding the photo to the linen training set → **kept model as-is**. Full record in §7 |
| 11 | "three more zip files added to the dataset dir, add them so i can retrain, running only the 2 notebooks" | New zips `cotton.zip` (42 ctn screenshots), `final_linen.zip` (78 linen imgs), `wool2.zip` (59 wool screenshots) staged once into `extra6/{Cotton,Linen,Wool}`. Added **`EXTRA_ZIPS`** config + automatic zip-stage step to the notebook SPLIT cell (generator) AND `train_cnn6.py`, so a from-scratch run of `fabric_classifier_simple.ipynb` re-extracts them; the existing per-class `extra6/` merge picks them into TRAIN only. extra6 now = Cotton 42, Linen 78, Wool 59, cord 400, denim 400 → expected train size **2302**. Verified: notebooks regenerate (33 cells, 0 syntax errors), trainer compiles, zip staging + counts confirmed. Not retrained yet (user runs the notebooks). |
| 12 | **IN PROGRESS**: "random images still classified + cotton/linen/silk hard to tell apart" | Added a real **7th `unknown` rejection class** (negatives = 6 ignored raw classes ×40 + 160 synthetic non-fabric = 400, cached in `ml/datasets/unknown_neg/unknown`) to BOTH notebooks + both trainers; K-means still fits **fabric classes only**; server now answers `is_unknown` (model says unknown **or** distance > thr), `best_guess` (closest trained fabric) and `similar` (Cotton/Linen/Silk top-2 gap <0.15); UI shows "Not one of the trained fabrics", the closest trained fabric and a similar-fabrics note, and the trained count excludes `unknown`. Also: **early stopping** `PATIENCE=6`/`MIN_EPOCHS=10`, `/api/health` now returns `fabric_classes`+`unknown_class`. Config iterations: balanced class weights **reverted** (they hurt Cotton), aug-softening **reverted** (hurt Silk), final `UNK_PER_IGNORED=40`/`UNK_SYNTH=160`. **User runs the 2 notebooks themselves** — see §8 for measured results. |

---

## 3. CODEBASE KNOWLEDGE (the map)

### Critical rule — TWO COPIES
- My working root: `E:/auysha_clone` (where session runs)
- The REAL project: `E:/Auysha_Prachi/new_fabriccc`
- **After EVERY change: copy files over, then hash-verify each file.** Both copies byte-identical as of the 6-class + quality/care upgrade (18 files + dist + 12-class backup verified).

### Files
| File | Role |
|------|------|
| `ml/datasets/raw/my_fabric_dataset/` | Raw dataset. **12 folders**, but only 6 are trained (`KEEP_CLASSES`): Cotton 150, Linen 150, Silk 150, Wool 150, cord 571, denim 483 (= 1,654 imgs). Ignored: Abaca 150, Hessian 150, canvas 575, fleece 623, rfd 624, twill 644 — the 6 ignored classes supply 40 images each to the `unknown` negatives |
| `ml/datasets/unknown_neg/unknown/` | Cached `unknown` rejection negatives (400): 6 ignored classes ×40 + 160 synthetic non-fabric (noise/solid/gradient/stripes/blocks/pixelated). Built by `build_unknown()` in the main notebook, RF notebook and both trainers (same SEED → identical set; reused if ≥180 files exist) |
| `ml/datasets/fabric_simple/` | 7-class split + `train_aug/` augmented copies **(output of the MAIN NOTEBOOK; the AUG cell `rmtree`s and rebuilds OUT_DIR every run)** |
| `ml/datasets/fabric6/` | **Same** output dir, but produced by the standalone `train_cnn6.py` (OUT_DIR hardcoded here). Both are equivalent (same data/seed/recipe); the notebook uses `fabric_simple`, the trainer uses `fabric6` |
| `ml/tools/make_simple_notebook.py` | **Generator** → `ml/notebooks/fabric_classifier_simple.ipynb` (33 cells). Contains the FULL recipe as cell strings: CFG (`KEEP_CLASSES` + `UNKNOWN_*` + `PATIENCE`/`MIN_EPOCHS` + `EXTRA_ZIPS`), QUALITY (CARE + score_quality), LOAD (`KEEP_CLASSES` + `build_unknown()` + 7 classes), SPLIT (extra6 merge + tar.gz + `EXTRA_ZIPS` extract), AUG (14× incl. `degrade`), DATASET, BUILD, TRAIN, EVAL, CURVES, SUBSET, KMEANS (streaming, **fabric-only** fit, server-view calibrate), RFCELL, SAVE (saves `care`/`fabric_classes`/`unknown_class`), PREDICT (prints quality+care), SERVER (writes `simple-backend/server.py`) |
| `ml/tools/make_randomforest_notebook.py` | Generator → `ml/notebooks/randomforest_training.ipynb` (19 cells). Same `KEEP_CLASSES` filter + `build_unknown()` + `extra6` merge → 7 classes. Requires main notebook's `fabricnet_embed.ts` |
| `ml/tools/train_cnn6.py` | Standalone 7-class CNN trainer (**OUT_DIR fabric6**, exact recipe + `EXTRA_DIR` merge + **`EXTRA_ZIPS`** auto-stage + `build_unknown()` + streaming K-means on **fabric-only** + `degrade()` aug + **resume-skip**: reuses existing split/aug if `len(tr_x)*AUG_PER_IMAGE` copies exist — delete `fabric6` if you changed the augmentation recipe). `NUM_WORKERS = 0` (Windows spawn) |
| `ml/tools/train_randomforest.py` | Standalone RF trainer (7 classes incl. `unknown`, 400 trees, updates labels.json). Run AFTER train_cnn6.py |
| `ml/tools/use_textilenet.py` | Standalone TextileNet (`fabric.tar.gz`) extractor — denim/corduroy only; `_safe_path` sanitizes Windows-illegal `?` filenames |
| `ml/datasets/extra6/{denim,cord}` | 400 TextileNet image copies each, merged into TRAIN only (extra data for cord/denim) |
| `ml/datasets/extra6/{Cotton,Linen,Wool}` | Staged once from `cotton.zip`(42) / `final_linen.zip`(78) / `wool2.zip`(59) — user's real photos, TRAIN only. Re-extracted automatically by the notebooks/trainer via `EXTRA_ZIPS` |
| `ml/tools/quality_dev.py` | Quality-scorer calibration harness (per-fabric thresholds vs clean dataset photos + synthetic damage) |
| `ml/tools/quality_metrics.py` | Metric-exploration script used while designing the scorer (FFT periodicity, blur gates…) |
| `ml/tools/export_server.py` | Executes the notebook's `export-server` cell to write `simple-backend/server.py` from the template |
| `ml/tools/full_dataset_check.py` | Verifies "no confusion" on ALL 1,654 photos for both models (batched, GPU); prints per-model errors, confusion pairs and both-wrong cases |
| `ml/tools/realworld_eval.py` | Real-world accuracy check: clean swatches + realistic variants (bg-margin, rotate+light, blur, **low-res downsample**) → % correct. Default TEST_DIR `fabric_simple/test`; pass `fabric6/test` for the trainer's split. Baseline before `degrade()`: 98.67%; after: 98.22% (Linen variants 96→99%) |
| `simple-backend/server.py` | Flask, port 5001. **Numpy-free on purpose** (custom `_load_npy_tensor()` .npy reader + `torch.frombuffer`; RF inference via `.tolist()`). Now also has `QUAL_THR` + `score_quality()` + `CARE` and returns `quality` + `care` from `/api/predict`. **NEW**: `is_unknown` (model says unknown **or** dist > thr), `best_guess` (closest trained fabric), `similar` (Cotton/Linen/Silk top-2 gap <0.15); alternatives exclude `unknown`; `/api/health` returns `fabric_classes` + `unknown_class`. Re-export by running the notebook's last cell |
| `simple-backend/model/` | `fabricnet_simple.ts`, `fabricnet_embed.ts` (512-d, backbone fc=Identity), `kmeans_centroids.npy`, `randomforest.joblib`, `labels.json` (**7 classes** = 6 fabrics + `unknown`, plus `fabric_classes`, `unknown_class`, img_size, thresholds, accuracies, better_model, `care`) |
| `simple-backend/model_12class_backup/` | **Backup of the 12-class artifacts made before the 6-class retrain** |
| `frontend-simple/src/App.jsx`, `styles.css` | React UI: upload → result card, winner banner, model scores, overall winner badge, **fabric-condition bar + issues + care-tips panel**; **NEW**: unknown panel ("Not one of the trained fabrics" + closest trained fabric + "similar fabrics" note when Cotton/Linen/Silk look alike); trained count = `fabric_classes.length` (excludes `unknown`). `npm run build` works |
| `README_SIMPLE.md` | Runbook (Step 1, Step 1b RF, retraining notes, quality/care) |
| `Context.md` | THIS file |

### THE EXACT CNN RECIPE (must be replicated in any retrain)
- Filter classes: `sorted(dirs)` minus dot/underscore folders, lowercased ∩ `KEEP_CLASSES` → `FABRIC_CLASSES` (6). `CLASSES = FABRIC_CLASSES + ['unknown']` — the **7th rejection class**, appended last
- `unknown` negatives: `build_unknown()` → `ml/datasets/unknown_neg/unknown/` = the 6 ignored raw classes × `UNK_PER_IGNORED=40` + `UNK_SYNTH=160` synthetic = **400**; deterministic (`random.Random(SEED)` + `np.random.RandomState(SEED)`), rebuilt only if <180 files exist
- Split: `train_test_split(test_size=0.20, stratify, random_state=42)` then val/test `test_size=0.50` → 80/10/10 (over all 7 classes)
- Split copies saved to `OUT_DIR/{train,val,test}/<Class>/*.jpg` (224×224, LANCZOS, q92)
- Augmentation: **14 copies per TRAIN image** to `OUT_DIR/train_aug/` (zoom_in 0.45–0.95, rotate ±25°, h/v flips, brightness 0.60–1.40, contrast 0.70–1.30, saturation 0.60–1.40, sharpness jitter, hue ±18° (p=.5), gaussian noise amp .06 (p=.35), perspective warp, random-backdrop, lighting falloff, soft-blur, **`degrade()` low-res→upscale + JPEG q30–60**) — saved to DISK, q90
- Normalization: ImageNet MEAN=[0.485,0.456,0.406], STD=[0.229,0.224,0.225]
- Model: torchvision `resnet18(weights=IMAGENET1K_V1)`, fc=Identity (512-d embed), head = Dropout(.3)→Linear(512,256)→ReLU→Dropout(.3)→Linear(256,n)
- Train: CE(label_smoothing=0.05); **epochs 1–3 head only @ Adam 1e-3; epoch 4+ unfreeze all @ Adam 1e-4; 25 epochs**; restore best-val-acc weights; batch 64
- Early stopping: **PATIENCE=6, MIN_EPOCHS=10** (`PATIENCE` epochs with no val-accuracy improvement after epoch `MIN_EPOCHS` stops early) — in the notebook TRAIN cell and train_cnn6.py, so retrains typically finish ~epoch 17–19 instead of 25
- Test-time augmentation in SERVER: mean softmax over 3 views (full resize / 80% centre crop / gray-pad square)
- K-means guard: K=40, n_init=10, seed 42, on train embeddings of the **6 fabric classes only** (`fabric_mask = tr_y < len(FABRIC_CLASSES)`); `unknown_threshold` = 99.9th percentile distance over realistic variants of those fabric photos (server-view)
- labels.json: classes (7), `fabric_classes` (6), `unknown_class`, img_size [224,224], test_accuracy, format, normalisation, unsure_threshold 0.60, unknown_threshold, kmeans_k, cnn_test_accuracy, rf_test_accuracy, better_model

### Env quirks / recurring pitfalls
- Server template must stay **numpy-free**; RF inference uses `feat.detach().cpu().reshape(1,-1).tolist()`
- VS Code stale notebook tab: "content is newer" dialog → click **Revert**, never Overwrite; close-without-save, reopen
- Terminal broker: explicit `cwd:` param can break bash spawn (ENOENT) → run without `cwd` (project root is already `/e/auysha_clone`); ripgrep/code_search broken this session → use `grep` in terminal
- Kill port 5001 owner: `netstat -ano | grep ":5001" | grep LISTEN` → `cmd //c "taskkill /F /PID <pid>"`
- Server refuses to start if port busy (friendly SystemExit)
- `/api/predict` expects POST form field **`file`**
- Windows bash: POSIX syntax only; python via `.venv/Scripts/python.exe`
- Notebooks' `DATA_DIR`/`MODEL_DIR` hardcoded to `E:/auysha_clone/...`
- **Background big-training can die right before save** (Start-Process child killed when the launching tool call's job ends — no traceback logged). If retraining, run `train_cnn6.py` in the foreground of ONE call with a long timeout, OR rely on its resume-skip (reuses `fabric6` split+aug if counts match). Server background-start is small/fast and has survived.

---

## 4. CURRENT STATE (at time of writing)

- **Apps stopped per user**: no listener on `:5001`; the user's own Jupyter kernels may still be alive — leave them alone
- **The user is retraining BOTH models themselves**, Run All on `ml/notebooks/fabric_classifier_simple.ipynb` then `ml/notebooks/randomforest_training.ipynb`. The code side of task #12 is **done** and both notebooks regenerate clean (33 / 19 cells, 0 syntax errors); nothing to edit before running
- Notebooks were regenerated after the final config (7 classes, `UNK_PER_IGNORED=40`/`UNK_SYNTH=160`, early stopping, hard aug), so `fabric_simple/` will be rebuilt from scratch by the AUG cell
- `simple-backend/model/` currently holds a mid-iteration CNN only (no `randomforest.joblib`, no `rf_test_accuracy`) — it is overwritten by the notebook runs; `server.py` still needs its last-cell re-export
- `ml/datasets/unknown_neg/unknown/` = 400 negatives, matches current constants → will be reused as-is
- `ml/datasets/fabric6/` (trainer's split) is stale w.r.t. the final config; the trainer's resume-skip only compares counts, so **delete `fabric6` before ever re-running `train_cnn6.py` after an aug change**
- Old verified baselines (6-class, task #10): CNN+RF test 100%, real-world 98.22%, `unknown_threshold` 16.48 — these are the numbers to beat/keep
- All changed files synced byte-identical to the real project `E:/Auysha_Prachi/new_fabriccc`
- `server_run.log`/`server_run.err.log`/`train_rf.log` in `ml/` are throwaway run logs

---

## 5. TASK #8 — COMPLETE (record)

### Decisions made
- **6 classes only**: Cotton, Linen, Silk, Wool (capitalized folders), cord, denim (lowercase folders) — case-insensitive filter `KEEP_CLASSES = {'cotton','linen','silk','wool','cord','denim'}`. Other 6 classes dropped from training (unknown-guard will catch them as "not a trained fabric").
- Dataset for 6 classes = 1,654 imgs (Cotton 150, Linen 150, Silk 150, Wool 150, cord 571, denim 483)
- **Quality score** = deterministic image-condition heuristic (no labeled condition data exists, so no model training possible): wrinkle/crease edge density + stain/dark-blotch ratio + pilling/speckle + fading (saturation) + photo-clarity note → 0–100 score + label + issue list, computed with **torch ops only** (keeps server numpy-free). Thresholds are **per fabric** (cord/denim weave tolerated; smooth fabrics sensitive) so a fabric's own ribs are never called damage.
- **Care/preservation tips** = curated per-fabric knowledge (washing, drying, ironing, storage) stored in `labels.json['care']`, returned by API, rendered in UI
- Retrain BOTH models on the 6 classes with the exact same recipe; verify with confusion matrix + per-class recall + confidence stats for both models

### Results
- **CNN** (train_cnn6.py): test **100.00% (166 imgs)**, **0/166 confusions**, per-class recall 100% all 6, mean confidence 95.0%, KMeans purity 99.2%, unknown_threshold 18.007, labels.json saved with `care`
- **RF** (train_randomforest.py): val 100%, test **100.00%**, **0/166 confusions**, "photos wrong by BOTH models: 0/166"; `better_model: cnn` (tie on 1.0, labels order keeps cnn)
- **Full-dataset check** (`full_dataset_check.py`, all 1,654 photos, single view): **RF 0 errors (0.00%)**; **CNN 6 errors (0.36%)** — all among the white trio (Linen→Silk ×3, Linen→Cotton ×2, Silk→Cotton ×1); **BOTH wrong: 0**. (Under the server's 3-view TTA, 1 borderline photo — Linen_48 — flips to "Silk" by both; the more-confident-wins mechanism otherwise keeps the final answer correct.)
- **Quality scorer calibration** (`quality_dev.py`): clean dataset photos min 91 / med 100 (all 6 classes); synthetic wrinkled ≈50, stained ≈47, pilled ≈72, faded ≈95, blurred 94 (photo note only)
- **API e2e**: clean Cotton photo → `confidence 0.9578`, `quality 100 Excellent` + care tips; synthetic wrinkled+blurred → `quality 49 Worn` + 4 issues, `unsure: true`, `unknown: true`
- Notebooks regenerated (main 33 cells, RF 19 cells), all code cells compile; `server.py` exported via `ml/tools/export_server.py` and compiles
- Frontend rebuilt (`npm run build`); quality bar + issues + care panel added; "12 trained types" replaced with dynamic count

### Notes / gotchas found this task
- Generator string nesting: `SERVER = r'''...'''` cannot contain `'''` (it closes the outer string) — the SERVER_PY `score_quality` docstring was changed to `#` comments. Same for `"""` inside `SERVER_PY = r"""..."""`.
- Windows spawn: `train_cnn6.py` uses `NUM_WORKERS = 0`.
- `Start-Process` server can be killed if the launching shell command is interrupted — start detached, then poll `netstat` in a separate call.

---

## 6. HOW TO RUN EVERYTHING

```bash
# server (from E:/auysha_clone)
.venv/Scripts/python.exe simple-backend/server.py        # port 5001

# retrain CNN (6 classes)
.venv/Scripts/python.exe ml/tools/train_cnn6.py

# retrain RF (AFTER CNN — needs new fabricnet_embed.ts)
.venv/Scripts/python.exe ml/tools/train_randomforest.py

# regenerate notebooks
.venv/Scripts/python.exe ml/tools/make_simple_notebook.py
.venv/Scripts/python.exe ml/tools/make_randomforest_notebook.py

# frontend
cd frontend-simple && npm run build
```

---

## 7. TASK #10 — LINEN CLASSIFIED AS CORD (record)

### What the user reported
- A **Linen** photo (`C:\Users\Asus\Pictures\Screenshots\Screenshot 2026-10-12 181347.png`, 646×502 screenshot) classified as **cord** on both the frontend and the API.

### Diagnosis (what was actually true)
- **Dataset is clean**: centroid-based 1-NN across all 1,654 raw photos → no Linen↔cord confusion anywhere (0 cord imgs closer to Linen than cord; 0 Linen imgs closer to cord than Linen; cord's nearest-to-Linen photo is still 22.0 from Linen centroid vs 9.6 from its own).
- **The photo is out-of-distribution, not mislabelled**: object-level stats — edge density **0.019** vs Linen training 0.09–0.11, cord 0.03, denim 0.03; 646×502 vs 2560×1440 training; 9% dark pixels vs ~0%. Its embedding sits nearest a cord training photo (dist **10.2**) vs nearest Linen (14.2), on both the old and the retrained model.
- Both models agree (CNN 0.907→0.949, RF 0.83→0.81); kmeans_distance ~8.4–9.9 → never flagged unknown/unsure.

### Fix attempted
- Added **`degrade()`** to the augmentation (both `make_simple_notebook.py` AUG cell and `train_cnn6.py`): random downscale to 10–45%, upscale back, optional JPEG q30–60 — teaching the model that low-res/screenshot captures are still their class. Also widened soft-blur (to 2.2σ). Regenerated both notebooks (33/19 cells, 0 syntax errors, `degrade` in the pipeline). `realworld_eval.py` gained a low-res variant for measurement.
- Retrained CNN (29,722 augmented copies, foreground, ~50 min) + RF: **test 100%, 0 confusions each**, unknown_thr → **16.48**.

### Result
- Real-world eval (450 variants): 98.22%, with Linen synthetic variants **96→99%** (Cotton/Wool dipped 97–100 → a wash).
- **The actual screenshot STILL classifies as cord** (CNN 0.949, RF 0.808, dist 8.4) — even more confident. The model family genuinely reads this photo as cord; only adding it (or similar linen) to training would change that.

### Decision (user)
- Chose **"Keep current model as-is"** — did NOT add the photo to Linen training. End-state is accepted: this specific low-res photo returns cord; the honesty layers can't catch it because the model is genuinely confident.

### Gotchas learned this task
- `degrade()` did move the *generated* low-res variants (Linen 96→99%) but NOT the real screenshot — an augmentation can't conjure an unseen linen texture class.
- `realworld_eval.py`'s default TEST_DIR was stale (`fabric6/test` didn't exist) — now defaults to `fabric_simple/test`; pass `fabric6/test` to test the trainer's split.
- Background (Start-Process) CNN training died **silently right before model save** (no traceback, model files untouched) — reran `train_cnn6.py` in the foreground of a single tool call with a long timeout and it completed. Added **resume-skip** so re-runs reuse the existing split+augmented dir instead of rebuilding it.
- Standalone trainer writes to `fabric6/`, the notebook to `fabric_simple/` — same data/seed/recipe, cosmetic discrepancy only.

---

## 8. TASK #12 — UNKNOWN REJECTION + TRIO (record, implementation complete)

### Problem (as reported)
1. Random / untrained photos still get a confident class name.
2. Cotton / Linen / Silk are hard to tell apart.

### Evidence gathered before changing anything
Live inference over every raw class + synthetic junk (old 6-class model, THR 16.48):
- trained dist medians 4.0–7.3 (conf .83–.95); untrained Abaca 12.4, Hessian 11.6, canvas 10.4, twill 10.0, rfd 9.5, fleece 8.1 (.51–.89)
- synthetic noise 11.2/.76, gray 11.8/.78, solid 11.2/.91 — **none flagged unknown**

=> both the K-means distance guard and the confidence flag are structurally blind here (overlapping ranges). The fix had to be an explicit rejection class, not a threshold tweak.

### What was implemented (code final; both notebooks regenerated, 0 syntax errors)
- **7th class `unknown`**, appended last: `CLASSES = FABRIC_CLASSES + ['unknown']`. `build_unknown()` exists in the main notebook, the RF notebook, `train_cnn6.py` and `train_randomforest.py` (same SEED) so all four cache the identical set in `ml/datasets/unknown_neg/unknown/` = the 6 ignored raw classes x `UNK_PER_IGNORED` + `UNK_SYNTH` synthetic.
  - Final config **`UNK_PER_IGNORED=40`, `UNK_SYNTH=160` = 400 negatives** (660 was tried and dropped — see trials).
- K-means still fits **only the fabric classes** (`fabric_mask`), so the distance guard stays an OOD check instead of being polluted by junk. Also fixed a latent bug: the notebook's KMEANS cell indexed a list (`tr_y`) that only the trainer converted — it now does `np.asarray`.
- **Server** (template lives in the generator's `export-server` cell): `is_unknown = (fabric == unknown) OR (dist > thr)`; adds `best_guess` (most probable trained fabric, so the UI shows a closest match instead of a bare name), `similar` (top-2 both in `{Cotton, Linen, Silk}` and gap < 0.15), `alternatives` now exclude `unknown`; `/api/health` exposes `fabric_classes` + `unknown_class`.
- **UI** (`frontend-simple/src/App.jsx` + `styles.css`): unknown panel "Not one of the trained fabrics" + "Closest trained fabric: X (n%)" + a "near-identical smooth weaves" note when `similar` is set; trained count = `fabric_classes.length` (6). `npm run build` passes.
- **Early stopping** `PATIENCE=6` / `MIN_EPOCHS=10`, and augmentation is back on the **hard** recipe (soft-blur 0.4–2.2, degrade 0.10–0.45).
- Balanced class weights were tried and **removed** (see trials).

### Measurements from the three CNN trial runs (train_cnn6.py, same split, all else equal)

| trial | unknown count | aug | class weights | CNN test | real-world | unknown recall (held-out) |
|---|---|---|---|---|---|---|
| A | 600 | soft | no | 98.23% | 95.62% (Silk 84) | 60/60 |
| B | 600 | hard | yes | 97.79% | 95.05% (Cotton 79) | 58/60, Silk 100 |
| C (final) | 400 | hard | no | 98.54% | not yet measured | 40/40 |

- C's 3 test confusions: Cotton->Linen 1, Cotton->cord 1, Linen->Cotton 1; `unknown_threshold` 16.534; K-means purity 96.8%.
- Trial B proved **balanced class weights hurt** (Cotton real-world 95% -> 79%) — they are NOT in the final recipe.
- Trial A proved softening the augmentation hurt robustness (Silk 84%) — reverted to the task #10 recipe, which scored 98.22% real-world on the 6-class model.
- **Open question for the user's run:** does the 7-class model hold the 6 fabrics near that 98.22% baseline? A trade-off exists — adding the class demonstrably cost ~2-3% real-world in trials A/B. Watch Cotton and the real-world eval after Run All.

### Follow-ups after the user's notebook run
1. `ml/tools/realworld_eval.py ml/datasets/fabric_simple/test` (or `fabric6/test`) — compare vs the 98.22% baseline.
2. `ml/tools/full_dataset_check.py` — confirm no fabric-vs-fabric confusion on all 1,654 photos.
3. Live `/api/predict`: a noise image, an ignored-class image (canvas/fleece), the old Linen screenshot, and a clean Cotton photo — expect `is_unknown: true` + `best_guess` for the first three, a normal answer for the last.
4. Re-export the server (the notebook's last cell) and restart it; frontend `dist/` is already built.
5. If real-world fabric accuracy stays below ~97%, the next lever is a **separate fabric-vs-not gate** (keeps the 6-class head intact) instead of a 7th softmax class.
