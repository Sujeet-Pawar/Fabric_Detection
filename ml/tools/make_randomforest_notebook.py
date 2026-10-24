'''Generates ml/notebooks/randomforest_training.ipynb (FABRIC IQ second model).

Run from the repo root:  python ml/tools/make_randomforest_notebook.py
Regenerating overwrites the notebook with the same content (idempotent).

Trains the Random Forest SECOND model on the CNN's learned 512-d features.
Requires the main notebook (fabric_classifier_simple.ipynb) to have been run
at least once, so model/fabricnet_embed.ts exists.
'''
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'ml' / 'notebooks' / 'randomforest_training.ipynb'


def md(name, source):
    return {'cell_type': 'markdown', 'id': name, 'metadata': {}, 'source': source}


def code(name, source):
    return {'cell_type': 'code', 'id': name, 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}


CFG = '''# ---------------------------------------------------------------------------
# STEP 1 -- SETUP. Paths, seed, device. Nothing to edit here.
# Requires the MAIN notebook to have been Run All at least once (it saves
# model/fabricnet_embed.ts, the feature extractor used below).
# ---------------------------------------------------------------------------
import glob
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

DATA_DIR = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'
MODEL_DIR = 'E:/auysha_clone/simple-backend/model'
IMG_SIZE = (224, 224)
BATCH = 64

# Extra train-only photos staged by the MAIN notebook (same folder it writes).
# Added to the TRAIN split below so the forest sees the same data as the CNN.
EXTRA_DIR = 'E:/auysha_clone/ml/datasets/extra6'

# UNKNOWN ("not a fabric") rejection class -- the MAIN notebook builds this cache
# folder; it is reused here so both models learn the exact same 7 classes.
UNKNOWN_DIR = 'E:/auysha_clone/ml/datasets/unknown_neg'
UNKNOWN_CLASS = 'unknown'
UNK_PER_IGNORED = 40
UNK_SYNTH = 160

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch', torch.__version__, '| feature extractor device:', DEVICE)

EMBED_PATH = os.path.join(MODEL_DIR, 'fabricnet_embed.ts')
assert os.path.isfile(EMBED_PATH), (
    'fabricnet_embed.ts not found in ' + MODEL_DIR +
    ' -- run the MAIN notebook (fabric_classifier_simple.ipynb, Run All) first.')
EMBED = torch.jit.load(EMBED_PATH, map_location=DEVICE).eval()
CNN = torch.jit.load(os.path.join(MODEL_DIR, 'fabricnet_simple.ts'), map_location=DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
print('loaded fabricnet_embed.ts + fabricnet_simple.ts')
'''

LOAD = '''# ---------------------------------------------------------------------------
# STEP 2 -- LOAD THE DATASET. Only the trained fabrics are used (same
# KEEP_CLASSES filter as the main notebook) so both models stay comparable.
# ---------------------------------------------------------------------------
KEEP_CLASSES = {'cotton', 'linen', 'silk', 'wool', 'cord', 'denim'}
FABRIC_CLASSES = sorted(d for d in os.listdir(DATA_DIR)
                        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(('.', '_'))
                        and d.lower() in KEEP_CLASSES)
assert len(FABRIC_CLASSES) >= 2, 'Need at least 2 class folders in ' + DATA_DIR
CLASSES = FABRIC_CLASSES + [UNKNOWN_CLASS]      # same 7 classes as the main notebook

# Reuse (or, if missing, rebuild) the SAME unknown negatives the main notebook uses.
def build_unknown():
    d = os.path.join(UNKNOWN_DIR, UNKNOWN_CLASS)
    if os.path.isdir(d) and len(glob.glob(os.path.join(d, '*.*'))) >= (UNK_PER_IGNORED + UNK_SYNTH) * 0.9:
        return d
    os.makedirs(d, exist_ok=True)
    rng = random.Random(SEED)
    npr = np.random.RandomState(SEED)
    ignored = sorted(dd for dd in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, dd)) and not dd.startswith(('.', '_'))
                     and dd.lower() not in KEEP_CLASSES)
    for ic in ignored:
        fs = [p for p in glob.glob(os.path.join(DATA_DIR, ic, '**', '*.*'), recursive=True)
              if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        rng.shuffle(fs)
        for j, p in enumerate(fs[:UNK_PER_IGNORED]):
            try:
                Image.open(p).convert('RGB').save(os.path.join(d, 'other_%s_%04d.jpg' % (ic.lower(), j)), quality=90)
            except Exception:
                pass
    for k in range(UNK_SYNTH):
        w, h = int(npr.randint(200, 420)), int(npr.randint(200, 420))
        kind = k % 6
        if kind == 0:
            arr = npr.randint(0, 256, (h, w, 3)).astype('uint8')
        elif kind == 1:
            arr = np.clip(npr.normal(128, 60, (h, w, 3)), 0, 255).astype('uint8')
        elif kind == 2:
            arr = np.empty((h, w, 3), 'uint8'); arr[:] = npr.randint(0, 256, 3)
        elif kind == 3:
            c0, c1 = npr.randint(0, 256, 3), npr.randint(0, 256, 3)
            t = np.linspace(0, 1, w)[None, :, None]
            arr = np.repeat((c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype('uint8'), h, axis=0)
        elif kind == 4:
            arr = np.full((h, w, 3), 240, 'uint8')
            arr[:, ::max(4, int(npr.randint(10, 40)))] = npr.randint(0, 256, 3)
        else:
            bh = max(4, h // int(npr.randint(8, 40))); bw = max(4, w // int(npr.randint(8, 40)))
            small = npr.randint(0, 256, (bh, bw, 3)).astype('uint8')
            arr = np.asarray(Image.fromarray(small).resize((w, h), Image.NEAREST))
        Image.fromarray(arr).save(os.path.join(d, 'synth_%04d.jpg' % k), quality=90)
    return d

unknown_dir = build_unknown()

paths, labels = [], []
for yi, cname in enumerate(FABRIC_CLASSES):
    for root, _, files in os.walk(os.path.join(DATA_DIR, cname)):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                paths.append(os.path.join(root, fn))
                labels.append(yi)
unk_files = [p for p in glob.glob(os.path.join(unknown_dir, '*.*'))
             if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
paths += unk_files
labels += [len(FABRIC_CLASSES)] * len(unk_files)
labels = np.array(labels)
print('classes (%d):' % len(CLASSES), ', '.join(CLASSES))
print('images:', len(paths))
'''

SPLIT = '''# ---------------------------------------------------------------------------
# STEP 3 -- SAME SPLIT AS THE MAIN NOTEBOOK (same SEED, stratified), so the
# Random Forest accuracy is directly comparable with the CNN's.
# ---------------------------------------------------------------------------
tr_x, tmp_x, tr_y, tmp_y = train_test_split(
    paths, list(labels), test_size=0.20, stratify=list(labels), random_state=SEED)
va_x, te_x, va_y, te_y = train_test_split(
    tmp_x, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED)
tr_x, tr_y = list(tr_x), list(tr_y)
print('train %d | val %d | test %d' % (len(tr_x), len(va_x), len(te_x)))

# Add the extra real-world photos (staged by the main notebook) to TRAIN ONLY,
# so the Random Forest uses the same training data as the CNN.
if os.path.isdir(EXTRA_DIR):
    _ex_paths, _ex_y = [], []
    for _ci, _cn in enumerate(CLASSES):
        _fs = [p for p in glob.glob(os.path.join(EXTRA_DIR, _cn, '**', '*.*'), recursive=True)
               if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        _ex_paths += _fs
        _ex_y += [_ci] * len(_fs)
    if _ex_paths:
        tr_x, tr_y = tr_x + _ex_paths, tr_y + list(_ex_y)
        print('extra images added to TRAIN only:', len(_ex_paths), '-> train size', len(tr_x))
'''

EMBED = '''# ---------------------------------------------------------------------------
# STEP 4 -- FEATURES: run every photo through the frozen ResNet18 backbone to
# get its 512-d learned feature vector (console shows progress + speed).
# ---------------------------------------------------------------------------
def embed_paths(paths_):
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths_), BATCH):
            xs = []
            for p in paths_[i:i + BATCH]:
                img = Image.open(p).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
                x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                xs.append((x - MEAN) / STD)
            xs = torch.stack(xs).to(DEVICE)
            feats.append(EMBED(xs).cpu().numpy())
            if (i // BATCH) % 10 == 0:
                print('  embedded %5d / %d' % (min(i + BATCH, len(paths_)), len(paths_)))
    dt = time.time() - t0
    print('  done in %.1fs (%.0f img/s)' % (dt, len(paths_) / dt))
    return np.concatenate(feats).astype('float32')

print('computing embeddings for', len(tr_x), 'training photos ...')
tr_feats = embed_paths(tr_x)
va_feats = embed_paths(va_x)
te_feats = embed_paths(te_x)
print('feature matrix:', tr_feats.shape)
'''

TRAIN = '''# ---------------------------------------------------------------------------
# STEP 5 -- TRAIN THE RANDOM FOREST on those features (console: training time).
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)
t0 = time.time()
rf.fit(tr_feats, tr_y)
print('RandomForest trained in %.1fs (400 trees)' % (time.time() - t0))

rf_val_acc = float(accuracy_score(va_y, rf.predict(va_feats)))
rf_test_acc = float(accuracy_score(te_y, rf.predict(te_feats)))
print('RandomForest  val %.2f%% | test %.2f%%' % (rf_val_acc * 100, rf_test_acc * 100))
'''

COMPARE = '''# ---------------------------------------------------------------------------
# STEP 6 -- CNN vs RANDOM FOREST: which model is more accurate on the test set?
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, 'labels.json')) as f:
    META = json.load(f)
cnn_test_acc = float(META.get('cnn_test_accuracy', META.get('test_accuracy', 0.0)))

print('================ MODEL COMPARISON (test set) ================')
print('  CNN (ResNet18)  test accuracy: %6.2f%%' % (cnn_test_acc * 100))
print('  Random Forest   test accuracy: %6.2f%%' % (rf_test_acc * 100))
better = 'Random Forest' if rf_test_acc > cnn_test_acc else 'CNN (ResNet18)'
print('  -> higher accuracy: %s' % better)
print('At prediction time BOTH models answer and the MORE CONFIDENT one wins;')
print('the app shows which model produced the answer.')

print()
print('Random Forest per-class report:')
print(classification_report(te_y, rf.predict(te_feats), target_names=CLASSES, digits=3))
'''

CONFUSION = '''# ---------------------------------------------------------------------------
# STEP 7 -- CONFUSION MATRIX for the Random Forest on the test set.
# ---------------------------------------------------------------------------
cm = confusion_matrix(te_y, rf.predict(te_feats))
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(CLASSES, fontsize=8)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title('Random Forest -- test confusion matrix')
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
plt.show()
'''

SAVE = '''# ---------------------------------------------------------------------------
# STEP 8 -- SAVE the Random Forest next to the TorchScript models and update
# labels.json with both accuracies + which one wins on accuracy.
# ---------------------------------------------------------------------------
from joblib import dump

dump(rf, os.path.join(MODEL_DIR, 'randomforest.joblib'))

META['cnn_test_accuracy'] = round(cnn_test_acc, 4) if cnn_test_acc else META.get('cnn_test_accuracy')
META['rf_val_accuracy'] = round(rf_val_acc, 4)
META['rf_test_accuracy'] = round(rf_test_acc, 4)
META['better_model'] = 'randomforest' if rf_test_acc > cnn_test_acc else 'cnn'
with open(os.path.join(MODEL_DIR, 'labels.json'), 'w') as f:
    json.dump(META, f, indent=2)

print('saved RandomForest ->', os.path.join(MODEL_DIR, 'randomforest.joblib'))
print('labels.json updated: cnn=%.2f%% | rf=%.2f%% | better_model=%s'
      % (cnn_test_acc * 100, rf_test_acc * 100, META['better_model']))
print('Restart simple-backend/server.py to pick the Random Forest up.')
'''

PREDICT = '''# ---------------------------------------------------------------------------
# STEP 9 -- DEMO: one photo, BOTH models answer, the more confident wins.
# This is exactly what the Flask server does for every uploaded photo.
# NOTE: run the cells ABOVE first (Run All is easiest) -- this demo needs the
# trained forest (STEP 5) and the setup from STEP 1.
# ---------------------------------------------------------------------------
_missing = [n for n in ('CLASSES', 'DATA_DIR', 'IMG_SIZE', 'MEAN', 'STD',
                        'DEVICE', 'CNN', 'EMBED', 'rf') if n not in globals()]
if _missing:
    raise RuntimeError('This demo cell needs the earlier cells. Missing: '
                       + ', '.join(_missing)
                       + ' -- use Run All (or run cells 1..8 in order) first.')

import random

def predict_both(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    x = ((x - MEAN) / STD).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cnn_probs = torch.softmax(CNN(x), dim=1)[0].cpu().numpy()
        feat = EMBED(x).cpu().numpy()
    rf_probs = rf.predict_proba(feat)[0]
    ci, ri = int(np.argmax(cnn_probs)), int(np.argmax(rf_probs))
    winner = 'Random Forest' if rf_probs[ri] > cnn_probs[ci] else 'CNN (ResNet18)'
    true_cls = os.path.relpath(image_path, DATA_DIR).split(os.sep)[0]
    print(os.path.basename(image_path), '   true class ->', true_cls)
    print('  [CNN]           %-12s %5.1f%%' % (CLASSES[ci], cnn_probs[ci] * 100))
    print('  [RandomForest]  %-12s %5.1f%%' % (CLASSES[ri], rf_probs[ri] * 100))
    print('  -> higher confidence: %s  ->  answer shown: %s (%.1f%%)'
          % (winner, CLASSES[ri if winner == 'Random Forest' else ci],
             max(rf_probs[ri], cnn_probs[ci]) * 100))

# Prefer an unseen test photo from STEP 3's split; if that cell was not run,
# fall back to any dataset photo so the demo always works.
if 'te_x' in globals() and len(te_x):
    demo_path = te_x[0]
else:
    _dir = os.path.join(DATA_DIR, random.choice(FABRIC_CLASSES))
    _imgs = [f for f in os.listdir(_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    demo_path = os.path.join(_dir, random.choice(_imgs))
predict_both(demo_path)
'''

cells = [
    md('intro', '''# FABRIC IQ -- Random Forest (second model) Training Notebook

Trains the **Random Forest** on the CNN's learned 512-d features. The CNN
(`fabric_classifier_simple.ipynb`) stays exactly as it is -- this notebook only
**adds** the second model.

At prediction time **both models answer** and the one with the **higher
confidence wins**; the app shows which model produced the answer.

> **Run the main notebook first** (Run All) -- this notebook needs
> `simple-backend/model/fabricnet_embed.ts`, the frozen ResNet18 feature
> extractor it saves. Steps: features -> train -> compare with the CNN ->
> confusion matrix -> save -> demo prediction.
'''),
    md('md-setup', '## 1. Setup -- paths, seed, load the CNN feature extractor'),
    code('setup', CFG),
    md('md-load', '## 2. Load the dataset'),
    code('load', LOAD),
    md('md-split', '## 3. Same split as the main notebook (comparable accuracies)'),
    code('split', SPLIT),
    md('md-embed', '## 4. Compute the 512-d CNN features for every photo'),
    code('embed', EMBED),
    md('md-train', '## 5. Train the Random Forest'),
    code('train', TRAIN),
    md('md-compare', '## 6. CNN vs Random Forest -- which is more accurate?'),
    code('compare', COMPARE),
    md('md-confusion', '## 7. Random Forest confusion matrix'),
    code('confusion', CONFUSION),
    md('md-save', '## 8. Save the model + update labels.json'),
    code('save', SAVE),
    md('md-predict', '## 9. Demo -- both models answer, higher confidence wins'),
    code('predict', PREDICT),
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': '.venv (Python 3.10.11)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('wrote', OUT, '(%d cells)' % len(cells))
