"""Train the Random Forest SECOND model and save it next to the TorchScript ones.

Uses the SAME learned 512-d features and the SAME train/val/test split as the
notebook (same SEED, same stratified split), so its accuracy is directly
comparable with the CNN's. Safe to re-run anytime: nothing here touches the
CNN model or the K-means artifacts.

Run:  .venv/Scripts/python.exe ml/tools/train_randomforest.py
"""
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'
MODEL_DIR = 'E:/auysha_clone/simple-backend/model'
IMG_SIZE = (224, 224)
SEED = 42
VAL_SPLIT, TEST_SPLIT = 0.10, 0.10
BATCH = 64
KEEP = {'cotton', 'linen', 'silk', 'wool', 'cord', 'denim'}
UNKNOWN_CLASS = 'unknown'
UNKNOWN_DIR = 'E:/auysha_clone/ml/datasets/unknown_neg'
UNK_PER_IGNORED = 40
UNK_SYNTH = 160

CLASSES = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(('.', '_'))
                 and d.lower() in KEEP)
assert len(CLASSES) == 6, 'expected the 6 requested classes, found: %r' % CLASSES
FABRIC_CLASSES = list(CLASSES)
CLASSES = FABRIC_CLASSES + [UNKNOWN_CLASS]     # 'unknown' is the LAST (rejection) class
print('classes (%d):' % len(CLASSES), ', '.join(CLASSES))


def build_unknown():
    # Identical to the CNN pipeline (cached on disk, so both share one set).
    d = os.path.join(UNKNOWN_DIR, UNKNOWN_CLASS)
    if os.path.isdir(d) and len(glob.glob(os.path.join(d, '*.*'))) >= (UNK_PER_IGNORED + UNK_SYNTH) * 0.9:
        return d
    os.makedirs(d, exist_ok=True)
    rng = random.Random(SEED)
    npr = np.random.RandomState(SEED)
    ignored = sorted(dd for dd in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, dd)) and not dd.startswith(('.', '_'))
                     and dd.lower() not in KEEP)
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
print('images:', len(paths))

tr_x, tmp_x, tr_y, tmp_y = train_test_split(
    paths, labels, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=labels, random_state=SEED)
va_x, te_x, va_y, te_y = train_test_split(
    tmp_x, tmp_y, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), stratify=tmp_y, random_state=SEED)
print('train %d | val %d | test %d' % (len(tr_x), len(va_x), len(te_x)))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('embedder device:', DEVICE)
EMBED = torch.jit.load(os.path.join(MODEL_DIR, 'fabricnet_embed.ts'), map_location=DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def embed_paths(paths_):
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths_), BATCH):
            xs = []
            for p in paths_[i:i + BATCH]:
                img = Image.open(p).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
                x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                xs.append((x - MEAN) / STD)
            xs = torch.stack(xs).to(DEVICE)
            feats.append(EMBED(xs).cpu().numpy())
    return np.concatenate(feats).astype('float32')


print('computing embeddings for', len(tr_x), 'training photos ...')
tr_feats = embed_paths(tr_x)

rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)
rf.fit(tr_feats, tr_y)

va_feats = embed_paths(va_x)
te_feats = embed_paths(te_x)
rf_val_acc = float(accuracy_score(va_y, rf.predict(va_feats)))
rf_test_acc = float(accuracy_score(te_y, rf.predict(te_feats)))
try:
    with open(os.path.join(MODEL_DIR, 'labels.json'), 'r') as f:
        meta = json.load(f)
    cnn_test_acc = float(meta.get('cnn_test_accuracy', meta.get('test_accuracy', 0.0)))
except Exception:
    cnn_test_acc, meta = 0.0, {}

if meta.get('classes') and list(meta['classes']) != list(CLASSES):
    raise SystemExit(
        'labels.json classes %r do not match this script\'s classes %r --\n'
        'the CNN looks stale; retrain it first (ml/tools/train_cnn6.py).'
        % (meta['classes'], CLASSES))

print('RandomForest  val %.2f%% | test %.2f%%' % (rf_val_acc * 100, rf_test_acc * 100))
print('CNN           test %.2f%%   (higher test accuracy: %s)'
      % (cnn_test_acc * 100, 'Random Forest' if rf_test_acc > cnn_test_acc else 'CNN'))

# ---------------------------------------------------------------------------
# CONFUSION CHECK for the Random Forest (the "no confusion" verification).
# ---------------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

te_pred = rf.predict(te_feats)
cm = confusion_matrix(te_y, te_pred, labels=list(range(len(CLASSES))))
print()
print('RF CONFUSION MATRIX (rows = true, cols = predicted)')
print('%-9s' % '' + ''.join('%9s' % c[:8] for c in CLASSES))
for i, c in enumerate(CLASSES):
    print('%-9s' % c[:8] + ''.join('%9d' % cm[i, j] for j in range(len(CLASSES))))
print(classification_report(te_y, te_pred, target_names=CLASSES, digits=3))
print('PER-CLASS recall:')
for i, c in enumerate(CLASSES):
    rec = cm[i, i] / max(cm[i].sum(), 1)
    others = ', '.join('%s:%d' % (CLASSES[j], cm[i, j])
                       for j in range(len(CLASSES)) if j != i and cm[i, j] > 0) or 'none'
    print('  %-8s recall %6.2f%%  confused with -> %s' % (c, rec * 100, others))
rf_off = int(cm.sum() - np.trace(cm))
print('total RF confusions on test set: %d / %d' % (rf_off, len(te_y)))
if rf_off:
    print('!! CONFUSION PRESENT -- see matrix above; iterate (more trees/data) if needed')

# Which photos would BOTH models get wrong? (worst-case check)
try:
    cnn_pred = []
    CNN = torch.jit.load(os.path.join(MODEL_DIR, 'fabricnet_simple.ts'), map_location=DEVICE).eval()
    with torch.no_grad():
        for i in range(0, len(te_x), BATCH):
            xs = []
            for p in te_x[i:i + BATCH]:
                img = Image.open(p).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
                x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                xs.append((x - MEAN) / STD)
            cnn_pred += list(CNN(torch.stack(xs).to(DEVICE)).argmax(1).cpu().numpy())
    both_wrong = sum(1 for t, cp, rp in zip(te_y, cnn_pred, te_pred) if t != cp and t != rp)
    either_wrong = sum(1 for t, cp, rp in zip(te_y, cnn_pred, te_pred) if t != cp or t != rp)
    print('photos wrong by BOTH models: %d / %d  (wrong by either: %d)'
          % (both_wrong, len(te_y), either_wrong))
except Exception as _e:
    print('both-model cross-check skipped (%s)' % _e)

from joblib import dump
dump(rf, os.path.join(MODEL_DIR, 'randomforest.joblib'))
print('saved RandomForest ->', os.path.join(MODEL_DIR, 'randomforest.joblib'))

meta['cnn_test_accuracy'] = round(cnn_test_acc, 4) if cnn_test_acc else meta.get('cnn_test_accuracy')
meta['rf_test_accuracy'] = round(rf_test_acc, 4)
meta['rf_val_accuracy'] = round(rf_val_acc, 4)
meta['better_model'] = 'randomforest' if rf_test_acc > cnn_test_acc else 'cnn'
with open(os.path.join(MODEL_DIR, 'labels.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print('labels.json updated (better_model = %s)' % meta['better_model'])
