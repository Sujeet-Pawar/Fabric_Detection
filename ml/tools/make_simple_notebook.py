'''Generates ml/notebooks/fabric_classifier_simple.ipynb (FABRIC IQ simple track).

Run from the repo root:  python ml/tools/make_simple_notebook.py
Regenerating overwrites the notebook with the same content (idempotent).

This version trains on GPU with PyTorch (torch_cuda env). TensorFlow cannot use
the GPU on native Windows for this setup, so the notebook is PyTorch-based.
'''
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'ml' / 'notebooks' / 'fabric_classifier_simple.ipynb'
G = 'E:/auysha_clone'


def md(name, source):
    return {'cell_type': 'markdown', 'id': name, 'metadata': {}, 'source': source}


def code(name, source):
    return {'cell_type': 'code', 'id': name, 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}


CFG = f'''import glob
import io
import json
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================================================
# CONFIG -- THE DATASET IS THIS FOLDER, nothing else needed:
# ===========================================================================
DATA_DIR = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'

# Where outputs go (kept near the project; no need to touch these):
OUT_DIR = 'E:/auysha_clone/ml/datasets/fabric_simple'    # split + augmented copies
MODEL_DIR = 'E:/auysha_clone/simple-backend/model'       # where the trained model is saved

# Optional EXTRA real-world photos (e.g. the TextileNet-fabric archive). If the
# archive exists and its usable classes are not staged yet, the notebook extracts
# ONLY the classes it can reuse and adds them to the TRAINING split (val/test stay
# clean swatches). Leave EXTRA_ARCHIVE = '' to train on your own dataset only.
EXTRA_ARCHIVE = 'E:/auysha_clone/ml/datasets/fabric.tar.gz'
EXTRA_DIR = 'E:/auysha_clone/ml/datasets/extra6'
EXTRA_MAP = {{'denim': 'denim', 'corduroy': 'cord'}}     # archive class -> our class
EXTRA_CAP = 400                                         # max extra images per class

# Optional ZIP archives of extra real-world photos (flat folders, one per class).
# Each entry is (zip_path, target_class_folder). The zip is extracted into
# EXTRA_DIR/<class> once; the TRAIN-split merge below then picks it up. Leave the
# list empty to train on your own dataset only.
EXTRA_ZIPS = [
    ('E:/auysha_clone/ml/datasets/cotton.zip', 'Cotton'),
    ('E:/auysha_clone/ml/datasets/final_linen.zip', 'Linen'),
    ('E:/auysha_clone/ml/datasets/wool2.zip', 'Wool'),
]
MAX_REAL_PER_CLASS = 400                                 # cap for zip images per class

# UNKNOWN / "not a fabric" class -- teaches the model to REJECT photos that are
# not one of the trained fabrics (real "other textile" photos from the raw
# classes we do NOT train on + synthetic non-fabric images) instead of
# confidently mislabelling them. Negatives are built once into UNKNOWN_DIR and
# shared by the CNN notebook and the RandomForest notebook.
UNKNOWN_CLASS = 'unknown'
UNKNOWN_DIR = 'E:/auysha_clone/ml/datasets/unknown_neg'
UNK_PER_IGNORED = 40      # real "other textile" photos sampled per ignored raw class
UNK_SYNTH = 160            # synthetic non-fabric images (noise / solid / gradient / ...)
IMG_SIZE = (224, 224)      # (height, width)
BATCH_SIZE = 64            # 8 GB GPU handles this comfortably at 224x224
VAL_SPLIT, TEST_SPLIT = 0.10, 0.10
AUG_PER_IMAGE = 14          # extra augmented copies per training image
EPOCHS = 25
PATIENCE = 6            # epochs of NO val-accuracy improvement before early stop
MIN_EPOCHS = 10         # never stop before this epoch (new head always gets a full fine-tune window)
NUM_WORKERS = 0            # DataLoader workers; keep 0 in Jupyter on Windows, 2-4 if run as a script

# DEVICE SELECTION -- set USE_GPU = False to force CPU training.
# With USE_GPU = True the model trains on the GPU when one is visible and
# silently falls back to CPU otherwise, so the notebook always runs.
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('PyTorch', torch.__version__)
print('Training device:', DEVICE, end=' ')
if DEVICE.type == 'cuda':
    print('|', torch.cuda.get_device_name(0))
else:
    print('(CPU -- set USE_GPU = True to train on the GPU if one is available)')
print('DATA_DIR  =', DATA_DIR)
print('MODEL_DIR =', MODEL_DIR)
if not os.path.isdir(DATA_DIR):
    raise SystemExit('Dataset folder not found: ' + DATA_DIR)
'''

LOAD = '''# ---------------------------------------------------------------------------
# LOAD -- only the TRAINED fabrics are used. The raw dataset may contain extra
# class folders; KEEP_CLASSES filters them out so the model is specialised and
# every trained fabric is recognised with high confidence, no confusion.
# ---------------------------------------------------------------------------
KEEP_CLASSES = {'cotton', 'linen', 'silk', 'wool', 'cord', 'denim'}
FABRIC_CLASSES = sorted(d for d in os.listdir(DATA_DIR)
                        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(('.', '_'))
                        and d.lower() in KEEP_CLASSES)
assert len(FABRIC_CLASSES) >= 2, 'Need at least 2 class folders in ' + DATA_DIR + ' (found: ' + ', '.join(FABRIC_CLASSES) + ')'
CLASSES = FABRIC_CLASSES + [UNKNOWN_CLASS]      # 'unknown' is the LAST (rejection) class
print('Found', len(CLASSES), 'classes ->', CLASSES)

# Build the UNKNOWN ("not a fabric") negatives ONCE and cache them on disk so the
# RandomForest notebook reuses the exact same set. Two sources:
#   1. real "other textile" photos from the raw classes we do NOT train on
#      (canvas, fleece, twill, ... -- genuinely not one of the 6 trained fabrics), and
#   2. synthetic non-fabric images (noise / solid / gradient / stripes / blocks).
# This turns "is this even a trained fabric?" into a normal prediction instead of
# a fragile distance threshold.
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
        if kind == 0:                                   # uniform noise
            arr = npr.randint(0, 256, (h, w, 3)).astype('uint8')
        elif kind == 1:                                 # gaussian grain
            arr = np.clip(npr.normal(128, 60, (h, w, 3)), 0, 255).astype('uint8')
        elif kind == 2:                                 # flat solid colour
            arr = np.empty((h, w, 3), 'uint8'); arr[:] = npr.randint(0, 256, 3)
        elif kind == 3:                                 # linear gradient
            c0, c1 = npr.randint(0, 256, 3), npr.randint(0, 256, 3)
            t = np.linspace(0, 1, w)[None, :, None]
            arr = np.repeat((c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype('uint8'), h, axis=0)
        elif kind == 4:                                 # stripe pattern
            arr = np.full((h, w, 3), 240, 'uint8')
            arr[:, ::max(4, int(npr.randint(10, 40)))] = npr.randint(0, 256, 3)
        else:                                           # coarse random blocks
            bh = max(4, h // int(npr.randint(8, 40))); bw = max(4, w // int(npr.randint(8, 40)))
            small = npr.randint(0, 256, (bh, bw, 3)).astype('uint8')
            arr = np.asarray(Image.fromarray(small).resize((w, h), Image.NEAREST))
        Image.fromarray(arr).save(os.path.join(d, 'synth_%04d.jpg' % k), quality=90)
    return d

unknown_dir = build_unknown()

paths, labels = [], []
for idx, cls in enumerate(FABRIC_CLASSES):
    files = glob.glob(os.path.join(DATA_DIR, cls, '**', '*.*'), recursive=True)
    files = [p for p in files if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    paths += files
    labels += [idx] * len(files)
unk_files = [p for p in glob.glob(os.path.join(unknown_dir, '*.*'))
             if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
paths += unk_files
labels += [len(FABRIC_CLASSES)] * len(unk_files)        # label index of 'unknown'
paths, labels = np.array(paths), np.array(labels)
print('Total images:', len(paths))
for i, c in enumerate(CLASSES):
    print('  %-14s %5d' % (c, int((labels == i).sum())))

# One sample image per class, for a quick visual sanity check.
ncols = min(len(CLASSES), 6)
nrows = (len(CLASSES) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
for ax in np.atleast_1d(axes).flat[len(CLASSES):]:
    ax.axis('off')
for ax, i in zip(np.atleast_1d(axes).flat, range(len(CLASSES))):
    ax.imshow(Image.open(paths[labels == i][0]).convert('RGB').resize(IMG_SIZE))
    ax.set_title(CLASSES[i], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()
'''

SPLIT = '''tr_x, tmp_x, tr_y, tmp_y = train_test_split(
    paths, labels, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=labels, random_state=SEED)
va_x, te_x, va_y, te_y = train_test_split(
    tmp_x, tmp_y, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), stratify=tmp_y, random_state=SEED)
tr_x, tr_y = list(tr_x), list(tr_y)
print('train %d | val %d | test %d' % (len(tr_x), len(va_x), len(te_x)))

# ---------------------------------------------------------------------------
# OPTIONAL EXTRA REAL-WORLD PHOTOS (runs only if the archive/extra dir exists).
# Extracts just the classes we reuse from the archive, then adds them to the
# TRAINING split only -- val/test stay clean swatches so metrics stay comparable.
# Safe no-op when EXTRA_ARCHIVE is '' and EXTRA_DIR is missing.
# ---------------------------------------------------------------------------
if (EXTRA_ARCHIVE and os.path.isfile(EXTRA_ARCHIVE)) or os.path.isdir(EXTRA_DIR):
    import shutil as _sh
    import tarfile as _tar

    def _safe_path(name):
        bad = '<>:"|?*\\n\\r\\t'
        parts = [''.join('_' if ch in bad else ch for ch in p).rstrip('. ')
                 for p in name.split('/')]
        return os.path.join(*[p for p in parts if p not in ('', '.')])

    _staged = all(os.path.isdir(os.path.join(EXTRA_DIR, d)) and
                  glob.glob(os.path.join(EXTRA_DIR, d, '*')) for d in EXTRA_MAP.values())
    if not _staged and EXTRA_ARCHIVE and os.path.isfile(EXTRA_ARCHIVE):
        _stage = os.path.join(os.path.dirname(EXTRA_DIR), '_tn_stage')
        os.makedirs(_stage, exist_ok=True)
        _wanted = tuple('./fabric/train/%s/' % s for s in EXTRA_MAP)
        print('extracting', list(EXTRA_MAP), 'from', os.path.basename(EXTRA_ARCHIVE), '...')
        with _tar.open(EXTRA_ARCHIVE, 'r:gz') as _tf:
            for _m in _tf:
                if _m.isfile() and any(_m.name.startswith(w) for w in _wanted):
                    _t = os.path.join(_stage, _safe_path(_m.name))
                    os.makedirs(os.path.dirname(_t), exist_ok=True)
                    _src = _tf.extractfile(_m)
                    if _src is not None:
                        with open(_t, 'wb') as _o:
                            _sh.copyfileobj(_src, _o)
        for _s, _d in EXTRA_MAP.items():
            _fs = [p for p in glob.glob(os.path.join(_stage, 'fabric', 'train', _s, '*'))
                   if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            random.shuffle(_fs)
            _out = os.path.join(EXTRA_DIR, _d)
            if os.path.isdir(_out):
                _sh.rmtree(_out)
            os.makedirs(_out, exist_ok=True)
            for _i, _p in enumerate(_fs[:EXTRA_CAP]):
                _sh.copy2(_p, os.path.join(_out, 'tn_%s_%04d%s' % (_d, _i, os.path.splitext(_p)[1].lower())))
        _sh.rmtree(_stage, ignore_errors=True)

    # Extra ZIP archives of the user's own real-world photos -> EXTRA_DIR/<class>.
    import zipfile as _zipf
    for _zip, _cls in EXTRA_ZIPS:
        if not os.path.isfile(_zip):
            continue
        _out = os.path.join(EXTRA_DIR, _cls)
        if os.path.isdir(_out) and glob.glob(os.path.join(_out, '*.*')):
            continue
        print('extracting', os.path.basename(_zip), '->', _cls, '...')
        os.makedirs(_out, exist_ok=True)
        _views = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        with _zipf.ZipFile(_zip) as _zf:
            _sel = [_m for _m in _zf.infolist()
                    if not _m.is_dir() and os.path.splitext(_m.filename)[1].lower() in _views]
            random.shuffle(_sel)
            for _i, _m in enumerate(_sel[:MAX_REAL_PER_CLASS]):
                _name = os.path.basename(_safe_path(_m.filename)) or ('img_%04d' % _i)
                _dst = os.path.join(_out, 'us_%s_%04d_%s' % (_cls.lower(), _i, _name))
                with _zf.open(_m) as _src, open(_dst, 'wb') as _o:
                    _sh.copyfileobj(_src, _o)

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

AUG = '''# ---------------------------------------------------------------------------
# DATA AUGMENTATION -- tuned for REAL-WORLD photos (this is the fix for the
# misidentified outside-fabric photo). Every training image is copied
# AUG_PER_IMAGE times INTO THE DATASET ITSELF (OUT_DIR/train_aug on disk) with:
#   - random zoom crops onto the weave (up to ~2.2x, was ~1.8x)
#   - rotations up to 25 deg + horizontal/vertical flips
#   - brightness / contrast / saturation / sharpness jitter (wider range:
#     lamp light, daylight, shade)
#   - hue shifts (different white balance / dye rendering)
#   - mild gaussian noise (phone-camera sensor grain)
#   - slight perspective warp (photo taken at an angle)
#   - low-resolution "degrade" pass (downscale+upscale, optional JPEG) so
#     screenshots/thumbnails/reposts are still recognised (fixes low-detail
#     linen being mistaken for cord)
# Copies are SAVED to disk, so the augmented dataset persists next to the raw
# one. Only the TRAIN split is augmented; val/test stay clean for honest
# metrics. Raise AUG_PER_IMAGE in the config cell for even more copies.
# ---------------------------------------------------------------------------
def zoom_in(img, zmin=0.45, zmax=0.95):
    # Crop a random centre-ish window (zoom onto the weave), then resize back.
    w, h = img.size
    z = random.uniform(zmin, zmax)
    cw, ch = max(8, int(w * z)), max(8, int(h * z))
    span_x, span_y = max(1, (w - cw) // 3), max(1, (h - ch) // 3)
    x0 = max(0, min((w - cw) // 2 + random.randint(-span_x, span_x), w - cw))
    y0 = max(0, min((h - ch) // 2 + random.randint(-span_y, span_y), h - ch))
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)

def rotate(img, max_deg=25):
    return img.rotate(random.uniform(-max_deg, max_deg), resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def flips(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def color_jitter(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.60, 1.40))   # lamp / daylight / shade
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.70, 1.30))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.60, 1.40))        # saturation
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.70, 1.30))
    return img

def hue_shift(img, max_deg=18):
    # Small hue rotation -- different white balance / dye rendering.
    hsv = np.array(img.convert('HSV'), dtype=np.uint8)  # np.array -> writable copy (np.asarray can be read-only)
    shift = int(random.uniform(-max_deg, max_deg) * 255.0 / 360.0)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 256).astype(np.uint8)
    return Image.fromarray(hsv, 'HSV').convert('RGB')

def add_noise(img, amp=0.06):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr += np.random.normal(0.0, amp, arr.shape)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))

def perspective(img, max_off=0.06):
    # Slight keystone, like a phone photo taken at an angle.
    w, h = img.size
    ox, oy = max(1, int(w * max_off)), max(1, int(h * max_off))
    quad = (random.randint(0, ox), random.randint(0, oy),
            random.randint(0, ox), h - random.randint(0, oy),
            w - random.randint(0, ox), h - random.randint(0, oy),
            w - random.randint(0, ox), random.randint(0, oy))
    return img.transform((w, h), Image.QUAD, quad, Image.BILINEAR)

# -- extra domain randomization so real phone photos (fabric on a table, uneven
# -- light, slight shake) look like data the model has already seen ------------
BG_COLORS = [(20, 20, 20), (240, 240, 240), (90, 70, 50), (140, 120, 90),
             (60, 80, 60), (200, 180, 150), (120, 120, 130), (30, 40, 60),
             (180, 60, 50), (220, 210, 190), (40, 40, 40), (160, 150, 140)]

def make_background(size):
    w, h = size
    mode = random.random()
    if mode < 0.4:
        base = Image.new('RGB', (w, h), random.choice(BG_COLORS))
    elif mode < 0.7:
        c1, c2 = random.choice(BG_COLORS), random.choice(BG_COLORS)
        col = np.zeros((h, 1, 3), np.uint8)
        for y in range(h):
            t = y / max(1, h - 1)
            col[y, 0] = [int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)]
        base = Image.fromarray(np.repeat(col, w, axis=1))
    else:
        sw, sh = max(2, w // 32), max(2, h // 32)
        small = Image.new('RGB', (sw, sh))
        small.putdata([tuple(random.randint(40, 220) for _ in range(3)) for _ in range(sw * sh)])
        base = small.resize((w, h), Image.BILINEAR)
    arr = np.asarray(base, dtype=np.float32) / 255.0
    arr += np.random.normal(0.0, 0.02, arr.shape)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))

def random_background(img, p=0.6):
    # fabric lying on a random backdrop at a random scale/offset
    if random.random() > p:
        return img
    w, h = img.size
    s = random.uniform(0.55, 0.92)
    nw, nh = max(16, int(w * s)), max(16, int(h * s))
    fab = img.resize((nw, nh), Image.LANCZOS)
    if random.random() < 0.5:
        fab = perspective(fab, max_off=0.10)
    bg = make_background((w, h))
    bg.paste(fab, (random.randint(0, max(0, w - nw)), random.randint(0, max(0, h - nh))))
    return bg

def lighting(img, p=0.5):
    # directional/radial light falloff -- phone photos are rarely flat-lit
    if random.random() > p:
        return img
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
    d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    v = np.clip(1.15 - 0.9 * d, 0.45, 1.35)
    arr = np.asarray(img, dtype=np.float32) * v[..., None]
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))

def soften(img, p=0.4):
    # slight focus/shake softness (kept light so fabric texture survives -- the
    # white trio cotton/linen/silk differ mostly in fine weave, which heavy blur erases)
    if random.random() > p:
        return img
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.2)))

def degrade(img, p=0.45):
    # Simulate LOW-RESOLUTION captures (screenshots, thumbnails, downscaled
    # exports/reposts). Hard downscale then upscale destroys fine weave detail,
    # which previously made a low-detail LINEN photo drift toward the smoother
    # cord class. Optional JPEG recompression adds screenshot-style artifacts.
    # Floor raised to 0.25 so the weave of cotton/linen/silk is not fully erased.
    if random.random() > p:
        return img
    w, h = img.size
    f = random.uniform(0.10, 0.45)
    small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
    out = small.resize((w, h), Image.BILINEAR)
    if random.random() < 0.5:
        buf = io.BytesIO()
        out.save(buf, 'JPEG', quality=random.randint(30, 60))
        buf.seek(0)
        out = Image.open(buf).convert('RGB')
    return out

def make_split_copy(files, labels_, split):
    # Pre-resize val/test copies to IMG_SIZE so decoding is fast and every
    # split goes through the same 224x224 pipeline as the augmented copies.
    for src, y in zip(files, labels_):
        dst_dir = os.path.join(OUT_DIR, split, CLASSES[y])
        os.makedirs(dst_dir, exist_ok=True)
        try:
            img = Image.open(src).convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
            img.save(os.path.join(dst_dir, os.path.splitext(os.path.basename(src))[0] + '.jpg'), quality=92)
        except Exception:
            pass

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
make_split_copy(tr_x, tr_y, 'train')
make_split_copy(va_x, va_y, 'val')
make_split_copy(te_x, te_y, 'test')

t0, made = time.time(), 0
for src, y in zip(tr_x, tr_y):
    base = os.path.splitext(os.path.basename(src))[0]
    dst_dir = os.path.join(OUT_DIR, 'train_aug', CLASSES[y])
    os.makedirs(dst_dir, exist_ok=True)
    try:
        orig = Image.open(src).convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
    except Exception:
        continue
    for k in range(AUG_PER_IMAGE):
        img = perspective(zoom_in(orig) if random.random() < 0.85 else orig)
        img = rotate(flips(img))
        img = random_background(img)
        img = color_jitter(img)
        if random.random() < 0.5:
            img = hue_shift(img)
        img = lighting(img)
        if random.random() < 0.35:
            img = add_noise(img)
        img = soften(img)
        img = degrade(img)
        img.save(os.path.join(dst_dir, base + '_aug' + str(k) + '.jpg'), quality=90)
        made += 1

def count_saved(sub):
    return sum(len(glob.glob(os.path.join(OUT_DIR, sub, c, '**', '*.*'), recursive=True)) for c in CLASSES)

print('augmented copies written:', made, 'in', round(time.time() - t0, 1), 's')
print('dataset (with augmentation saved in it) ->', OUT_DIR)
print('  train %d | train_aug %d (augmented) | val %d | test %d'
      % (count_saved('train'), count_saved('train_aug'), count_saved('val'), count_saved('test')))
'''

DATASET = '''# ---------------------------------------------------------------------------
# TORCH DATASET + DATALOADER -- images are normalised with the same 0..1 scale
# the Keras version used, so the exported Flask server stays unchanged.
# ---------------------------------------------------------------------------
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class FabricDataset(Dataset):
    def __init__(self, files, labels_):
        self.files, self.labels = list(files), list(labels_)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - MEAN) / STD
        return x, self.labels[i]

def make_loader(sub, shuffle):
    root = os.path.join(OUT_DIR, sub)
    files, ys = [], []
    for i, c in enumerate(CLASSES):
        found = glob.glob(os.path.join(root, c, '**', '*.*'), recursive=True)
        files += [p for p in found if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        ys += [i] * len(found)
    return DataLoader(FabricDataset(files, ys), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
                      persistent_workers=(NUM_WORKERS > 0))

train_loader = make_loader('train_aug', True)
val_loader = make_loader('val', False)
test_loader = make_loader('test', False)
print('batches per epoch: train %d | val %d' % (len(train_loader), len(val_loader)))
'''

BUILD = '''# ---------------------------------------------------------------------------
# MODEL -- FabricNet = a PRETRAINED ResNet18 backbone (transfer learning) with
# a new classification head sized to the trained fabrics. The ImageNet-pretrained layers already know edges,
# threads and textures, so the model generalises to real-world photos far
# better than a small CNN trained from scratch -- this is the main accuracy
# fix for the misidentified outside photo.
# ---------------------------------------------------------------------------
import torch.nn as nn
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print('torchvision weights downloaded/loaded')
except Exception as e:
    print('pretrained weights unavailable (%s) -- falling back to random init' % e)
    _backbone = resnet18(weights=None)

_nclasses = len(CLASSES)
_backbone.fc = nn.Identity()
_embed_dim = 512

class FabricNet(nn.Module):
    def __init__(self, n_classes, embed_dim=512):
        super().__init__()
        self.features = _backbone          # ResNet18 up to (and incl.) global average pool
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def embed(self, x):
        return self.features(x)            # 512-dim feature vector (used by K-means + server)

model = FabricNet(_nclasses, _embed_dim).to(DEVICE)

# STAGED TRAINING: epoch 1-3 train only the new head (backbone frozen -- the
# pretrained features stay intact); epoch 4+ unfreeze everything and fine-tune.
for p in model.features.parameters():
    p.requires_grad = False
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # calmer probabilities -> the "not sure" threshold stays reliable
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
UNFREEZE_AT_EPOCH = 3
print(model)
print('parameters:', sum(p.numel() for p in model.parameters()))
'''

TRAIN = '''# ---------------------------------------------------------------------------
# TRAIN on GPU -- prints a progress line for EVERY epoch: seconds taken,
# loss, accuracy, val_loss, val_accuracy and estimated time remaining.
# Epochs 1-UNFREEZE_AT_EPOCH train only the new head; afterwards the whole
# network fine-tunes (that switch happens automatically below).
# ---------------------------------------------------------------------------
def run_epoch(loader, training):
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_val, best_state, wait = 0.0, None, 0

print()
print('=' * 88)
print((' TRAINING FabricNet -- up to %d epochs (early stop after %d flat epochs, min %d) ' % (EPOCHS, PATIENCE, MIN_EPOCHS)).center(88))
print('=' * 88)
t_train_start = time.time()
for epoch in range(1, EPOCHS + 1):
    if epoch == UNFREEZE_AT_EPOCH + 1:
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print('   ... backbone unfrozen at lr 1e-4 (fine-tuning the pretrained layers now)')
    t_ep = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    va_loss, va_acc = run_epoch(val_loader, training=False)
    history['loss'].append(tr_loss)
    history['accuracy'].append(tr_acc)
    history['val_loss'].append(va_loss)
    history['val_accuracy'].append(va_acc)
    if va_acc > best_val:
        best_val, best_state = va_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
    dt = time.time() - t_ep
    eta = dt * (EPOCHS - epoch)
    filled = int(28 * epoch / EPOCHS)
    bar = '#' * filled + '-' * (28 - filled)
    print('Epoch %3d/%d |%s| %6.1fs | loss %8.4f | acc %6.2f%% | val_loss %8.4f | val_acc %6.2f%% | ETA %5.1f min'
          % (epoch, EPOCHS, bar, dt, tr_loss, tr_acc * 100, va_loss, va_acc * 100, eta / 60))
    if wait >= PATIENCE and epoch >= MIN_EPOCHS:
        print('   early stopping at epoch %d -- no val_acc improvement for %d epochs (best %.2f%%)'
              % (epoch, PATIENCE, best_val * 100))
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print('restored best weights (val_acc %.2f%%)' % (best_val * 100))
print('=' * 88)
print(' TRAINING COMPLETE -- epochs %d/%d | %.1f min | best val_acc %.2f%%'
      % (epoch, EPOCHS, (time.time() - t_train_start) / 60, best_val * 100))
print('=' * 88)
'''

EVAL = '''model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        y_true += list(y.numpy())
        y_pred += list(model(x.to(DEVICE)).argmax(1).cpu().numpy())
y_true, y_pred = np.array(y_true), np.array(y_pred)
test_acc = float((y_true == y_pred).mean())
print('TEST accuracy %.2f%%  (%d images)' % (test_acc * 100, len(y_true)))
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

print()
print('PER-CLASS accuracy (anything under ~90%% is still getting misidentified):')
for i, c in enumerate(CLASSES):
    m = y_true == i
    print('  %-14s %6.2f%%  (%d images)' % (c, (y_pred[m] == i).mean() * 100, int(m.sum())))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(CLASSES)))
ax.set_yticklabels(CLASSES, fontsize=8)
for r in range(len(CLASSES)):
    for c in range(len(CLASSES)):
        ax.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=8,
                color='white' if cm[r, c] > cm.max() / 2 else 'black')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Test confusion matrix')
plt.colorbar(im)
plt.tight_layout()
plt.show()
'''

SUBSET = '''model.eval()
import torch as _t
class _Tex(Dataset):
    def __init__(self, paths_, labels_):
        self.p, self.y = list(paths_), list(labels_)
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (x - MEAN) / STD, self.y[i]

_tex = _Tex(te_x, te_y)
_n = min(len(_tex), 400)
_idx = np.random.choice(len(_tex), _n, replace=False)
_xs = _t.stack([_tex[int(j)][0] for j in _idx])
_ys = _t.tensor([te_y[int(j)] for j in _idx])
with _t.no_grad():
    _p = _t.softmax(model(_xs.to(DEVICE)), dim=1).cpu()
_conf = _p.max(1).values.numpy()
_correct = (_p.argmax(1).numpy() == _ys.numpy())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.hist(_conf[_correct], bins=25, range=(0, 1), color='#2563eb', label='correct')
a1.hist(_conf[~_correct], bins=25, range=(0, 1), color='#dc2626', alpha=0.75, label='wrong')
a1.axvline(0.60, color='k', ls='--', lw=1, label='not-sure threshold')
a1.set_title('Confidence on test photos')
a1.set_xlabel('confidence')
a1.legend()
a2.plot([0], [0])
a2.set_title('Calibration (reliability)')
_bins = np.linspace(0, 1, 11)
_ids = np.digitize(_conf, _bins) - 1
_cx, _cy = [], []
for b in range(10):
    m = _ids == b
    if m.sum() > 0:
        _cx.append(float(_conf[m].mean()))
        _cy.append(float(_correct[m].mean()))
a2.plot([0, 1], [0, 1], 'k--', lw=1)
a2.plot(_cx, _cy, 'o-', color='#2563eb')
a2.set_xlabel('model confidence')
a2.set_ylabel('true accuracy')
plt.tight_layout()
plt.show()
print('photos below the 0.60 threshold: %d / %d  -> the app says "not sure" on those'
      % (int((_conf < 0.60).sum()), _n))
'''

CURVES = '''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(history['loss'], label='train')
a1.plot(history['val_loss'], label='val')
a1.set_title('Loss per epoch')
a1.set_xlabel('epoch')
a1.legend()
a2.plot(history['accuracy'], label='train')
a2.plot(history['val_accuracy'], label='val')
a2.set_title('Accuracy per epoch')
a2.set_xlabel('epoch')
a2.legend()
plt.tight_layout()
plt.show()
'''

KMEANS = '''# ---------------------------------------------------------------------------
# K-MEANS over the model's learned features (512-dim, one embedding per
# training photo). Two jobs:
#   1. OUTLIER SCAN -- photos that sit far from every cluster are likely
#      mislabels or junk in the dataset; they get flagged for review.
#   2. UNKNOWN GUARD -- the cluster centroids are saved with the model; at
#      prediction time a photo far from ALL centroids is reported as
#      "unknown fabric" instead of a confident wrong class.
# ---------------------------------------------------------------------------
# The server computes the guard from the 80% CENTRE CROP of the photo, so the
# centroids and threshold are built from that SAME view plus realistic
# conditions -- otherwise ordinary trained photos get flagged as unknown.
def _server_view(src):
    img = (Image.open(src) if isinstance(src, str) else src).convert('RGB')
    w, h = img.size
    s = int(min(w, h) * 0.8)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

def _realistic_views(src):
    img = (Image.open(src) if isinstance(src, str) else src).convert('RGB')
    w, h = img.size
    base = int(max(w, h) * 1.44)
    bg = Image.new('RGB', (base, base), (120, 120, 120))
    bg.paste(img, ((base - w) // 2, (base - h) // 2))
    rot = ImageEnhance.Brightness(
        img.rotate(random.uniform(-18, 18), resample=Image.BILINEAR, fillcolor=(120, 120, 120))
    ).enhance(random.uniform(0.7, 1.3))
    return [img, _server_view(img), bg, rot]

def embed_iter(imgs_iter):
    # stream batches so we never hold every decoded image in RAM at once
    feats = []
    model.eval()
    with torch.no_grad():
        xs = []
        for im in imgs_iter:
            im = im.convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            xs.append((x - MEAN) / STD)
            if len(xs) == 64:
                feats.append(model.embed(torch.stack(xs).to(DEVICE)).cpu())
                xs = []
        if xs:
            feats.append(model.embed(torch.stack(xs).to(DEVICE)).cpu())
    return torch.cat(feats).numpy().astype('float32')

def embed_paths(paths_, view_fn=None):
    def gen():
        for p in paths_:
            im = Image.open(p).convert('RGB')
            yield (view_fn(im) if view_fn else im)
    return embed_iter(gen())

def _realistic_iter(paths_):
    for p in paths_:
        for im in _realistic_views(p):
            yield im

def _min_dist(feats, centers):
    return np.min(np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2), axis=1)

print('computing server-view embeddings for', len(tr_x), 'training photos ...')
tr_feats = embed_paths(list(tr_x), _server_view)
tr_y = np.asarray(tr_y)
fabric_mask = tr_y < len(FABRIC_CLASSES)     # the distance guard is anchored to the 6 fabrics only
fabric_tr_paths = [p for p, m in zip(tr_x, fabric_mask) if m]
print('embedding matrix:', tr_feats.shape)

K = 40
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
cluster_ids = km.fit_predict(tr_feats[fabric_mask])
dists = _min_dist(tr_feats[fabric_mask], km.cluster_centers_)

# Calibrate on realistic variants so a real photo of a trained fabric is never
# falsely flagged as unknown.
UNKNOWN_THRESHOLD = float(np.percentile(
    _min_dist(embed_iter(_realistic_iter(fabric_tr_paths[:400])), km.cluster_centers_), 99.9))
print('unknown guard: server-view train median %.1f | realistic-variant p99.9 %.1f (threshold)'
      % (np.median(dists), UNKNOWN_THRESHOLD))

# Cluster quality: how pure are the clusters w.r.t. the true fabric labels?
purity = sum(np.bincount(tr_y[fabric_mask][cluster_ids == c], minlength=len(CLASSES)).max()
             for c in range(K)) / int(fabric_mask.sum())
print('K-means (%d clusters) label purity: %.1f%% -- clusters align with fabric classes' % (K, purity * 100))

# ---- OUTLIER SCAN: the 24 farthest training photos ----
order = np.argsort(dists)[::-1]
n_out = min(24, len(order))
fig, axes = plt.subplots(4, 6, figsize=(13, 9))
for ax, j in zip(axes.flat, order[:n_out]):
    ax.imshow(Image.open(fabric_tr_paths[j]).convert('RGB').resize(IMG_SIZE))
    ax.set_title('%s\\nd=%.2f' % (CLASSES[int(tr_y[fabric_mask][j])], dists[j]), fontsize=8)
    ax.axis('off')
for ax in list(axes.flat)[n_out:]:
    ax.axis('off')
fig.suptitle('FARTHEST training photos -- if one is mislabelled or junk, remove it from the dataset and retrain', fontsize=10)
plt.tight_layout()
plt.show()

# How well would the guard behave? Test photos should sit inside the threshold;
# a genuinely unseen fabric type would land far from every centroid.
te_feats = embed_paths(list(te_x), _server_view)
te_dists = _min_dist(te_feats, km.cluster_centers_)
print('test photos beyond threshold (would be flagged unknown): %d / %d' % (int((te_dists > UNKNOWN_THRESHOLD).sum()), len(te_dists)))
'''

RFCELL = '''# ---------------------------------------------------------------------------
# RANDOM FOREST -- a SECOND, independent classifier trained on the SAME
# learned 512-d features. It never replaces the CNN: at prediction time both
# models answer, the backend compares their confidences and reports WHICH ONE
# is more confident (better_model) alongside both scores.
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)
rf.fit(tr_feats, tr_y)
rf_val_acc = float(accuracy_score(va_y, rf.predict(embed_paths(list(va_x)))))
rf_test_acc = float(accuracy_score(te_y, rf.predict(te_feats)))
print('RandomForest  val %.2f%% | test %.2f%%' % (rf_val_acc * 100, rf_test_acc * 100))
print('CNN           test %.2f%%   (higher test accuracy: %s)'
      % (test_acc * 100, 'Random Forest' if rf_test_acc > test_acc else 'CNN'))
'''

SAVE = '''# ---------------------------------------------------------------------------
# SAVE -- writes the TorchScript model (classifier + .embed for K-means), the
# K-means centroids and labels.json for the Flask backend.
# ---------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
example = torch.randn(1, 3, *IMG_SIZE).to(DEVICE)
torch.jit.trace(model.eval(), example).save(os.path.join(MODEL_DIR, 'fabricnet_simple.ts'))
# the backbone itself (fc = Identity) is the 512-d embedder used by the K-means guard
torch.jit.trace(model.features.eval(), example, check_trace=False).save(os.path.join(MODEL_DIR, 'fabricnet_embed.ts'))
np.save(os.path.join(MODEL_DIR, 'kmeans_centroids.npy'), km.cluster_centers_.astype('float32'))
# Random Forest goes next to the TorchScript models (plain pickle via joblib)
try:
    from joblib import dump as _jl_dump
    _jl_dump(rf, os.path.join(MODEL_DIR, 'randomforest.joblib'))
    _rf_saved = True
except Exception as _e:
    print('WARNING: could not save RandomForest (%s) -- server falls back to CNN-only' % _e)
    _rf_saved = False
with open(os.path.join(MODEL_DIR, 'labels.json'), 'w') as f:
    json.dump({'classes': CLASSES, 'fabric_classes': FABRIC_CLASSES,
               'unknown_class': UNKNOWN_CLASS,
               'img_size': list(IMG_SIZE), 'test_accuracy': test_acc,
               'format': 'torchscript', 'normalisation': 'mean_std_imagenet',
               'unsure_threshold': 0.60, 'unknown_threshold': round(UNKNOWN_THRESHOLD, 4),
               'kmeans_k': int(K),
               'cnn_test_accuracy': round(float(test_acc), 4),
               'rf_test_accuracy': round(float(rf_test_acc), 4),
               'better_model': 'randomforest' if rf_test_acc > test_acc else 'cnn',
               'care': CARE}, f, indent=2)
print('saved TorchScript classifier + embed model + %d K-means centroids ->' % K, MODEL_DIR)
if _rf_saved:
    print('saved RandomForest ->', os.path.join(MODEL_DIR, 'randomforest.joblib'))
'''

PREDICT = '''UNSURE_THRESHOLD = 0.60   # below this the notebook and the app say "not sure" instead of guessing

def predict_fabric(image_path, threshold=UNSURE_THRESHOLD):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    x = ((x - MEAN) / STD).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        feat = model.embed(x)[0].cpu().numpy()
    order = np.argsort(probs)[::-1]
    d_unknown = float(np.min(np.linalg.norm(km.cluster_centers_ - feat, axis=1)))
    rf_probs = rf.predict_proba(feat)[0]
    rf_best = int(np.argmax(rf_probs))
    print(os.path.basename(image_path))
    for i in order[:3]:
        print('  %-12s %6.1f%%' % (CLASSES[i], probs[i] * 100))
    print('  [CNN] best: %-12s %5.1f%% | [RandomForest] best: %-12s %5.1f%%  -> higher confidence: %s'
          % (CLASSES[order[0]], probs[order[0]] * 100, CLASSES[rf_best], rf_probs[rf_best] * 100,
             'Random Forest' if rf_probs[rf_best] > probs[order[0]] else 'CNN'))
    if CLASSES[order[0]] == UNKNOWN_CLASS or CLASSES[rf_best] == UNKNOWN_CLASS:
        print('  -> NOT A TRAINED FABRIC (model rejected the photo)')
    elif d_unknown > UNKNOWN_THRESHOLD:
        print('  -> UNKNOWN FABRIC (K-means distance %.2f > %.2f) -- likely not one of the trained fabrics'
              % (d_unknown, UNKNOWN_THRESHOLD))
    elif probs[order[0]] < threshold:
        print('  -> NOT SURE (below %.0f%%) -- treat "%s" as a guess'
              % (threshold * 100, CLASSES[order[0]]))
    return CLASSES[order[0]]

demo = te_x[0]
print('true class ->', CLASSES[int(te_y[0])])
fab = predict_fabric(demo)

# QUALITY + CARE -- same fields the app shows under the prediction.
_q = score_quality(Image.open(demo).convert('RGB'), fab)
print('  quality: %d/100 (%s) | issues: %s' % (_q['score'], _q['label'],
      ', '.join(_q['issues']) if _q['issues'] else 'none detected'))
print('  care tips for %s:' % fab)
for _tip in CARE.get(fab, []):
    print('    -', _tip)
'''

QUALITY = '''# ---------------------------------------------------------------------------
# CONDITION KNOWLEDGE -- per-fabric CARE/PRESERVATION tips (saved into
# labels.json and served to the app) + a quality scorer that reads how well
# the fabric in a photo is maintained (0-100 + issues).
#
# The thresholds are PER FABRIC on purpose: a fabric's own weave (corduroy
# ribs, denim twill) must never read as damage, while smooth fabrics (cotton,
# silk...) get full sensitivity. Calibrated with ml/tools/quality_dev.py.
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F

CARE = {
    'Cotton': [
        'Machine wash warm or cool with mild detergent; separate lights and darks',
        'Tumble dry low and remove while slightly damp to avoid deep wrinkles',
        'Iron damp on medium-high heat; use steam for stubborn creases',
        'Store folded in a dry drawer; cotton breathes, no need for plastic bags',
        'Treat stains promptly with cold water -- hot water sets many stains',
    ],
    'Linen': [
        'Hand wash or gentle machine cycle in cool water with mild detergent',
        'Line dry or dry flat; tumble heat crisps and shrinks linen fibres',
        'Iron while DAMP on high heat with steam -- dry linen is hard to press',
        'Store folded in a breathable cotton bag; never sealed plastic',
        'Expect natural softening with every wash -- that is the fabric aging well',
    ],
    'Silk': [
        'Hand wash cold with a few drops of mild detergent, or dry clean',
        'Never wring or twist -- roll in a towel to blot water out',
        'Dry flat away from direct sunlight; sun bleaches and weakens silk',
        'Iron inside-out on the lowest heat setting, barely touching',
        'Store folded in breathable cloth; keep away from rough surfaces that snag',
    ],
    'Wool': [
        'Hand wash cold or use the wool/machine cycle with wool detergent',
        'Never hang wet wool -- dry flat on a towel to keep the shape',
        'No hot iron; press with a steam cloth or use a garment steamer',
        'Store folded with cedar blocks or lavender sachets to repel moths',
        'Air garments out between wears instead of washing after every use',
    ],
    'cord': [
        'Machine wash cool on gentle, inside-out to protect the rib pile',
        'Skip fabric softener -- it flattens the characteristic ribs',
        'Air dry; dryer heat can crush and crease the corduroy ribs',
        'Brush the ribs gently with a soft brush once dry to lift the pile',
        'Iron on the reverse side at low heat; never press directly on the ribs',
    ],
    'denim': [
        'Wash cold, inside-out, and only when genuinely needed (rare washing keeps colour)',
        'Spot-clean small marks instead of a full wash',
        'Air dry always -- dryer heat shrinks denim and fades the indigo',
        'Fold denim, or hang it by the hem to avoid crease lines',
        'Turn inside-out before washing to protect seams and outer colour',
    ],
}

# fabric -> (crease_thr, stain_thr, pill_thr); unknown fabrics get the
# texture-tolerant defaults so a patterned cloth is not called damaged.
QUAL_THR = {
    'cotton': (0.08, 0.015, 0.025),
    'linen': (0.08, 0.015, 0.025),
    'silk': (0.08, 0.015, 0.025),
    'wool': (0.08, 0.015, 0.025),
    'cord': (0.40, 0.12, 0.40),
    'denim': (0.32, 0.05, 0.25),
}

def score_quality(img, fabric=None):
    """Condition of the FABRIC in a PIL photo -> 0..100 + label + issues.
    torch-only on purpose: the Flask server never imports numpy."""
    w0, h0 = img.size
    cw, ch = int(w0 * 0.7), int(h0 * 0.7)
    im = img.convert('RGB').crop(((w0 - cw) // 2, (h0 - ch) // 2,
                                  (w0 - cw) // 2 + cw, (h0 - ch) // 2 + ch))
    im = im.resize((256, 256), Image.BILINEAR)
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(256, 256, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    mx, mn = x.max(0).values, x.min(0).values
    sat = (mx - mn) / (mx + 1e-6)

    def blur(t, k):
        p = k // 2
        t = F.pad(t[None, None], (p, p, p, p), mode='replicate')
        return F.avg_pool2d(t, k, stride=1)[0, 0]

    issues, score = [], 100.0

    # creases: coarse luminance steps (13-41 px band)
    band3 = (blur(lum, 13) - blur(lum, 41)).abs()
    wrinkle = float((band3 > 0.04).float().mean())
    wt, st_thr, pt = QUAL_THR.get(str(fabric or '').lower(), QUAL_THR['cord'])
    p = max(0.0, min(25.0, (wrinkle - wt) * 200.0))
    if p > 4:
        issues.append('Uneven surface -- possible creases')
    score -= p

    # stains / dark patches: mid-frequency darkness
    mid = blur(lum, 9) - blur(lum, 41)
    stain = float((mid > 0.12).float().mean())
    p = max(0.0, min(30.0, (stain - st_thr) * 700.0))
    if p > 4:
        issues.append('Discoloration or dark patches')
    score -= p

    # pilling / speckle: clustered fine-scale outliers
    fine = lum - blur(lum, 5)
    mad = (fine - fine.median()).abs().median()
    sig = 1.4826 * mad + 1e-8
    out = (fine.abs() > 4.0 * sig).float()
    dens = F.avg_pool2d(out[None, None], 4, stride=4)[0, 0]
    pill = float((dens > 0.35).float().mean())
    p = max(0.0, min(15.0, (pill - pt) * 250.0))
    if p > 3:
        issues.append('Fine speckles / possible pilling')
    score -= p

    # fading: only judged for coloured fabric (white/grey is never "faded")
    sat_mean = float(sat.mean())
    contrast = float(lum.quantile(0.95) - lum.quantile(0.05))
    if 0.04 < sat_mean < 0.20 and contrast < 0.32:
        fade = max(0.0, min(10.0, (0.20 - sat_mean) * 60.0))
        if fade > 3:
            issues.append('Colour looks faded / washed out')
        score -= fade

    # photo clarity: a photo issue, flagged so the score is read with care
    hf_var = float(fine.var())
    blurry = hf_var < 0.00007
    if blurry:
        issues.append('Photo is blurry -- score is less reliable')
        score -= 6.0

    score = max(0.0, min(100.0, score))
    label = ('Excellent' if score >= 85 else 'Good' if score >= 70 else
             'Fair' if score >= 55 else 'Worn' if score >= 40 else 'Poorly maintained')
    return {'score': round(score), 'label': label, 'issues': issues,
            'photo': 'blurry' if blurry else 'clear'}
'''


SERVER = r'''# ---------------------------------------------------------------------------
# EXPORT -- writes simple-backend/server.py, the tiny Flask API the React
# frontend talks to. Run it afterwards with:  python simple-backend/server.py
# ---------------------------------------------------------------------------
from pathlib import Path

# r-string on purpose: keeps backslash-n escapes literal so the generated file stays valid Python.
SERVER_PY = r"""# FABRIC IQ -- simple inference server (auto-generated by the notebook).
# Run:  python simple-backend/server.py   ->   http://127.0.0.1:5001
# Loads the TorchScript model saved by the training notebook.
# NOTE: no numpy import on purpose -- the server avoids the numpy<->torch
# bridge (numpy 1.x vs 2.x incompatibility), so it runs with any numpy or none.
import ast
import io
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = torch.jit.load(str(HERE / 'model' / 'fabricnet_simple.ts'), map_location='cpu')
MODEL.eval()
EMBED = torch.jit.load(str(HERE / 'model' / 'fabricnet_embed.ts'), map_location='cpu')
EMBED.eval()


def _load_npy_tensor(path):
    # Minimal .npy reader for float32 C-order arrays -- avoids the numpy<->torch bridge.
    with open(path, 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('not a .npy file: %s' % path)
        major = f.read(1)[0]
        f.read(1)                                        # minor version
        if major < 3:
            hlen = struct.unpack('<H', f.read(2))[0]
        else:
            hlen = struct.unpack('<I', f.read(4))[0]
        header = ast.literal_eval(f.read(hlen).decode('latin1').strip())
        if header['descr'] != '<f4':
            raise ValueError('expected float32 .npy, got %r' % header['descr'])
        shape = header['shape']
        count = 1
        for s in shape:
            count *= s
        raw = f.read(4 * count)
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()


CENTROIDS = _load_npy_tensor(str(HERE / 'model' / 'kmeans_centroids.npy'))   # (K, 512)
META = json.loads((HERE / 'model' / 'labels.json').read_text())
CLASSES = META['classes']
IMG_SIZE = tuple(META['img_size'])          # (height, width)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
UNSURE_THRESHOLD = META.get('unsure_threshold', 0.60)   # below this the API answers unsure=True
UNKNOWN_THRESHOLD = META.get('unknown_threshold', 10.0) # beyond this K-means distance the photo is likely not a trained fabric
UNKNOWN_NAME = META.get('unknown_class', 'unknown')     # the rejection class trained into the model
TRIO = {'Cotton', 'Linen', 'Silk'}                       # near-white plain weaves that look alike

# SECOND MODEL -- Random Forest over the same 512-d features (saved by the
# notebook next to the TorchScript models). If it is missing the server still
# runs CNN-only and the response reports better_model='cnn'.
RF_JOBLIB = HERE / 'model' / 'randomforest.joblib'
RF = None
if RF_JOBLIB.exists():
    try:
        import joblib
        RF = joblib.load(str(RF_JOBLIB))
        print('RandomForest loaded -- both models answer and the more confident one wins')
    except Exception as _e:
        print('WARNING: RandomForest present but failed to load (%s) -- CNN only' % _e)


def _predict_with_rf(feat):
    if RF is None:
        return None
    # torch -> plain python lists on purpose: no numpy<->torch bridge involved,
    # so this works with numpy 1.x, 2.x or with numpy missing entirely.
    row = feat.detach().cpu().reshape(1, -1).tolist()
    return [float(p) for p in RF.predict_proba(row)[0]]

app = Flask(__name__)
CORS(app)


def to_tensor(img):
    w, h = img.size
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(h, w, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    return ((x - MEAN) / STD).contiguous()


# ---------------------------------------------------------------------------
# QUALITY SCORER + CARE TIPS -- how well is the fabric maintained (0-100),
# and how should it be preserved. Thresholds are PER FABRIC: a fabric's own
# weave (corduroy ribs, denim twill) must never read as damage, while smooth
# fabrics get full sensitivity. Calibrated with ml/tools/quality_dev.py.
# ---------------------------------------------------------------------------
# fabric -> (crease_thr, stain_thr, pill_thr); unknown -> texture-tolerant
QUAL_THR = {
    'cotton': (0.08, 0.015, 0.025),
    'linen': (0.08, 0.015, 0.025),
    'silk': (0.08, 0.015, 0.025),
    'wool': (0.08, 0.015, 0.025),
    'cord': (0.40, 0.12, 0.40),
    'denim': (0.32, 0.05, 0.25),
}


def score_quality(img, fabric=None):
    # Condition of the FABRIC in a PIL photo -> 0..100 + label + issues.
    # torch-only on purpose: the Flask server never imports numpy.
    w0, h0 = img.size
    cw, ch = int(w0 * 0.7), int(h0 * 0.7)
    im = img.convert('RGB').crop(((w0 - cw) // 2, (h0 - ch) // 2,
                                  (w0 - cw) // 2 + cw, (h0 - ch) // 2 + ch))
    im = im.resize((256, 256), Image.BILINEAR)
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(256, 256, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    mx, mn = x.max(0).values, x.min(0).values
    sat = (mx - mn) / (mx + 1e-6)

    def blur(t, k):
        p = k // 2
        t = F.pad(t[None, None], (p, p, p, p), mode='replicate')
        return F.avg_pool2d(t, k, stride=1)[0, 0]

    issues, score = [], 100.0

    # creases: coarse luminance steps (13-41 px band)
    band3 = (blur(lum, 13) - blur(lum, 41)).abs()
    wrinkle = float((band3 > 0.04).float().mean())
    wt, st_thr, pt = QUAL_THR.get(str(fabric or '').lower(), QUAL_THR['cord'])
    p = max(0.0, min(25.0, (wrinkle - wt) * 200.0))
    if p > 4:
        issues.append('Uneven surface -- possible creases')
    score -= p

    # stains / dark patches: mid-frequency darkness
    mid = blur(lum, 9) - blur(lum, 41)
    stain = float((mid > 0.12).float().mean())
    p = max(0.0, min(30.0, (stain - st_thr) * 700.0))
    if p > 4:
        issues.append('Discoloration or dark patches')
    score -= p

    # pilling / speckle: clustered fine-scale outliers
    fine = lum - blur(lum, 5)
    mad = (fine - fine.median()).abs().median()
    sig = 1.4826 * mad + 1e-8
    out = (fine.abs() > 4.0 * sig).float()
    dens = F.avg_pool2d(out[None, None], 4, stride=4)[0, 0]
    pill = float((dens > 0.35).float().mean())
    p = max(0.0, min(15.0, (pill - pt) * 250.0))
    if p > 3:
        issues.append('Fine speckles / possible pilling')
    score -= p

    # fading: only judged for coloured fabric (white/grey is never "faded")
    sat_mean = float(sat.mean())
    contrast = float(lum.quantile(0.95) - lum.quantile(0.05))
    if 0.04 < sat_mean < 0.20 and contrast < 0.32:
        fade = max(0.0, min(10.0, (0.20 - sat_mean) * 60.0))
        if fade > 3:
            issues.append('Colour looks faded / washed out')
        score -= fade

    # photo clarity: a photo issue, flagged so the score is read with care
    hf_var = float(fine.var())
    blurry = hf_var < 0.00007
    if blurry:
        issues.append('Photo is blurry -- score is less reliable')
        score -= 6.0

    score = max(0.0, min(100.0, score))
    label = ('Excellent' if score >= 85 else 'Good' if score >= 70 else
             'Fair' if score >= 55 else 'Worn' if score >= 40 else 'Poorly maintained')
    return {'score': round(score), 'label': label, 'issues': issues,
            'photo': 'blurry' if blurry else 'clear'}


CARE = META.get('care', {})


@app.get('/api/health')
def health():
    return jsonify(
        status='ok', classes=CLASSES, device='cpu',
        fabric_classes=META.get('fabric_classes'),
        unknown_class=META.get('unknown_class', 'unknown'),
        rf_loaded=RF is not None,
        cnn_test_accuracy=META.get('cnn_test_accuracy'),
        rf_test_accuracy=META.get('rf_test_accuracy'),
        better_model=META.get('better_model'),
    )


@app.post('/api/predict')
def predict():
    if 'file' not in request.files:
        return jsonify(error='no file uploaded', hint='POST form field: file'), 400
    try:
        img = Image.open(io.BytesIO(request.files['file'].read())).convert('RGB')
    except Exception:
        return jsonify(error='could not read that image'), 400
    # TEST-TIME AUGMENTATION: average the probabilities of three views
    # (centre / zoomed onto the weave / zoomed out) -- much steadier on
    # real-world photos than a single squashed resize.
    w, h = img.size
    views = [img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.LANCZOS)]
    s = int(min(w, h) * 0.8)
    views.append(img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.LANCZOS))
    bg = Image.new('RGB', (max(w, h), max(w, h)), (128, 128, 128))
    bg.paste(img, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
    views.append(bg.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.LANCZOS))
    batch = torch.stack([to_tensor(v) for v in views])
    with torch.no_grad():
        probs = torch.softmax(MODEL(batch), dim=1).mean(dim=0).tolist()
        feat = EMBED(batch[[1]]).squeeze(0)                    # zoomed view drives the K-means guard + RF
    d_unknown = float(torch.min(torch.cdist(CENTROIDS, feat.unsqueeze(0))).item())
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    top = float(probs[order[0]])

    # SECOND OPINION -- the Random Forest answers from the same features; the
    # model with the higher confidence wins and the response reports WHICH one.
    rf_probs = _predict_with_rf(feat)
    rf_top, rf_best = 0.0, 0
    if rf_probs:
        rf_best = max(range(len(rf_probs)), key=rf_probs.__getitem__)
        rf_top = float(rf_probs[rf_best])
    winner = 'randomforest' if rf_top > top else 'cnn'
    final_conf = rf_top if winner == 'randomforest' else top
    fabric = CLASSES[rf_best] if winner == 'randomforest' else CLASSES[order[0]]

    # NOT-A-FABRIC detection: the model REJECTS the photo (predicts the UNKNOWN
    # class) or the photo sits far from every FABRIC cluster. This is a decision,
    # not a confidence threshold -- a random photo can be "confident" yet unknown.
    is_unknown = (fabric == UNKNOWN_NAME) or (d_unknown > UNKNOWN_THRESHOLD)

    # best_guess: the most probable TRAINED fabric even when the photo is unknown,
    # so the UI can still show the closest match instead of a bare rejection.
    fabric_order = [i for i in order if CLASSES[i] != UNKNOWN_NAME]
    best_idx = fabric_order[0] if fabric_order else order[0]
    best_guess = {'fabric': CLASSES[best_idx], 'confidence': round(float(probs[best_idx]), 4)}

    # SIMILAR-FABRICS hint: cotton / linen / silk are near-white plain weaves that
    # are genuinely hard to tell apart; when the top two are both in that trio and
    # close, say so instead of forcing a single confident answer.
    similar = None
    if len(fabric_order) >= 2:
        a, b = fabric_order[0], fabric_order[1]
        if CLASSES[a] in TRIO and CLASSES[b] in TRIO and (probs[a] - probs[b]) < 0.15:
            similar = [CLASSES[a], CLASSES[b]]

    # QUALITY + CARE for the reported fabric -- the scorer is fabric-aware so
    # the fabric's own weave is never counted as damage.
    quality = score_quality(img, fabric)
    care = {'fabric': fabric, 'tips': CARE.get(fabric, [])}
    return jsonify(
        fabric=fabric,
        is_unknown=bool(is_unknown),
        unknown=bool(is_unknown),
        best_guess=best_guess,
        similar=similar,
        confidence=round(final_conf, 4),
        unsure=bool(final_conf < UNSURE_THRESHOLD),
        kmeans_distance=round(d_unknown, 3),
        better_model=winner,
        cnn={'fabric': CLASSES[order[0]], 'confidence': round(top, 4)},
        rf=(None if rf_probs is None else {'fabric': CLASSES[rf_best], 'confidence': round(rf_top, 4)}),
        alternatives=[
            {'fabric': CLASSES[i], 'confidence': round(float(probs[i]), 4)}
            for i in fabric_order[1:3]
        ],
        quality=quality,
        care=care,
    )


if __name__ == '__main__':
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _busy = _s.connect_ex(('127.0.0.1', 5001)) == 0
    if _busy:
        raise SystemExit(
            'Port 5001 is already in use -- another server instance is running.\n'
            'Close it first (find its PID with:  netstat -ano | findstr :5001\n'
            'then:  taskkill /F /PID <pid>), or just refresh the browser page.')
    app.run(host='127.0.0.1', port=5001, debug=False)
"""

sb = Path(DATA_DIR).parents[3] / 'simple-backend'
sb.mkdir(exist_ok=True)
(sb / 'server.py').write_text(SERVER_PY, encoding='utf-8')
print('wrote', sb / 'server.py')
'''

cells = [
    md('intro', '''# FABRIC IQ -- Simple Fabric Classifier Notebook (GPU / PyTorch)

Photograph a fabric, and the model names it with a confidence score and the closest alternatives.

This notebook is self-contained and trains **on the GPU** with PyTorch. Run the cells top to bottom:

1. Setup and configuration (DATA_DIR points at your dataset)
2. Condition knowledge -- per-fabric CARE tips + a quality scorer (how well is the fabric maintained?)
3. Load and inspect the dataset (only the trained fabrics: Cotton, Linen, Silk, Wool, cord, denim)
4. Train / val / test split
5. Data augmentation -- real-photo robustness copies (zoom onto the weave, rotate, flip,
   colour/shadow shifts, hue, noise, perspective), SAVED INTO THE DATASET on disk
6. Torch Dataset + DataLoaders
7. Build the CNN (FabricNet = pretrained ResNet18 backbone + your fabric head)
8. Train on GPU -- the console shows time, loss, accuracy, val metrics and ETA for every epoch
9. Evaluate on the held-out test set
10. Loss / accuracy curves
11. K-means over learned features -- outlier scan + calibrated "unknown fabric" rejection
12. Random Forest -- a SECOND model trained on the same learned features
13. Save both models for the backend
14. Predict a single image -- both models answer, the more confident one wins + quality + care tips
15. Export the tiny Flask server used by the React frontend

Two honesty layers fight misidentification of real-world photos:
- below **60% confidence** -> the app says **not sure** instead of guessing
- if K-means centroids from cell 10 put the photo far from every trained cluster -> the app
  says **not one of the trained fabrics** instead of a confident wrong name

> **Kernel:** the **.venv (3.10.11)** kernel works -- it now has torch 2.2.2+cu121 with CUDA.
> The torch_cuda (3.11) kernel works too. Any kernel with torch installed can run this notebook.
>
TWO MODELS, ONE ANSWER: cell 11 trains a **Random Forest** on the CNN's learned features.
At prediction time (cell 13 and the Flask server) BOTH models answer and the one with the
**higher confidence wins** -- the app displays which model was more confident.

> **CPU vs GPU:** cell 1 has a `USE_GPU = True` flag. The model trains on the GPU when one is
> visible (your RTX 4070) and falls back to CPU otherwise. Set `USE_GPU = False` to force CPU.
'''),
    md('md-setup', '## 1. Setup and configuration'),
    code('setup', CFG),
    md('md-quality', '## 2. Condition knowledge -- care tips + quality scorer'),
    code('quality', QUALITY),
    md('md-load', '## 3. Load and inspect the dataset'),
    code('load', LOAD),
    md('md-split', '## 4. Train / val / test split'),
    code('split', SPLIT),
    md('md-aug', '## 5. Data augmentation -- real-photo robustness (saved into the dataset)'),
    code('augment', AUG),
    md('md-dataset', '## 6. Torch Dataset and DataLoaders'),
    code('dataset', DATASET),
    md('md-build', '## 7. Build the CNN (FabricNet = pretrained ResNet18 backbone)'),
    code('build', BUILD),
    md('md-train', '## 8. Train on GPU -- time / loss / accuracy / ETA per epoch'),
    code('train', TRAIN),
    md('md-eval', '## 9. Evaluate on the test set'),
    code('evaluate', EVAL),
    md('md-curves', '## 10. Loss and accuracy curves'),
    code('curves', CURVES),
    md('md-subset', '## 11. Confidence check -- does the "not sure" flag fire where it should?'),
    code('subset', SUBSET),
    md('md-kmeans', '## 12. K-means over learned features -- outlier scan + unknown-fabric guard'),
    code('kmeans', KMEANS),
    md('md-rf', '## 13. Random Forest -- second model on the same features (higher confidence wins)'),
    code('randomforest', RFCELL),
    md('md-save', '## 14. Save both models for the backend'),
    code('save', SAVE),
    md('md-predict', '## 15. Try one prediction -- both models answer + quality + care'),
    code('predict', PREDICT),
    md('md-server', '## 16. Export the Flask server (simple-backend/server.py)'),
    code('export-server', SERVER),
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
