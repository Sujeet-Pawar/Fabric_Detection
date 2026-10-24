"""FABRIC IQ -- retrain on the 6 requested fabrics (Cotton, Linen, Silk, Wool, cord, denim).

Replicates the main notebook recipe EXACTLY: same stratified 80/10/10 split
(SEED=42), same 6x disk augmentation, FabricNet (pretrained ResNet18 + new head),
25 epochs with staged unfreeze, K-means unknown guard (K=40, p99 threshold),
TorchScript export + labels.json (now including per-fabric CARE tips).

Run:  .venv/Scripts/python.exe ml/tools/train_cnn6.py
"""
import glob
import io
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ===========================================================================
DATA_DIR = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'
OUT_DIR = 'E:/auysha_clone/ml/datasets/fabric6'      # fresh split + augmentation
EXTRA_DIR = 'E:/auysha_clone/ml/datasets/extra6'     # optional extra images (per-class folders), TRAIN only
EXTRA_ZIPS = [                                        # (zip path, target class folder) -> staged on demand
    ('E:/auysha_clone/ml/datasets/cotton.zip', 'Cotton'),
    ('E:/auysha_clone/ml/datasets/final_linen.zip', 'Linen'),
    ('E:/auysha_clone/ml/datasets/wool2.zip', 'Wool'),
]
MAX_REAL_PER_CLASS = 400
UNKNOWN_CLASS = 'unknown'                             # rejection class ("not a trained fabric")
UNKNOWN_DIR = 'E:/auysha_clone/ml/datasets/unknown_neg'
UNK_PER_IGNORED = 40
UNK_SYNTH = 160
MODEL_DIR = 'E:/auysha_clone/simple-backend/model'
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
VAL_SPLIT, TEST_SPLIT = 0.10, 0.10
AUG_PER_IMAGE = 14
EPOCHS = 25
PATIENCE = 6
MIN_EPOCHS = 10
NUM_WORKERS = 0
SEED = 42
KEEP = {'cotton', 'linen', 'silk', 'wool', 'cord', 'denim'}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Care / preservation knowledge shipped with the model (labels.json['care']).
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

# ---------------------------------------------------------------------------
# LOAD + FILTER to the 6 requested classes
# ---------------------------------------------------------------------------
CLASSES = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(('.', '_'))
                 and d.lower() in KEEP)
assert len(CLASSES) == 6, 'expected the 6 requested classes, found: %r' % CLASSES
FABRIC_CLASSES = list(CLASSES)
CLASSES = FABRIC_CLASSES + [UNKNOWN_CLASS]     # 'unknown' is the LAST (rejection) class
print('classes (%d):' % len(CLASSES), ', '.join(CLASSES))


def build_unknown():
    # Real "other textile" photos (raw classes we do NOT train on) + synthetic
    # non-fabric images, cached on disk and shared with the RandomForest trainer.
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
for idx, cls in enumerate(FABRIC_CLASSES):
    files = glob.glob(os.path.join(DATA_DIR, cls, '**', '*.*'), recursive=True)
    files = [p for p in files if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    paths += files
    labels += [idx] * len(files)
unk_files = [p for p in glob.glob(os.path.join(unknown_dir, '*.*'))
             if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
paths += unk_files
labels += [len(FABRIC_CLASSES)] * len(unk_files)
print('total images:', len(paths))
for i, c in enumerate(CLASSES):
    print('  %-8s %5d' % (c, labels.count(i)))

tr_x, tmp_x, tr_y, tmp_y = train_test_split(
    paths, labels, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=labels, random_state=SEED)
va_x, te_x, va_y, te_y = train_test_split(
    tmp_x, tmp_y, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), stratify=tmp_y, random_state=SEED)
print('train %d | val %d | test %d' % (len(tr_x), len(va_x), len(te_x)))
tr_y, va_y, te_y = np.array(tr_y), np.array(va_y), np.array(te_y)

# Optional extra images (e.g. TextileNet denim/corduroy) go into TRAIN ONLY so
# the clean val/test swatches stay untouched and metrics remain comparable.
if not os.path.isdir(EXTRA_DIR):
    os.makedirs(EXTRA_DIR, exist_ok=True)

# Stage the user's own real-world ZIP archives (flat folders) if not extracted yet.
import zipfile
for _zip, _cls in EXTRA_ZIPS:
    if not os.path.isfile(_zip):
        continue
    _out = os.path.join(EXTRA_DIR, _cls)
    if os.path.isdir(_out) and glob.glob(os.path.join(_out, '*.*')):
        continue
    print('extracting', os.path.basename(_zip), '->', _cls, '...')
    os.makedirs(_out, exist_ok=True)
    _views = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    with zipfile.ZipFile(_zip) as _zf:
        _sel = [_m for _m in _zf.infolist()
                if not _m.is_dir() and os.path.splitext(_m.filename)[1].lower() in _views]
        random.shuffle(_sel)
        for _i, _m in enumerate(_sel[:MAX_REAL_PER_CLASS]):
            _name = os.path.basename(_m.filename.replace('?', '_').lstrip('/').replace('../', '')) or ('img_%04d' % _i)
            _dst = os.path.join(_out, 'us_%s_%04d_%s' % (_cls.lower(), _i, _name))
            with _zf.open(_m) as _src, open(_dst, 'wb') as _o:
                shutil.copyfileobj(_src, _o)

if os.path.isdir(EXTRA_DIR):
    _ex_paths, _ex_labels = [], []
    for _i, _c in enumerate(CLASSES):
        _fs = glob.glob(os.path.join(EXTRA_DIR, _c, '**', '*.*'), recursive=True)
        _fs = [p for p in _fs if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        _ex_paths += _fs
        _ex_labels += [_i] * len(_fs)
    if _ex_paths:
        tr_x = list(tr_x) + _ex_paths
        tr_y = np.concatenate([tr_y, np.array(_ex_labels, dtype=tr_y.dtype)])
        print('extra images added to TRAIN only:', len(_ex_paths),
              '-> new train size', len(tr_x))

# === APPEND-AUG ===
# ---------------------------------------------------------------------------
# DATA AUGMENTATION -- identical to the main notebook recipe: 6 copies per
# TRAIN image written to disk (zoom, rotate, flip, colour jitter, hue, noise,
# perspective, low-res degrade). Val/test stay clean for honest metrics.
# ---------------------------------------------------------------------------
def zoom_in(img, zmin=0.45, zmax=0.95):
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
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.60, 1.40))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.70, 1.30))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.60, 1.40))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.70, 1.30))
    return img

def hue_shift(img, max_deg=18):
    hsv = np.array(img.convert('HSV'), dtype=np.uint8)
    shift = int(random.uniform(-max_deg, max_deg) * 255.0 / 360.0)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 256).astype(np.uint8)
    return Image.fromarray(hsv, 'HSV').convert('RGB')

def add_noise(img, amp=0.06):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr += np.random.normal(0.0, amp, arr.shape)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))

def perspective(img, max_off=0.06):
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
    # slight focus/shake softness (kept light so fabric texture survives)
    if random.random() > p:
        return img
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.2)))

def degrade(img, p=0.45):
    # Simulate LOW-RESOLUTION captures (screenshots, thumbnails, downscaled
    # exports/reposts). Hard downscale then upscale destroys fine weave detail,
    # which previously made a low-detail LINEN photo drift toward cord. Floor
    # raised to 0.25 so the weave of cotton/linen/silk is not fully erased.
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
    for src, y in zip(files, labels_):
        dst_dir = os.path.join(OUT_DIR, split, CLASSES[y])
        os.makedirs(dst_dir, exist_ok=True)
        try:
            img = Image.open(src).convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
            img.save(os.path.join(dst_dir, os.path.splitext(os.path.basename(src))[0] + '.jpg'), quality=92)
        except Exception:
            pass

def _aug_count():
    if not os.path.isdir(os.path.join(OUT_DIR, 'train_aug')):
        return 0
    return sum(len(glob.glob(os.path.join(OUT_DIR, 'train_aug', c, '*.*'))) for c in CLASSES)

expected = len(tr_x) * AUG_PER_IMAGE
if os.path.isdir(OUT_DIR) and _aug_count() == expected:
    print('resume: split + augmentation already in', OUT_DIR, '(%d copies) -- keeping it' % expected)
else:
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
# === END-AUG ===

# === APPEND-DATASET ===
# ---------------------------------------------------------------------------
# TORCH DATASET + DATALOADERS (ImageNet normalisation, same as the notebook).
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
print('batches per epoch: train %d | val %d | test %d'
      % (len(train_loader), len(val_loader), len(test_loader)))

# ---------------------------------------------------------------------------
# MODEL -- FabricNet = pretrained ResNet18 backbone + new 6-class head.
# ---------------------------------------------------------------------------
from torchvision.models import resnet18, ResNet18_Weights
try:
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
        self.features = _backbone
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def embed(self, x):
        return self.features(x)

model = FabricNet(_nclasses, _embed_dim).to(DEVICE)

# Staged training: epochs 1-3 head only, epoch 4+ unfreeze the backbone.
for p in model.features.parameters():
    p.requires_grad = False
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
UNFREEZE_AT_EPOCH = 3
print('classes:', CLASSES)
print('parameters:', sum(p.numel() for p in model.parameters()))
# === END-DATASET ===

# === APPEND-TRAIN ===
# ---------------------------------------------------------------------------
# TRAIN -- 25 epochs, per-epoch progress line, best-val weights restored.
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
print((' TRAINING FabricNet (6 fabrics) -- up to %d epochs (early stop after %d flat epochs, min %d) ' % (EPOCHS, PATIENCE, MIN_EPOCHS)).center(88))
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
          % (epoch, EPOCHS, bar, dt, tr_loss, tr_acc * 100, va_loss, va_acc * 100, eta / 60), flush=True)
    if wait >= PATIENCE and epoch >= MIN_EPOCHS:
        print('   early stopping at epoch %d -- no val_acc improvement for %d epochs (best %.2f%%)'
              % (epoch, PATIENCE, best_val * 100), flush=True)
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print('restored best weights (val_acc %.2f%%)' % (best_val * 100))
print('=' * 88)
print(' TRAINING COMPLETE -- epochs %d/%d | %.1f min | best val_acc %.2f%%'
      % (epoch, EPOCHS, (time.time() - t_train_start) / 60, best_val * 100))
print('=' * 88)
# === END-TRAIN ===

# === APPEND-EVAL ===
# ---------------------------------------------------------------------------
# EVALUATE -- test accuracy, per-class recall, TEXT confusion matrix and
# confidence stats (the "no confusion" verification for the CNN).
# ---------------------------------------------------------------------------
model.eval()
y_true, y_pred = [], []
test_conf, test_correct = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.to(DEVICE))
        probs = torch.softmax(logits, dim=1)
        y_true += list(y.numpy())
        y_pred += list(probs.argmax(1).cpu().numpy())
        test_conf += list(probs.max(1).values.cpu().numpy())
        test_correct += list((probs.argmax(1).cpu().numpy() == y.numpy()))
y_true, y_pred = np.array(y_true), np.array(y_pred)
test_conf, test_correct = np.array(test_conf), np.array(test_correct)
test_acc = float((y_true == y_pred).mean())
print()
print('TEST accuracy %.2f%%  (%d images)' % (test_acc * 100, len(y_true)))
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
print('CONFUSION MATRIX (rows = true, cols = predicted)')
print('%-9s' % '' + ''.join('%9s' % c[:8] for c in CLASSES))
for i, c in enumerate(CLASSES):
    print('%-9s' % c[:8] + ''.join('%9d' % cm[i, j] for j in range(len(CLASSES))))

print()
print('PER-CLASS recall (100.00% = never confused with another fabric):')
worst = 1.0
for i, c in enumerate(CLASSES):
    rec = cm[i, i] / max(cm[i].sum(), 1)
    worst = min(worst, rec)
    others = ', '.join('%s:%d' % (CLASSES[j], cm[i, j])
                       for j in range(len(CLASSES)) if j != i and cm[i, j] > 0) or 'none'
    print('  %-8s recall %6.2f%%  confused with -> %s' % (c, rec * 100, others))

off_diag = int(cm.sum() - np.trace(cm))
print('total confusions on test set: %d / %d' % (off_diag, len(y_true)))
if off_diag:
    print('!! CONFUSION PRESENT -- see matrix above; iterate (more epochs/aug) if needed')
print('mean confidence: correct %.1f%% | wrong %.1f%%'
      % (test_conf[test_correct].mean() * 100,
         test_conf[~test_correct].mean() * 100 if (~test_correct).any() else 0.0))
print('test photos below the 0.60 "not sure" threshold: %d / %d'
      % (int((test_conf < 0.60).sum()), len(test_conf)))
# === END-EVAL ===

# === APPEND-KMEANS ===
# ---------------------------------------------------------------------------
# K-MEANS unknown guard, calibrated in the SAME view the Flask server uses.
#
# The server computes the guard distance from the 80% CENTRE CROP of the photo.
# Calibrating on the plain full-resize embedding (as an earlier version did)
# produced a threshold that ordinary trained photos exceeded, so real user
# photos of cotton/denim/wool were falsely reported as "not a trained fabric".
# Here the centroids and the threshold are built from that same centre-crop
# view PLUS realistic conditions (background, rotation, lighting), so a trained
# fabric in a real photo is never falsely flagged.
# ---------------------------------------------------------------------------
def _server_view(src):
    # the view the server embeds for the K-means guard: 80% centre crop
    img = (Image.open(src) if isinstance(src, str) else src).convert('RGB')
    w, h = img.size
    s = int(min(w, h) * 0.8)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

def _realistic_views(src):
    # what a real phone photo of the fabric may look like
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

print()
print('computing server-view embeddings for', len(tr_x), 'training photos ...')
tr_view = embed_paths(list(tr_x), _server_view)         # same view the server uses
print('embedding matrix:', tr_view.shape)

K = 40
fabric_mask = tr_y < len(FABRIC_CLASSES)                 # the distance guard is anchored to the 6 fabrics only
fabric_tr = [p for p, m in zip(tr_x, fabric_mask) if m]
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
cluster_ids = km.fit_predict(tr_view[fabric_mask])
dists = _min_dist(tr_view[fabric_mask], km.cluster_centers_)
purity = sum(np.bincount(tr_y[fabric_mask][cluster_ids == c], minlength=len(CLASSES)).max()
             for c in range(K)) / int(fabric_mask.sum())
print('K-means (%d clusters) label purity: %.1f%%' % (K, purity * 100))

# Calibrate the unknown threshold on realistic variants of the training photos.
calib_d = _min_dist(embed_iter(_realistic_iter(fabric_tr[:400])), km.cluster_centers_)
UNKNOWN_THRESHOLD = float(np.percentile(calib_d, 99.9))
print('unknown guard: server-view train med %.1f | realistic-variant p99.9 %.1f (threshold) | max %.1f'
      % (np.median(dists), UNKNOWN_THRESHOLD, calib_d.max()))

te_view = embed_paths(list(te_x), _server_view)
te_dists = _min_dist(te_view, km.cluster_centers_)
print('test photos beyond threshold (would be flagged unknown): %d / %d'
      % (int((te_dists > UNKNOWN_THRESHOLD).sum()), len(te_dists)))
# === END-KMEANS ===

# === APPEND-SAVE ===
# ---------------------------------------------------------------------------
# SAVE -- TorchScript classifier + embedder, K-means centroids, labels.json
# (now carrying the per-fabric CARE tips). The old 12-class Random Forest is
# removed here so the server can never mix 12-class RF with 6-class CNN;
# retrain it with ml/tools/train_randomforest.py afterwards.
# ---------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
example = torch.randn(1, 3, *IMG_SIZE).to(DEVICE)
torch.jit.trace(model.eval(), example).save(os.path.join(MODEL_DIR, 'fabricnet_simple.ts'))
torch.jit.trace(model.features.eval(), example, check_trace=False).save(os.path.join(MODEL_DIR, 'fabricnet_embed.ts'))
np.save(os.path.join(MODEL_DIR, 'kmeans_centroids.npy'), km.cluster_centers_.astype('float32'))

old_rf = os.path.join(MODEL_DIR, 'randomforest.joblib')
if os.path.exists(old_rf):
    os.remove(old_rf)
    print('removed the previous (12-class) Random Forest -- retrain it with '
          'ml/tools/train_randomforest.py')

with open(os.path.join(MODEL_DIR, 'labels.json'), 'w') as f:
    json.dump({'classes': CLASSES, 'fabric_classes': FABRIC_CLASSES,
               'unknown_class': UNKNOWN_CLASS,
               'img_size': list(IMG_SIZE), 'test_accuracy': test_acc,
               'format': 'torchscript', 'normalisation': 'mean_std_imagenet',
               'unsure_threshold': 0.60, 'unknown_threshold': round(UNKNOWN_THRESHOLD, 4),
               'kmeans_k': int(K),
               'cnn_test_accuracy': round(float(test_acc), 4),
               'better_model': 'cnn',
               'care': CARE}, f, indent=2)

print('saved TorchScript classifier + embed model + %d K-means centroids ->' % K, MODEL_DIR)
print()
print('================ 6-FABRIC SUMMARY ================')
print('  classes        :', ', '.join(CLASSES))
print('  CNN test acc   : %.2f%%' % (test_acc * 100))
print('  confusions     : %d' % off_diag)
print('  unknown thresh : %.3f' % UNKNOWN_THRESHOLD)
print('  care tips      : %d fabrics in labels.json' % len(CARE))
print('next: .venv/Scripts/python.exe ml/tools/train_randomforest.py')
# === END-SAVE ===
