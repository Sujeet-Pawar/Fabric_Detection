import glob
import json
import os
import random

import torch
from PIL import Image, ImageEnhance

random.seed(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, 'simple-backend', 'model')
DATA = os.path.join(ROOT, 'ml', 'datasets', 'raw', 'my_fabric_dataset')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EMB = torch.jit.load(os.path.join(MODEL, 'fabricnet_embed.ts'), map_location=DEV).eval()
NET = torch.jit.load(os.path.join(MODEL, 'fabricnet_simple.ts'), map_location=DEV).eval()

import struct


def load_npy(path):
    with open(path, 'rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        major = f.read(1)[0]; f.read(1)
        hlen = struct.unpack('<H', f.read(2))[0] if major < 3 else struct.unpack('<I', f.read(4))[0]
        header = eval(f.read(hlen).decode('latin1').strip(), {'False': False, 'True': True})
        raw = f.read()
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(header['shape']).clone()

CENT = load_npy(os.path.join(MODEL, 'kmeans_centroids.npy')).to(DEV)
META = json.loads(open(os.path.join(MODEL, 'labels.json')).read())
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
THR = META['unknown_threshold']
print('current unknown_threshold =', THR)


def t(img):
    img = img.convert('RGB').resize((224, 224), Image.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(224, 224, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    return (x - MEAN) / STD


def crop80(img):
    w, h = img.size
    s = int(min(w, h) * 0.8)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def bg_margin(img, m=0.22):
    img = img.convert('RGB')
    w, h = img.size
    base = int(max(w, h) * (1 + 2 * m))
    bg = Image.new('RGB', (base, base), (120, 120, 120))
    bg.paste(img, ((base - w) // 2, (base - h) // 2))
    return bg


def dist_of(img):
    with torch.no_grad():
        f = EMB(t(img).unsqueeze(0).to(DEV)).squeeze(0)
        return float(torch.min(torch.cdist(CENT, f.unsqueeze(0))).item())


def pct(v):
    v = sorted(v)
    return (v[0], v[len(v)//10], v[len(v)//2], v[len(v)*9//10], v[-1])


KEEP = ['Cotton', 'Linen', 'Silk', 'Wool', 'cord', 'denim']
UNTR = ['Abaca', 'Hessian', 'canvas', 'fleece', 'rfd', 'twill']
print('\ngroup                          min    p10    med    p90    max')
for c in KEEP:
    fs = []
    for e in ('*.jpg', '*.jpeg', '*.png'):
        fs += glob.glob(os.path.join(DATA, c, e))
    fs = random.sample(fs, min(25, len(fs)))
    full = [dist_of(Image.open(p)) for p in fs]
    crop = [dist_of(crop80(Image.open(p))) for p in fs]
    bg = [dist_of(bg_margin(Image.open(p))) for p in fs]
    rot = []
    for p in fs[
        :25]:
        im = Image.open(p).convert('RGB').rotate(random.uniform(-18, 18), resample=Image.BILINEAR, fillcolor=(120, 120, 120))
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
        rot.append(dist_of(im))
    print('%-7s FULL-VIEW            %6.1f %6.1f %6.1f %6.1f %6.1f' % ((c,) + pct(full)))
    print('%-7s CROP80 (server view) %6.1f %6.1f %6.1f %6.1f %6.1f' % ((c,) + pct(crop)))
    print('%-7s BG-MARGIN            %6.1f %6.1f %6.1f %6.1f %6.1f' % ((c,) + pct(bg)))
    print('%-7s ROTATE+BRIGHT        %6.1f %6.1f %6.1f %6.1f %6.1f' % ((c,) + pct(rot)))
print()
for c in UNTR:
    fs = []
    for e in ('*.jpg', '*.jpeg', '*.png'):
        fs += glob.glob(os.path.join(DATA, c, e))
    if not fs:
        print('%-7s (no files)' % c)
        continue
    fs = random.sample(fs, min(25, len(fs)))
    d = [dist_of(crop80(Image.open(p))) for p in fs]
    print('UNTR %-7s CROP80 %6.1f %6.1f %6.1f %6.1f %6.1f' % ((c,) + pct(d)))
