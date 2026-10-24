"""Real-world accuracy check: take the clean swatches in fabric6/test, generate
realistic-condition variants (background margin, rotation+lighting, slight blur)
and measure how often the model still gets the right fabric. Run before and after
a change to judge whether it actually helps real photos."""
import glob, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

MODEL_DIR = 'E:/auysha_clone/simple-backend/model'
TEST_DIR = sys.argv[1] if len(sys.argv) > 1 else 'E:/auysha_clone/ml/datasets/fabric_simple/test'
N_PER_CLASS = 15

random.seed(0)
np.random.seed(0)

lab = json.load(open(os.path.join(MODEL_DIR, 'labels.json'), encoding='utf-8'))
classes = lab['classes']
img_size = tuple(lab['img_size'])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
m = torch.jit.load(os.path.join(MODEL_DIR, 'fabricnet_simple.ts'), map_location='cpu')
m.eval()

def variants(im):
    w, h = im.size
    out = [im]
    base = int(max(w, h) * 1.44)
    bg = Image.new('RGB', (base, base), (120, 120, 120))
    bg.paste(im, ((base - w) // 2, (base - h) // 2))
    out.append(bg)
    out.append(ImageEnhance.Brightness(
        im.rotate(random.uniform(-15, 15), resample=Image.BILINEAR, fillcolor=(120, 120, 120))
    ).enhance(random.uniform(0.75, 1.25)))
    out.append(im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0))))
    w2, h2 = max(8, int(w * 0.28)), max(8, int(h * 0.28))
    out.append(im.resize((w2, h2), Image.BILINEAR).resize((w, h), Image.BILINEAR))
    return out

def predict(im):
    w, h = im.size
    vs = [im.resize(img_size, Image.BILINEAR)]
    s = int(min(w, h) * 0.8)
    vs.append(im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize(img_size, Image.BILINEAR))
    side = max(w, h)
    sq = Image.new('RGB', (side, side), (128, 128, 128))
    sq.paste(im, ((side - w) // 2, (side - h) // 2))
    vs.append(sq.resize(img_size, Image.BILINEAR))
    xs = [(torch.from_numpy(np.asarray(v, dtype=np.float32).transpose(2, 0, 1) / 255.0) - MEAN) / STD for v in vs]
    with torch.no_grad():
        p = F.softmax(m(torch.stack(xs)), 1).mean(0)
    return classes[int(p.argmax())], float(p.max())

total = correct = 0
per = {}
for ci, c in enumerate(classes):
    fs = [p for p in glob.glob(os.path.join(TEST_DIR, c, '*'))
          if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))][:N_PER_CLASS]
    cc = nn = 0
    for p in fs:
        im = Image.open(p).convert('RGB')
        for v in variants(im):
            pred, conf = predict(v)
            total += 1
            correct += (pred == c)
            cc += (pred == c)
            nn += 1
    per[c] = (cc, nn)
    print('  %-8s %3d/%3d  %.0f%%' % (c, cc, nn, 100 * cc / max(1, nn)))
print('REAL-WORLD accuracy over %d variant-images: %.2f%%' % (total, 100 * correct / max(1, total)))
