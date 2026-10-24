"""Full-dataset confusion check for both models (batched, single view -- fast diagnostic).

Prints how many photos each model gets wrong across EVERY image of the six trained
fabrics, plus the confusion pairs and the cases wrong by BOTH models (which is what
matters, since the more confident model's answer is the one shown).

Run:  .venv/Scripts/python.exe ml/tools/full_dataset_check.py
"""
import glob
import json
import os

import joblib
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', '..', 'simple-backend', 'model')
DATA = os.path.join(HERE, '..', 'datasets', 'raw', 'my_fabric_dataset')
KEEP = ['Cotton', 'Linen', 'Silk', 'Wool', 'cord', 'denim']

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, '| cuda:', torch.cuda.is_available())
NET = torch.jit.load(os.path.join(MODEL, 'fabricnet_simple.ts'), map_location=DEV).eval()
EMB = torch.jit.load(os.path.join(MODEL, 'fabricnet_embed.ts'), map_location=DEV).eval()
RF = joblib.load(os.path.join(MODEL, 'randomforest.joblib'))
META = json.loads(open(os.path.join(MODEL, 'labels.json')).read())
CLASSES = META['classes']
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_t(path):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(224, 224, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    return (x - MEAN) / STD


items = []
for true in KEEP:
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        files += glob.glob(os.path.join(DATA, true, ext))
    for p in files:
        items.append((p, true))

cnn_err, both_cm = {}, {}
cnn_wrong = rf_wrong = both = 0
B = 128
with torch.no_grad():
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        batch = torch.stack([to_t(p) for p, _ in chunk]).to(DEV)
        cnn = torch.softmax(NET(batch), dim=1).cpu()
        feat = EMB(batch).cpu()
        rf = RF.predict_proba(feat.reshape(len(chunk), -1).tolist())
        for j, (p, true) in enumerate(chunk):
            cpred = CLASSES[int(cnn[j].argmax())]
            rpred = CLASSES[int(rf[j].argmax())]
            if cpred != true:
                cnn_wrong += 1
                cnn_err[(true, cpred)] = cnn_err.get((true, cpred), 0) + 1
            if rpred != true:
                rf_wrong += 1
            if cpred != true and rpred != true:
                both += 1
                key = (true, cpred, rpred)
                both_cm[key] = both_cm.get(key, 0) + 1
        print('  %d/%d done' % (min(i + B, len(items)), len(items)), flush=True)

n = len(items)
print('images', n)
print('CNN wrong: %d (%.2f%%) | RF wrong: %d (%.2f%%) | BOTH wrong: %d (%.2f%%)'
      % (cnn_wrong, 100 * cnn_wrong / n, rf_wrong, 100 * rf_wrong / n, both, 100 * both / n))
print('CNN confusion pairs (true -> pred : count):')
for (t, p), c in sorted(cnn_err.items(), key=lambda kv: -kv[1]):
    print('   %-7s -> %-7s %d' % (t, p, c))
print('BOTH-wrong cases:')
for (t, cp, rp), c in sorted(both_cm.items(), key=lambda kv: -kv[1]):
    print('   true %-7s cnn->%-7s rf->%-7s : %d' % (t, cp, rp, c))
