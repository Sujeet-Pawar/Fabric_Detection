"""Quality-score heuristic -- calibration harness for the Flask server scorer.

The FINAL score_quality(img, fabric) lives in SERVER_PY (ml/tools/
make_simple_notebook.py) and is copied into simple-backend/server.py.
Thresholds are PER FABRIC (texture-aware): a fabric's own weave/ribs must never
read as damage, while smooth fabrics get full sensitivity. This harness proves
both properties against real dataset photos + synthetic corruptions.

Run:  .venv/Scripts/python.exe ml/tools/quality_dev.py
"""
import glob
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

Q = 256

# fabric -> (wrinkle_thr, stain_thr, pill_thr); unknown -> texture-tolerant
QUAL_THR = {
    'cotton': (0.08, 0.015, 0.025),
    'linen': (0.08, 0.015, 0.025),
    'silk': (0.08, 0.015, 0.025),
    'wool': (0.08, 0.015, 0.025),
    'cord': (0.40, 0.12, 0.40),
    'denim': (0.32, 0.05, 0.25),
}


def score_quality(img, fabric=None):
    """Condition of the FABRIC in a PIL photo -> 0..100 + label + issues."""
    w0, h0 = img.size
    cw, ch = int(w0 * 0.7), int(h0 * 0.7)
    im = img.convert('RGB').crop(((w0 - cw) // 2, (h0 - ch) // 2,
                                  (w0 - cw) // 2 + cw, (h0 - ch) // 2 + ch))
    im = im.resize((Q, Q), Image.BILINEAR)
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(Q, Q, 3)
    x = x.permute(2, 0, 1).float().div_(255.0)
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    mx, mn = x.max(0).values, x.min(0).values
    sat = (mx - mn) / (mx + 1e-6)

    def blur(t, k):
        p = k // 2
        t = F.pad(t[None, None], (p, p, p, p), mode='replicate')
        return F.avg_pool2d(t, k, stride=1)[0, 0]

    issues, score = [], 100.0

    # --- 1. creases: coarse luminance steps (13-41px band) -------------------
    band3 = (blur(lum, 13) - blur(lum, 41)).abs()
    wrinkle = float((band3 > 0.04).float().mean())
    wt, st_thr, pt = QUAL_THR.get(str(fabric or '').lower(), QUAL_THR['cord'])
    p = max(0.0, min(25.0, (wrinkle - wt) * 200.0))
    if p > 4:
        issues.append('Visible creases / wrinkles')
    score -= p

    # --- 2. stains: mid-frequency dark patches -------------------------------
    mid = blur(lum, 9) - blur(lum, 41)
    stain = float((mid > 0.12).float().mean())
    p = max(0.0, min(30.0, (stain - st_thr) * 700.0))
    if p > 4:
        issues.append('Possible stains or dark patches')
    score -= p

    # --- 3. pilling: clustered fine-scale outliers ---------------------------
    fine = lum - blur(lum, 5)
    mad = (fine - fine.median()).abs().median()
    sig = 1.4826 * mad + 1e-8
    out = (fine.abs() > 4.0 * sig).float()
    dens = F.avg_pool2d(out[None, None], 4, stride=4)[0, 0]
    pill = float((dens > 0.35).float().mean())
    p = max(0.0, min(15.0, (pill - pt) * 250.0))
    if p > 3:
        issues.append('Surface pilling / speckles')
    score -= p

    # --- 4. fading: only judged for coloured fabric --------------------------
    sat_mean = float(sat.mean())
    contrast = float(lum.quantile(0.95) - lum.quantile(0.05))
    if 0.04 < sat_mean < 0.20 and contrast < 0.32:
        fade = max(0.0, min(10.0, (0.20 - sat_mean) * 60.0))
        if fade > 3:
            issues.append('Colour looks faded / washed out')
        score -= fade

    # --- 5. photo clarity (photo issue, small penalty, flagged separately) ---
    hf_var = float(fine.var())
    blurry = hf_var < 0.00007
    if blurry:
        issues.append('Photo is blurry -- score is less reliable')
        score -= 6.0

    score = max(0.0, min(100.0, score))
    label = ('Excellent' if score >= 85 else 'Good' if score >= 70 else
             'Fair' if score >= 55 else 'Worn' if score >= 40 else 'Poorly maintained')
    return {'score': round(score), 'label': label, 'issues': issues,
            'photo': 'blurry' if blurry else 'clear',
            'metrics': {'wrinkle': round(wrinkle, 4), 'stain': round(stain, 4),
                        'pill': round(pill, 4), 'sat': round(sat_mean, 4),
                        'contrast': round(contrast, 4), 'hf_var': round(hf_var, 6)}}


def stats(scores, name):
    s = sorted(scores)
    print('%-26s n=%3d  min %5.1f  p10 %5.1f  med %5.1f  p90 %5.1f  max %5.1f'
          % (name, len(s), s[0], s[len(s) // 10], s[len(s) // 2], s[len(s) * 9 // 10], s[-1]))


if __name__ == '__main__':
    random.seed(42)
    DATA = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'
    KEEP = ['Cotton', 'Linen', 'Silk', 'Wool', 'cord', 'denim']

    print('== CLEAN dataset photos (class-aware scoring; want >= 85) ==')
    all_clean = []
    for c in KEEP:
        files = [f for f in glob.glob(os.path.join(DATA, c, '*'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        sc = [score_quality(Image.open(f), c)['score']
              for f in random.sample(files, 40)]
        all_clean += sc
        stats(sc, 'clean ' + c)
    stats(all_clean, 'clean ALL')

    SMOOTH = ['Cotton', 'Linen', 'Silk', 'Wool']
    base_files = []
    for c in SMOOTH:
        files = [f for f in glob.glob(os.path.join(DATA, c, '*'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        base_files += random.sample(files, 12)
    bases = [(Image.open(f).convert('RGB').resize((320, 320)), c)
             for f, c in zip(base_files, [c for c in SMOOTH for _ in range(12)])]
    random.shuffle(bases)

    def wrinkly(im):
        out = im.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(out)
        random.seed(3)
        w, h = out.size
        for _ in range(7):
            x0, y0 = random.randint(-40, w), random.randint(-40, h)
            x1, y1 = x0 + random.randint(60, 240), y0 + random.randint(-60, 180)
            tone = random.choice([250, 10])
            d.line([x0, y0, x1, y1], fill=(tone, tone, tone), width=random.randint(4, 8))
        return out.filter(ImageFilter.GaussianBlur(2.5))

    def stained(im):
        from PIL import ImageDraw
        out = im.copy()
        d = ImageDraw.Draw(out, 'RGBA')
        random.seed(1)
        for _ in range(6):
            x0, y0 = random.randint(60, 260), random.randint(60, 260)
            d.ellipse([x0, y0, x0 + 55, y0 + 45], fill=(60, 30, 10, 210))
        return out

    def pill_y(im):
        from PIL import ImageDraw
        out = im.copy()
        d = ImageDraw.Draw(out)
        random.seed(2)
        for _ in range(500):
            xx, yy = random.randint(0, 319), random.randint(0, 319)
            v = random.choice([240, 20])
            d.ellipse([xx, yy, xx + 3, yy + 3], fill=(v, v, v))
        return out.filter(ImageFilter.GaussianBlur(0.7))

    def faded(im):
        from PIL import ImageEnhance
        im = ImageEnhance.Color(im).enhance(0.35)
        return ImageEnhance.Contrast(im).enhance(0.55)

    print()
    print('== SYNTHETIC damage on smooth-class photos (want <= 80) ==')
    for name, fn in [('wrinkled', wrinkly), ('stained', stained), ('pilled', pill_y)]:
        sc = [score_quality(fn(im), c)['score'] for im, c in bases]
        stats(sc, 'SYNTH ' + name)
    colored = [(im, c) for im, c in bases if score_quality(im, c)['metrics']['sat'] >= 0.15]
    if colored:
        stats([score_quality(faded(im), c)['score'] for im, c in colored],
              'SYNTH faded (colored)')
    stats([score_quality(im.filter(ImageFilter.GaussianBlur(4)), c)['score']
           for im, c in bases], 'SYNTH blurred')

    print()
    print('== issue labels on 12 sample photos ==')
    for (im, c), fn, tag in [(bases[0], None, 'clean'),
                             (bases[1], wrinkly, 'wrinkled'),
                             (bases[2], stained, 'stained'),
                             (bases[3], pill_y, 'pilled')]:
        img = fn(im) if fn else im
        print('%-9s %-7s %s' % (tag, c, score_quality(img, c)))
