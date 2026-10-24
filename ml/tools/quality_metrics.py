"""Quality metrics v3 -- pick thresholds by comparing clean vs corrupted photos.

Candidate metrics are computed with REPLICATE padding (no zero-border artifact).
Prints clean-percentiles next to each synthetic corruption so thresholds can be
chosen so: clean ~ no penalty, corruption fires clearly.
"""
import glob
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

Q = 256


def blur(t, k):
    p = k // 2
    t = F.pad(t[None, None], (p, p, p, p), mode='replicate')
    return F.avg_pool2d(t, k, stride=1)[0, 0]


def metrics(img):
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

    m = {}
    # --- wrinkle candidates -------------------------------------------------
    s13 = blur(lum, 13)
    dx = (s13[:, 1:] - s13[:, :-1]).abs().mean()
    dy = (s13[1:, :] - s13[:-1, :]).abs().mean()
    m['wr_gradmean'] = float((dx + dy) / 2)
    band = (blur(lum, 5) - blur(lum, 21)).abs()
    m['wr_band'] = float((band > 0.045).float().mean())
    m['wr_band2'] = float((band > 0.030).float().mean())
    band3 = (blur(lum, 13) - blur(lum, 41)).abs()
    m['wr_b3_04'] = float((band3 > 0.04).float().mean())
    m['wr_b3_025'] = float((band3 > 0.025).float().mean())

    # --- stain candidates ---------------------------------------------------
    b32 = F.avg_pool2d(lum[None, None], 32, stride=32)[0, 0]
    gmed = b32.flatten().median()
    is_dark = (b32 < 0.70 * gmed).float()
    coh = F.avg_pool2d(is_dark[None, None], 3, stride=1, padding=1)[0, 0]
    m['st_blob'] = float((is_dark * (coh > 0.34)).float().mean())
    m['st_dark'] = float(is_dark.mean())
    b2 = F.avg_pool2d(lum[None, None], 16, stride=16)[0, 0]
    neigh = F.avg_pool2d(b2[None, None], 3, stride=1, padding=1)[0, 0]
    m['st_neigh'] = float((b2 < 0.72 * neigh).float().mean())
    mid = blur(lum, 9) - blur(lum, 41)
    m['st_mid12'] = float((mid > 0.12).float().mean())
    m['st_mid18'] = float((mid > 0.18).float().mean())
    sblk = F.avg_pool2d(sat[None, None], 32, stride=32)[0, 0]
    snei = F.avg_pool2d(sblk[None, None], 3, stride=1, padding=1)[0, 0]
    m['st_sblotch'] = float((sblk > snei + 0.25).float().mean())

    # --- periodicity (ribs/weave = periodic, wrinkles/stains/pills = not) ---
    def periodicity(resp2d, pmin=6.0, pmax=64.0):
        """Share of mid-band spectral energy concentrated in few 2D FFT bins.
        Ribs/weave -> sharp peaks (value ~1); aperiodic damage -> spread (~0)."""
        s = resp2d - resp2d.mean()
        if float(s.std()) < 1e-4:
            return 0.0
        h, w = s.shape
        P = torch.fft.rfft2(s).abs() ** 2
        fy = torch.fft.fftfreq(h, d=1.0).abs().view(h, 1)    # cycles / pixel
        fx = torch.fft.rfftfreq(w, d=1.0).abs().view(1, P.shape[1])
        rad = (fy ** 2 + fx ** 2).clamp(min=1e-9).sqrt()
        period = 1.0 / rad                                    # wavelength in px
        band = (period >= pmin) & (period <= pmax)
        Pb = P[band]
        if Pb.numel() < 8:
            return 0.0
        k = max(1, int(0.01 * Pb.numel()))
        top = Pb.sort(descending=True).values[:k].sum()
        return float(top / (Pb.sum() + 1e-8))

    m['per_mid'] = periodicity(band3, 6.0, 64.0)
    m['per_fine'] = periodicity((blur(lum, 5) - lum).abs(), 2.0, 14.0)

    def gate(period, lo=0.35, hi=0.60):
        return max(0.0, min(1.0, (hi - period) / (hi - lo)))

    m['wr_final'] = m['wr_b3_04'] * gate(m['per_mid'])
    m['st_final'] = m['st_mid12'] * gate(m['per_mid'])

    # --- pilling candidates -------------------------------------------------
    fine = lum - blur(lum, 5)
    mad = (fine - fine.median()).abs().median()
    sig = 1.4826 * mad + 1e-8
    out = (fine.abs() > 4.0 * sig).float()
    dens = F.avg_pool2d(out[None, None], 4, stride=4)[0, 0]
    m['pill_blob'] = float((dens > 0.35).float().mean())
    m['pill_tail'] = float(out.mean())
    m['pill_final'] = m['pill_blob'] * gate(m['per_fine'], 0.45, 0.75)

    # --- clarity ------------------------------------------------------------
    fine5 = lum - blur(lum, 5)
    m['hf_var'] = float(fine5.var())
    m['hf_var9'] = float((lum - blur(lum, 9)).var())

    # --- fade ---------------------------------------------------------------
    m['sat'] = float(sat.mean())
    m['contrast'] = float(lum.quantile(0.95) - lum.quantile(0.05))
    return m


def summ(rows, name):
    keys = sorted(rows[0])
    print('%-24s' % name, ' '.join('%s=%.4f' % (k, sorted(r[k] for r in rows)[len(rows) // 2]) for k in keys))


def rangev(rows, key):
    v = sorted(r[key] for r in rows)
    return 'p10 %.5f med %.5f p90 %.5f max %.5f' % (
        v[len(v) // 10], v[len(v) // 2], v[len(v) * 9 // 10], v[-1])


if __name__ == '__main__':
    random.seed(42)
    DATA = 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset'
    KEEP = ['Cotton', 'Linen', 'Silk', 'Wool', 'cord', 'denim']

    clean, per_class = [], {}
    for c in KEEP:
        files = [f for f in glob.glob(os.path.join(DATA, c, '*'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        rows = [metrics(Image.open(f)) for f in random.sample(files, 40)]
        per_class[c] = rows
        clean += rows
        print('clean %-7s %s' % (c, ' | '.join('%s p90 %.4f max %.4f' % (
            k, sorted(r[k] for r in rows)[36], max(r[k] for r in rows))
            for k in ['wr_final', 'st_final', 'pill_final', 'per_mid', 'hf_var'])))

    base_files = []
    for c in KEEP:
        files = [f for f in glob.glob(os.path.join(DATA, c, '*'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        base_files += random.sample(files, 8)
    bases = [Image.open(f).convert('RGB').resize((320, 320)) for f in base_files]

    def wrinkly(im):
        out = im.copy()
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
        out = im.copy()
        d = ImageDraw.Draw(out, 'RGBA')
        random.seed(1)
        for _ in range(6):
            x0, y0 = random.randint(60, 260), random.randint(60, 260)
            d.ellipse([x0, y0, x0 + 55, y0 + 45], fill=(60, 30, 10, 210))
        return out

    def pill_y(im):
        out = im.copy()
        d = ImageDraw.Draw(out)
        random.seed(2)
        for _ in range(500):
            xx, yy = random.randint(0, 319), random.randint(0, 319)
            v = random.choice([240, 20])
            d.ellipse([xx, yy, xx + 3, yy + 3], fill=(v, v, v))
        return out.filter(ImageFilter.GaussianBlur(0.7))

    def faded(im):
        im = ImageEnhance.Color(im).enhance(0.35)
        return ImageEnhance.Contrast(im).enhance(0.55)

    def blurred(im):
        return im.filter(ImageFilter.GaussianBlur(4))

    KEYS = ['wr_final', 'st_final', 'pill_final', 'per_mid', 'per_fine',
            'wr_b3_04', 'st_mid12', 'pill_blob', 'hf_var']

    print()
    for k in KEYS:
        print('CLEAN ALL  %-11s %s' % (k, rangev(clean, k)))
    print()
    for name, fn in [('SYNTH wrinkled', wrinkly), ('SYNTH stained', stained),
                     ('SYNTH pilled', pill_y), ('SYNTH blurred', blurred)]:
        rows = [metrics(fn(im)) for im in bases]
        print(name)
        for k in KEYS:
            print('           %-11s %s' % (k, rangev(rows, k)))
    colored = [im for im in bases if metrics(im)['sat'] >= 0.15]
    if colored:
        rows = [metrics(faded(im)) for im in colored]
        print('SYNTH faded (n=%d)' % len(rows))
        for k in ['sat', 'contrast']:
            print('           %-11s %s' % (k, rangev(rows, k)))
