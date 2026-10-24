"""Pull the classes we can actually use out of a TextileNet-fabric archive and
stage them as extra TRAINING images for our 6-class model.

TextileNet-fabric has 27 garment/fabric classes but does NOT contain cotton,
linen, silk or wool (those are in the separate TextileNet-FIBRE download). The
only ones matching our model are:
    denim    -> denim
    corduroy -> cord

Usage:
    python ml/tools/use_textilenet.py [path\to\fabric.tar.gz] [cap_per_class]

Output: ml/datasets/extra6/denim/* and ml/datasets/extra6/cord/*
(train_cnn6.py automatically adds EXTRA_DIR to the TRAINING split only, so the
clean val/test swatches and the reported metrics stay comparable.)
"""
import os
import random
import shutil
import sys
import tarfile

ARCHIVE = sys.argv[1] if len(sys.argv) > 1 else 'E:/auysha_clone/ml/datasets/fabric.tar.gz'
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 400

STAGE = 'E:/auysha_clone/ml/datasets/_tn_stage'
EXTRA = 'E:/auysha_clone/ml/datasets/extra6'
MAP = {'denim': 'denim', 'corduroy': 'cord'}          # TextileNet class -> our class
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

if not os.path.isfile(ARCHIVE):
    sys.exit('archive not found: %s' % ARCHIVE)

# 1) extract ONLY the two class folders we need (stream the archive, skip the rest)
#    filenames inside the archive contain '?' etc. which is illegal on Windows,
#    so each path component is sanitised before writing.
wanted = tuple('./fabric/train/%s/' % k for k in MAP)
_BAD = '<>:"|?*\n\r\t'

def safe_path(name):
    parts = [''.join('_' if ch in _BAD else ch for ch in p).rstrip('. ')
             for p in name.split('/')]
    return os.path.join(*[p for p in parts if p not in ('', '.')])

os.makedirs(STAGE, exist_ok=True)
print('extracting %s -> %s' % (list(MAP), STAGE))
kept = 0
with tarfile.open(ARCHIVE, 'r:gz') as tf:
    for member in tf:
        if member.isfile() and any(member.name.startswith(w) for w in wanted):
            target = os.path.join(STAGE, safe_path(member.name))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, 'wb') as out:
                shutil.copyfileobj(src, out)
            kept += 1
print('extracted files:', kept)

# 2) copy a capped, shuffled sample of each into extra6/<our class>/
random.seed(42)
os.makedirs(EXTRA, exist_ok=True)
for src_cls, dst_cls in MAP.items():
    src_dir = os.path.join(STAGE, 'fabric', 'train', src_cls)
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    files = [p for p in files if p.lower().endswith(IMG_EXT)]
    random.shuffle(files)
    files = files[:CAP]
    out_dir = os.path.join(EXTRA, dst_cls)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(files):
        shutil.copy2(p, os.path.join(out_dir, 'tn_%s_%04d%s' % (dst_cls, i, os.path.splitext(p)[1].lower())))
    print('%-10s -> %-6s : %d images' % (src_cls, dst_cls, len(files)))

# 3) tidy the staging dir
shutil.rmtree(STAGE, ignore_errors=True)
print('\nDONE. Extra training images are in', EXTRA)
print('Now retrain:  .venv\\Scripts\\python.exe ml\\tools\\train_cnn6.py')
print('then:         .venv\\Scripts\\python.exe ml\\tools\\train_randomforest.py')
