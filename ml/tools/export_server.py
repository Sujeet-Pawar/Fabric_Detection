"""Writes simple-backend/server.py by executing the notebook's export-server cell.

The SERVER_PY template inside ml/tools/make_simple_notebook.py is the single
source of truth for the Flask server; this helper runs that cell (in a scope
providing DATA_DIR, as the notebook would) so the exported file always matches
the notebook.

Run from the repo root:  .venv/Scripts/python.exe ml/tools/export_server.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / 'ml' / 'notebooks' / 'fabric_classifier_simple.ipynb'

nb = json.loads(NB.read_text(encoding='utf-8'))
cell = next(c for c in nb['cells'] if c.get('id') == 'export-server')
src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

# The cell computes sb from DATA_DIR's parents; provide the same path the notebook uses.
scope = {'DATA_DIR': 'E:/auysha_clone/ml/datasets/raw/my_fabric_dataset', '__name__': '__export__'}
exec(compile(src, str(NB) + '#export-server', 'exec'), scope)
print('server.py exported from the notebook cell')
