"""Build colab_package/project.zip: the minimal bundle the Colab notebook unzips.

Usage:  python scripts/make_colab_package.py
Upload the resulting zip to Google Drive at MyDrive/sdg_audit/project.zip, then open
notebooks/colab_run_full_grid.ipynb in Colab. Requires data/processed/ (built by notebooks 01-04).
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "scripts", "tests", "notebooks", "data/processed", "data/MANIFEST.json",
           "pyproject.toml", "requirements.txt"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "notebook_builders"}

out = ROOT / "colab_package" / "project.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        path = ROOT / item
        if path.is_file():
            z.write(path, item)
            continue
        for r, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith("egg-info")]
            for f in files:
                if not f.endswith(".pyc"):
                    fp = Path(r) / f
                    z.write(fp, fp.relative_to(ROOT).as_posix())
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
