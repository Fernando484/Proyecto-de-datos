# config.py
import os
from pathlib import Path

# Raíz del repositorio
ROOT = Path(__file__).resolve().parent.parent

# Ruta al Excel: cada persona la define en su ordenador sin tocar el código
DATA_PATH = Path(os.environ.get("PROJECTS_XLSX", ROOT / "data" / "projects.xlsx"))
SCICODES_PATH = Path(os.environ.get("SCIVOC_XLSX", ROOT / "data" / "SciVocCodes.xlsx"))

# Semilla común para splits, Word2Vec, PyTorch, etc.
SEED = 42

# Carpetas de salida
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SPLITS_PATH = ROOT / "data" / "splits.npz"