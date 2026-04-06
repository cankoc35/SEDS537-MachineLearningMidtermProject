"""Project-wide configuration values."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models" / "saved"
RANDOM_STATE = 42
