from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ARTIFACT_DIR = ROOT / "artifacts"
INSTANCE_DIR = ROOT / "instance"
MODEL_PATH = ARTIFACT_DIR / "hypoguard_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
DATABASE_PATH = INSTANCE_DIR / "hypoguard.sqlite3"

for directory in (RAW_DIR, SYNTHETIC_DIR, ARTIFACT_DIR, INSTANCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HORIZON_MINUTES = 30
MIN_GLUCOSE_MGDL = 20
MAX_GLUCOSE_MGDL = 600
MODEL_VERSION = "rf-v0.1"
