from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
RUNTIME_DIR = DATA_DIR / "runtime"

ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"

MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"

HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

DEFAULT_CAMERA_INDEX = 0
DEFAULT_SAMPLES = 50
DEFAULT_CONFIDENCE_THRESHOLD = 70.0
DEFAULT_IMAGE_SIZE = (200, 200)
