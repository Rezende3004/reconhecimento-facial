from reconhecimento_facial.cli import run_train
from reconhecimento_facial.config import DATASET_DIR, LABELS_PATH, MODEL_PATH


run_train(
    dataset_dir=DATASET_DIR,
    model_path=MODEL_PATH,
    labels_path=LABELS_PATH,
)
