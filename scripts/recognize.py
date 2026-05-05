import argparse

from reconhecimento_facial.cli import run_recognize
from reconhecimento_facial.config import DEFAULT_CAMERA_INDEX, DEFAULT_CONFIDENCE_THRESHOLD, LABELS_PATH, MODEL_PATH


parser = argparse.ArgumentParser(description="Reconhece rostos pela webcam.")
parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)

args = parser.parse_args()
run_recognize(
    camera_index=args.camera,
    threshold=args.threshold,
    model_path=MODEL_PATH,
    labels_path=LABELS_PATH,
)
