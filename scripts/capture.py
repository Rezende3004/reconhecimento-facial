import argparse

from reconhecimento_facial.cli import run_capture
from reconhecimento_facial.config import DATASET_DIR, DEFAULT_CAMERA_INDEX, DEFAULT_SAMPLES


parser = argparse.ArgumentParser(description="Captura imagens de uma pessoa pela webcam.")
parser.add_argument("--name", required=True, help="Nome da pessoa.")
parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)

args = parser.parse_args()
run_capture(
    camera_index=args.camera,
    name=args.name,
    samples=args.samples,
    dataset_dir=DATASET_DIR,
)
