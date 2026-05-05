import argparse

from reconhecimento_facial.cli import run_detect
from reconhecimento_facial.config import DEFAULT_CAMERA_INDEX


parser = argparse.ArgumentParser(description="Detecta rostos pela webcam.")
parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)

args = parser.parse_args()
run_detect(camera_index=args.camera)
