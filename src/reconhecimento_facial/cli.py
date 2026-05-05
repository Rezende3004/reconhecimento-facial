import argparse
from pathlib import Path

import cv2

from reconhecimento_facial.camera import open_camera, read_frame
from reconhecimento_facial.config import (
    DATASET_DIR,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SAMPLES,
    LABELS_PATH,
    MODEL_PATH,
    RUNTIME_DIR,
)
from reconhecimento_facial.detector import FaceDetector
from reconhecimento_facial.logger import get_logger
from reconhecimento_facial.recognizer import FaceRecognizer
from reconhecimento_facial.runtime import SingleInstanceLock
from reconhecimento_facial.trainer import train_model
from reconhecimento_facial.window import PreviewWindow


logger = get_logger(__name__)


def draw_box(frame, box, text: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def run_detect(camera_index: int) -> None:
    lock = SingleInstanceLock(RUNTIME_DIR / f"camera_{camera_index}.lock")
    lock.acquire()

    detector = FaceDetector()
    camera = open_camera(camera_index)
    window = PreviewWindow("Reconhecimento facial - deteccao")

    logger.info("Detecção iniciada. Pressione 'q' ou ESC para sair.")

    try:
        while True:
            frame = read_frame(camera)
            _, faces = detector.detect(frame)

            for box in faces:
                draw_box(frame, box, "Rosto", (0, 255, 0))

            if not window.show(frame):
                break
    finally:
        camera.release()
        window.close()
        lock.release()


def run_capture(camera_index: int, name: str, samples: int, dataset_dir: Path) -> None:
    lock = SingleInstanceLock(RUNTIME_DIR / f"camera_{camera_index}.lock")
    lock.acquire()

    detector = FaceDetector()
    camera = open_camera(camera_index)
    window = PreviewWindow("Reconhecimento facial - captura")

    person_dir = dataset_dir / name.strip()
    person_dir.mkdir(parents=True, exist_ok=True)

    count = len(list(person_dir.glob("*.jpg")))

    logger.info("Captura iniciada para: %s", name)
    logger.info("Pressione 'q' para sair antes do fim.")

    try:
        while count < samples:
            frame = read_frame(camera)
            gray, faces = detector.detect(frame)

            for box in faces:
                face = detector.crop_face(gray, box, DEFAULT_IMAGE_SIZE)

                count += 1
                image_path = person_dir / f"{count:04d}.jpg"
                cv2.imwrite(str(image_path), face)

                draw_box(frame, box, f"{name} | {count}/{samples}", (0, 255, 0))

                if count >= samples:
                    break

            if not window.show(frame):
                break
    finally:
        camera.release()
        window.close()
        lock.release()

    logger.info("Captura finalizada. Imagens salvas em: %s", person_dir)


def run_train(dataset_dir: Path, model_path: Path, labels_path: Path) -> None:
    train_model(
        dataset_dir=dataset_dir,
        model_path=model_path,
        labels_path=labels_path,
    )


def run_recognize(camera_index: int, threshold: float, model_path: Path, labels_path: Path) -> None:
    recognizer = FaceRecognizer(
        model_path=model_path,
        labels_path=labels_path,
        threshold=threshold,
    )

    lock = SingleInstanceLock(RUNTIME_DIR / f"camera_{camera_index}.lock")
    lock.acquire()

    camera = open_camera(camera_index)
    window = PreviewWindow("Reconhecimento facial - reconhecimento")

    logger.info("Reconhecimento iniciado. Pressione 'q' para sair.")

    try:
        while True:
            frame = read_frame(camera)
            results = recognizer.recognize_frame(frame)

            for result in results:
                name = result["name"]
                confidence = result["confidence"]
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                draw_box(frame, result["box"], text, color)

            if not window.show(frame):
                break
    finally:
        camera.release()
        window.close()
        lock.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconhecimento-facial",
        description="Reconhecimento facial com OpenCV.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Abre a webcam e detecta rostos.")
    detect_parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)

    capture_parser = subparsers.add_parser("capture", help="Captura imagens de uma pessoa.")
    capture_parser.add_argument("--name", required=True, help="Nome da pessoa.")
    capture_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    capture_parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    capture_parser.add_argument("--dataset", type=Path, default=DATASET_DIR)

    train_parser = subparsers.add_parser("train", help="Treina o modelo.")
    train_parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--labels", type=Path, default=LABELS_PATH)

    recognize_parser = subparsers.add_parser("recognize", help="Reconhece rostos pela webcam.")
    recognize_parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    recognize_parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    recognize_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    recognize_parser.add_argument("--labels", type=Path, default=LABELS_PATH)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "detect":
        run_detect(camera_index=args.camera)
    elif args.command == "capture":
        run_capture(
            camera_index=args.camera,
            name=args.name,
            samples=args.samples,
            dataset_dir=args.dataset,
        )
    elif args.command == "train":
        run_train(
            dataset_dir=args.dataset,
            model_path=args.model,
            labels_path=args.labels,
        )
    elif args.command == "recognize":
        run_recognize(
            camera_index=args.camera,
            threshold=args.threshold,
            model_path=args.model,
            labels_path=args.labels,
        )


if __name__ == "__main__":
    main()
