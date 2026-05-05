from pathlib import Path

import cv2
import numpy as np

from reconhecimento_facial.config import DEFAULT_IMAGE_SIZE, LABELS_PATH, MODEL_PATH
from reconhecimento_facial.detector import FaceDetector
from reconhecimento_facial.labels import LabelStore
from reconhecimento_facial.logger import get_logger


logger = get_logger(__name__)


class TrainingError(RuntimeError):
    pass


def collect_training_data(
    dataset_dir: str | Path,
    labels_path: str | Path = LABELS_PATH,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
):
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise TrainingError(f"A pasta do dataset não existe: {dataset_path}")

    label_store = LabelStore(labels_path)
    detector = FaceDetector()

    faces = []
    labels = []

    people_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())

    if not people_dirs:
        raise TrainingError("Nenhuma pessoa encontrada no dataset.")

    for person_dir in people_dirs:
        person_name = person_dir.name
        label_id = label_store.get_or_create_id(person_name)

        image_paths = sorted(
            path for path in person_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        if not image_paths:
            logger.warning("A pasta %s não tem imagens válidas.", person_dir)
            continue

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                logger.warning("Não foi possível ler a imagem: %s", image_path)
                continue

            gray, detected_faces = detector.detect(image)

            if len(detected_faces) == 0:
                logger.warning("Nenhum rosto encontrado em: %s", image_path)
                continue

            largest_face = max(detected_faces, key=lambda box: box[2] * box[3])
            face = detector.crop_face(gray, largest_face, image_size)

            faces.append(face)
            labels.append(label_id)

    if not faces:
        raise TrainingError("Nenhum rosto válido foi encontrado para treinamento.")

    return faces, np.array(labels, dtype=np.int32), label_store


def train_model(
    dataset_dir: str | Path,
    model_path: str | Path = MODEL_PATH,
    labels_path: str | Path = LABELS_PATH,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> None:
    faces, labels, label_store = collect_training_data(
        dataset_dir=dataset_dir,
        labels_path=labels_path,
        image_size=image_size,
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    recognizer.save(str(model_file))
    label_store.save()

    logger.info("Treinamento concluído com %s rosto(s).", len(faces))
    logger.info("Modelo salvo em: %s", model_file)
    logger.info("Labels salvos em: %s", labels_path)
