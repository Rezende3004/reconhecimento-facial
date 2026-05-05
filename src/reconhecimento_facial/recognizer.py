from pathlib import Path

import cv2

from reconhecimento_facial.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_SIZE, LABELS_PATH, MODEL_PATH
from reconhecimento_facial.detector import FaceDetector
from reconhecimento_facial.labels import LabelStore


class FaceRecognizer:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        labels_path: str | Path = LABELS_PATH,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = float(threshold)
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {self.labels_path}")

        self.detector = FaceDetector()
        self.labels = LabelStore(self.labels_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(self.model_path))

    def recognize_frame(self, frame):
        gray, faces = self.detector.detect(frame)
        results = []

        for face_box in faces:
            face = self.detector.crop_face(gray, face_box, self.image_size)
            label_id, confidence = self.recognizer.predict(face)

            if confidence <= self.threshold:
                name = self.labels.get_name(label_id)
            else:
                name = "Desconhecido"

            results.append({
                "box": tuple(int(value) for value in face_box),
                "label_id": int(label_id),
                "name": name,
                "confidence": float(confidence),
            })

        return results
