from pathlib import Path

import cv2

from reconhecimento_facial.config import HAAR_CASCADE_PATH


class FaceDetector:
    def __init__(self, cascade_path: str | Path = HAAR_CASCADE_PATH):
        if str(cascade_path) == HAAR_CASCADE_PATH:
            cascade_file = cv2.data.haarcascades + HAAR_CASCADE_PATH
        else:
            cascade_file = str(cascade_path)

        self.detector = cv2.CascadeClassifier(cascade_file)

        if self.detector.empty():
            raise RuntimeError("Não foi possível carregar o classificador Haar Cascade.")

    def detect(self, frame, scale_factor: float = 1.2, min_neighbors: int = 5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(60, 60),
        )

        return gray, faces

    @staticmethod
    def crop_face(gray_frame, face_box, image_size: tuple[int, int]):
        x, y, w, h = face_box
        face = gray_frame[y:y + h, x:x + w]
        return cv2.resize(face, image_size)
