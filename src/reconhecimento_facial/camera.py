from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    pass


def open_camera(camera_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise CameraError(f"Não foi possível abrir a câmera de índice {camera_index}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def read_frame(camera: cv2.VideoCapture):
    ok, frame = camera.read()

    if not ok or frame is None:
        raise CameraError("Não foi possível ler imagem da câmera.")

    return frame
