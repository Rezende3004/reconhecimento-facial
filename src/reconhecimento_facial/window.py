from __future__ import annotations

import time

import cv2


class PreviewWindow:
    def __init__(self, title: str, max_fps: int = 20):
        self.title = title
        self.delay = max(1, int(1000 / max_fps))
        self.created = False
        self.last_frame_time = 0.0

    def show(self, frame) -> bool:
        if not self.created:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 960, 540)
            self.created = True

        now = time.monotonic()
        if now - self.last_frame_time >= self.delay / 1000:
            cv2.imshow(self.title, frame)
            self.last_frame_time = now

        key = cv2.waitKey(self.delay) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyWindow(self.title)
