from __future__ import annotations

import atexit
import os
from pathlib import Path


class SingleInstanceError(RuntimeError):
    pass


class SingleInstanceLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.acquired = False

    def acquire(self) -> None:
        if self.lock_path.exists():
            try:
                pid = int(self.lock_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = None

            if pid and _process_exists(pid):
                raise SingleInstanceError(
                    "Já existe uma execução usando a câmera. "
                    "Feche a janela atual ou finalize o processo antes de abrir outra."
                )

            self.lock_path.unlink(missing_ok=True)

        self.lock_path.write_text(str(os.getpid()), encoding="utf-8")
        self.acquired = True
        atexit.register(self.release)

    def release(self) -> None:
        if self.acquired:
            try:
                current = self.lock_path.read_text(encoding="utf-8").strip()
                if current == str(os.getpid()):
                    self.lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            finally:
                self.acquired = False


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True
