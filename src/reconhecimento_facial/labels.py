import json
from pathlib import Path


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.name_to_id = {}
            self.id_to_name = {}
            return

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.name_to_id = {str(name): int(label_id) for name, label_id in data.items()}
        self.id_to_name = {label_id: name for name, label_id in self.name_to_id.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.name_to_id, file, ensure_ascii=False, indent=2, sort_keys=True)

    def get_or_create_id(self, name: str) -> int:
        clean_name = self._normalize_name(name)

        if clean_name in self.name_to_id:
            return self.name_to_id[clean_name]

        next_id = max(self.id_to_name.keys(), default=0) + 1

        self.name_to_id[clean_name] = next_id
        self.id_to_name[next_id] = clean_name
        self.save()

        return next_id

    def get_name(self, label_id: int) -> str:
        return self.id_to_name.get(int(label_id), "Desconhecido")

    @staticmethod
    def _normalize_name(name: str) -> str:
        clean_name = " ".join(name.strip().split())

        if not clean_name:
            raise ValueError("O nome não pode ficar vazio.")

        return clean_name
