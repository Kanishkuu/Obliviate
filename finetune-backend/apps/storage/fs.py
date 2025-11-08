import os
from typing import BinaryIO


class LocalStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, key: str, stream: BinaryIO):
        path = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(stream.read())

    def url(self, key: str) -> str:
        return os.path.join(self.base_path, key)
