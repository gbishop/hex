import os
from sb3_contrib import MaskablePPO


class ModelManager:
    """List and save models"""

    def __init__(self, base: str, size: int):
        self.size = size
        self.base = base
        self.dir = f"{base}-{size}"
        os.makedirs(self.dir, exist_ok=True)
        self.next = len(self.list()) + 1

    def list(self):
        """List the models"""
        models = sorted(
            [
                name.replace(".zip", "")
                for name in os.listdir(self.dir)
                if name.endswith(".zip")
            ]
        )
        return models

    def latest(self):
        names = self.list()
        if len(names) > 0:
            return names[-1], self.fullpath(names[-1])
        return "", ""

    def load(self, name, **kwargs):
        path = self.fullpath(name)
        return MaskablePPO.load(path, **kwargs)

    def save(self, model, number=None):
        if number is None:
            number = self.next
            self.next += 1
        name = f"{self.size:02d}-{number:03d}"
        path = f"{self.dir}/{name}"
        model.save(path)
        return name, path

    def new(self, **kwargs):
        return MaskablePPO("MlpPolicy", **kwargs)

    def fullpath(self, name):
        path = f"{self.dir}/{name}.zip"
        return path
