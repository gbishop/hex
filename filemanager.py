import os
from glob import glob
import re


class FileManager:
    def __init__(self, dir, base="model"):
        self.dir = dir
        self.base = base
        self.generation = 0
        os.makedirs(self.dir, exist_ok=True)

    def latest(self):
        files = sorted(glob(f"{self.dir}/*"))
        self.generation = 0
        last = ""
        if files:
            last = files[-1]
            nums = re.findall(r"\d\d\d", last)
            print(f"{nums=}")
            if nums:
                self.generation = int(nums[0])
                print(f"{self.generation=}")

        return last, self.generation

    def save(self, model):
        self.generation += 1
        name = f"{self.dir}/{self.base}{self.generation:03d}"
        model.save(name)
        return name
