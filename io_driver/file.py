import os
import json
import torch
import shutil

from .base import IODriver

class FileIODriver(IODriver):
    @staticmethod
    def load(path: str):
        assert os.path.exists(path), f"File {path} does not exist."
        with open(path, 'r') as f:
            return f.read()

    @staticmethod
    def torch_load(path: str):
        assert os.path.exists(path), f"File {path} does not exist."
        return torch.load(path, map_location=torch.device('cpu'))

    @staticmethod
    def json_load(path: str):
        assert os.path.exists(path), f"File {path} does not exist."
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save(obj, path: str, append: bool = False):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        if append:
            with open(path, 'a+') as f:
                f.write(obj)
        else:
            with open(path, 'w+') as f:
                f.write(obj)

    @staticmethod
    def torch_save(obj, path: str):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(obj, path)

    @staticmethod
    def json_save(obj, path: str):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f)

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)
    
    @staticmethod
    def isfile(path: str) -> bool:
        return os.path.isfile(path)
    
    @staticmethod
    def isdir(path: str) -> bool:
        return os.path.isdir(path)

    @staticmethod
    def list(path: str):
        return os.listdir(path)

    @staticmethod
    def delete(path: str):
        shutil.rmtree(path)

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        os.makedirs(path, exist_ok=exist_ok)

    @staticmethod
    def is_local():
       return True
