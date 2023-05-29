import json
from io import BytesIO

import torch

from .base import IODriver
class PetrelIODriver(IODriver):
    @staticmethod
    def load(path: str):
        from petrel_client.client import Client
        client = Client()
        obj = client.get(path)
        return obj.decode()

    @staticmethod
    def torch_load(path: str):
        from petrel_client.client import Client
        client = Client()
        with BytesIO(client.get(path)) as f:
            obj = torch.load(f, map_location='cpu')
        return obj

    @staticmethod
    def json_load(path: str):
        from petrel_client.client import Client
        client = Client()
        obj = client.get(path)
        return json.loads(obj.decode())
            
    @staticmethod
    def save(obj, path: str, append: bool = False):
        from petrel_client.client import Client
        client = Client()
        buffer = BytesIO()
        if append:
            pre_obj = PetrelIODriver.load(path, 'r')
            obj = pre_obj + obj
        buffer.write(obj.encode())
        buffer.seek(0)
        client.put(path, buffer)
        buffer.close()

    @staticmethod
    def torch_save(obj, path: str):
        from petrel_client.client import Client
        client = Client()
        with BytesIO() as f:
            torch.save(obj, f)
            client.put(path, f.getvalue())

    @staticmethod
    def json_save(obj, path: str):
        from petrel_client.client import Client
        client = Client()
        client.put(path, json.dumps(obj).encode('utf-8'))
            
    @staticmethod
    def exists(path: str) -> bool:
        from petrel_client.client import Client
        client = Client()
        return client.contains(path) or client.isdir(path)
    
    @staticmethod
    def isfile(path: str) -> bool:
        from petrel_client.client import Client
        client = Client()
        return client.contains(path)
    
    @staticmethod
    def isdir(path: str) -> bool:
        from petrel_client.client import Client
        client = Client()
        return client.isdir(path)
    
    @staticmethod
    def list(path: str):
        from petrel_client.client import Client
        client = Client()
        return list(client.list(path))
    
    @staticmethod
    def delete(path: str):
        from petrel_client.client import Client
        client = Client()
        client.delete(path)

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        pass