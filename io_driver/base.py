from abc import ABC, abstractmethod

class IODriver(ABC):
    @staticmethod
    @abstractmethod
    def load(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def torch_load(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def json_load(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(obj, path: str, append: bool = False):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def torch_save(obj, path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def json_save(obj, path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exists(path: str) -> bool:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def isfile(path: str) -> bool:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def isdir(path: str) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def list(path: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def delete(path: str):
        raise NotImplementedError

    @staticmethod
    def makedirs(path: str, exist_ok: bool = False):
        raise NotImplementedError
