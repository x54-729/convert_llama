from .file import FileIODriver
from .petrel import PetrelIODriver

def choose_driver(url):
    """
    选择 Driver。判定方式是判断是否包含 s3://
    TODO
    """
    if "s3://" in url:
        return PetrelIODriver
    else:
        return FileIODriver