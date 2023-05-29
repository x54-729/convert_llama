"""
从 OSS 对象存储文件夹中下载文件
"""
import argparse
import os

from io_driver import PetrelIODriver, FileIODriver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--tgt")
    args = parser.parse_args()

    assert PetrelIODriver.isdir(args.src)
    for filename in PetrelIODriver.list(args.src):
        print(f"Downloading {filename}")
        obj = PetrelIODriver.load(os.path.join(args.src, filename))
        FileIODriver.save(obj, os.path.join(args.tgt, filename))
