import argparse
import os
import random
import shutil
import warnings

from transformers import LlamaTokenizer

from io_driver import choose_driver

try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None


def write_tokenizer(args, src_drvier, tgt_driver):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {args.tgt}.")
    tokenizer = tokenizer_class(args.src)
    if tgt_driver.is_local():
        tokenizer.save_pretrained(args.tgt)
    else:
        folder = f'/dev/shm/wait_to_upload_weight_tmp_{random.random()}/'
        os.makedirs(folder, exist_ok=True)
        try:
            assert len(args.tgt.split("://")) == 2
            prefix, path = args.tgt.split("://")
            parts = path.split(os.sep)
            bucket_name = parts[0]
            path = os.sep.join(parts[1:]).strip('/')
            folder = f'/dev/shm/wait_to_upload_weight_tmp_{random.random()}/'
            print(f"Saving to temp folder {folder}")
            tokenizer.save_pretrained(folder)
            if args.ak is not None and args.sk is not None:
                tgt_url = f"{prefix}://{args.ak}:{args.sk}@{bucket_name}.{args.bucket_ip}/{path}/"
            else:
                tgt_url = f"{prefix}://{bucket_name}.{args.bucket_ip}/{path}/"
            os.system(f'/mnt/cache/share/sensesync cp {folder}/ {tgt_url}')
        finally:
            shutil.rmtree(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="tokenizer 的路径")
    parser.add_argument("--tgt", help="输出路径")
    parser.add_argument("--bucket_ip", help="对象存储桶的ip地址")
    parser.add_argument("--ak", default=None, help="OSS 对象存储的 Access Key")
    parser.add_argument("--sk", default=None, help="OSS 对象存储的 Secret Key")
    args = parser.parse_args()
    assert "s3://" not in args.src, "args.src needs to be a local path"
    src_driver = choose_driver(args.src)
    tgt_driver = choose_driver(args.tgt)
    write_tokenizer(args, src_driver, tgt_driver)
