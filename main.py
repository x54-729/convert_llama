import os

from arguments import parse_args
from io_driver import choose_driver
from merge import merge
from convert import convert_to_hf

if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']

if __name__ == "__main__":
    args = parse_args()
    if args.merge_only and args.from_llama:
        raise ValueError(
            "--merge_only and --from_llama cannot be assigned at the "
            "same time."
        )
    if not args.merge_only and args.model_size is None:
        raise ValueError(
            "--model_size must be assigned to convert weights to huggingface "
            "format."
        )
    src_driver = choose_driver(args.src)
    assert "s3://" in args.tgt, "args.tgt needs to be a path on OSS."
    tgt_driver = choose_driver(args.tgt)

    if not args.from_llama:
        merge(args, src_driver, tgt_driver)
        # change src and tgt
        args.src = args.tgt
        if args.tgt.endswith("/"):
            args.tgt = args.tgt[:-1]
        args.tgt = args.tgt + "_hf"
        src_driver = tgt_driver
    if not args.merge_only:
        convert_to_hf(args, src_driver, tgt_driver)
