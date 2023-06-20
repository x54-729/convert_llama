import os

def parse_target_url(args):
    assert len(args.tgt.split("://")) == 2
    prefix, path = args.tgt.split("://")
    parts = path.split(os.sep)
    bucket_name = parts[0]
    path = os.sep.join(parts[1:]).strip('/')

    if args.ak is not None and args.sk is not None:
        tgt_url = f"{prefix}://{args.ak}:{args.sk}@{bucket_name}.{args.bucket_ip}/{path}/"
    else:
        tgt_url = f"{prefix}://{bucket_name}.{args.bucket_ip}/{path}/"

    return tgt_url