import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="输入文件夹")
    parser.add_argument("--tgt", help="输出文件夹")
    parser.add_argument("--bucket_ip", help="对象存储桶的ip地址")
    parser.add_argument("--model_size", default=None, choices=[None, "7B", "13B", "30B", "65B"], help="对象存储桶的ip地址")
    parser.add_argument("--ak", default=None, help="OSS 对象存储的 Access Key")
    parser.add_argument("--sk", default=None, help="OSS 对象存储的 Secret Key")
    parser.add_argument("--merge_only", type=bool, default=False, help="指定该参数后，程序不会将权重转换为 huggingface 格式")
    parser.add_argument("--from_llama", type=bool, default=False, help="指定该参数后，程序会将 src 视为 llama 格式的权重文件")
    args = parser.parse_args()

    return args