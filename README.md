# LLaMA 简易格式转换工具

用于将 `colossalai` 训练的格式转换为 LLaMA 原本的 tp 格式和 `transformers` 格式并放置在集群上。

在权重文件较大的情况下进行转换可能会导致内存不足，请确保在计算节点上运行脚本。

## 文件格式

在源文件夹下的文件应该包含 `model_config.pt` 和命名格式形如 `model_tp1_pp0.pt` 的文件。转换后的仅包含张量并行的权重文件包含 `params.json` 和命名格式形如 `tp_1.pt` 的文件。

## 参数

- `--src`: 输入文件夹，可以是本地路径或对象存储的路径。
- `--tgt`: 输出文件夹，必须是对象存储的路径。
- `--bucket_ip`：对象存储桶的 ip 地址，用于在 `sensesync` 工具进行复制时使用。
- `--model_size`: 模型规模；如果指定了 `--merge_only` 则可以忽略该参数。
- `--ak`: 对象存储的 Access Key，用于在 `sensesync` 工具进行复制时使用。也可以参照手册通过环境变量 `AWS_ACCESS_KEY_ID` 指定。
- `--sk`: 对象存储的 Secret Key，用于在 `sensesync` 工具进行复制时使用。也可以参照手册通过环境变量 `AWS_SECRET_ACCESS_KEY` 指定。
- `--merge_only`: 是否仅执行 `merge` 部分，即将权重合并为张量并行的权重。
- `--from_llama`: 源文件是否是 llama 格式的张量并行权重。可以是通过 `--merge_only` 生成的，也可以是 llama 原始的权重（注意文件命名格式）。

## 运行

该工具会先将 `s3://model_weights/0331/pw_7132k/1317/` 下的权重：

- 转换为 llama 的张量并行格式，保存在 `s3://model_weights/0331/evaluation/exported_llama/pw_7132k/` 中
- 再转换为 `huggingface` 格式，保存在 `s3://model_weights/0331/evaluation/exported_llama/pw_7132k_hf/` 中

```bash
python main.py --src s3://model_weights/0331/pw_7132k/1317 \
               --tgt s3://model_weights/0331/evaluation/exported_llama/pw_7132k \
               --model_size 7B \
               --bucket_ip 10.140.2.204:80 \
               --ak your_ak \
               --sk your_sk
```

下面的参数则会将源文件夹中 llama 格式的权重转换为 huggingface 格式。注意添加了 `--from_llama` 参数后目标文件夹不会自动加上 `_hf` 后缀。

```bash
python main.py --src s3://model_weights/0331/evaluation/exported_llama/pw_7132k \
               --tgt s3://model_weights/0331/evaluation/exported_llama/pw_7132k_hf \
               --model_size 7B \
               --bucket_ip 10.140.2.204:80 \
               --ak your_ak \
               --sk your_sk \
               --from_llama
```

如果想要转换 tokenizer，则可以执行下面的命令：

huggingface 格式的 tokenizer 会被保存在 `tgt` 文件夹下。

```bash
python tokenizer.py --src ./tokenizer.model \
                    --tgt ./tokenizer \
                    --bucket_ip 10.140.27.254:80 \
                    --ak your_ak \
                    --sk your_sk \
```