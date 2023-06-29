"""
将 llama 格式的权重转换为 transformers 格式
"""
import os
import gc
import math
import random
import shutil
import json

import torch
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from utils import parse_target_url
from new_llama import LlamaForCausalLM as NewLlamaForCausalLM

NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}

def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256

def convert_to_hf(args, src_driver, tgt_driver):
    print("Converting to huggingface format...")
    src = args.src
    tgt = args.tgt

    params = src_driver.json_load(os.path.join(src, "params.json"))
    num_shards = NUM_SHARDS[args.model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # new
    if args.new:
        bias = params["bias"]

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    def permute_bias(b):
        return b.view(n_heads, dim // n_heads // 2, 2).transpose(1, 2).reshape(-1)

    # Load weights
    folder = f'/dev/shm/wait_to_upload_weight_tmp_{random.random()}/'
    os.makedirs(folder, exist_ok=True)
    print("Loading weights...")
    if args.model_size == "7B":
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = src_driver.torch_load(os.path.join(src, "tp_0.pt"))
    else:
        # Sharded
        loaded = [
            src_driver.torch_load(os.path.join(src, f"tp_{i}.pt"))
            for i in tqdm(range(num_shards))
        ]
    print("Start converting...")
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in tqdm(range(n_layers)):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if args.model_size == "7B":
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
            if args.new:
                state_dict.update({
                    f"model.layers.{layer_i}.self_attn.q_proj.bias": permute_bias(
                        loaded[f"layers.{layer_i}.attention.wq.bias"]
                    ),
                    f"model.layers.{layer_i}.self_attn.k_proj.bias": permute_bias(
                        loaded[f"layers.{layer_i}.attention.wk.bias"]
                    ),
                    f"model.layers.{layer_i}.self_attn.v_proj.bias": loaded[f"layers.{layer_i}.attention.wv.bias"],
                    f"model.layers.{layer_i}.self_attn.o_proj.bias": loaded[f"layers.{layer_i}.attention.wo.bias"],
                })
        else:
            # Sharded
            # Note that in the 13B checkpoint, not cloning the two following weights will result in the checkpoint
            # becoming 37GB instead of 26GB for some reason.
            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            if args.new:
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = permute_bias(
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wq.bias"].view(n_heads_per_shard, dims_per_head)
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(-1)
                )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            if args.new:
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = permute_bias(
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wk.bias"].view(n_heads_per_shard, dims_per_head)
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(-1)
                )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)
            if args.new:
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wv.bias"]
                        for i in range(num_shards)
                    ],
                    dim=0,
                )

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            if args.new:
                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = \
                loaded[0][f"layers.{layer_i}.attention.wo.bias"]
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        # save state_dict
        tmp_fp = os.path.join(folder, filename)
        torch.save(state_dict, tmp_fp)

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if args.model_size == "7B":
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    tmp_fp = os.path.join(folder, filename)
    torch.save(state_dict, tmp_fp)

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    with open(os.path.join(folder, "pytorch_model.bin.index.json"), "w") as fp:
        json.dump(index_dict, fp)

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    if args.new:
        config.bias = params["bias"]
    if params["vocab_size"] != -1:
        config.vocab_size = params["vocab_size"]
    config.save_pretrained(folder)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()
    if args.new:
        print("Loading the checkpoint in a new Llama model.")
        model = NewLlamaForCausalLM.from_pretrained(folder, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        print("Loading the checkpoint in a Llama model.")
        model = LlamaForCausalLM.from_pretrained(folder, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print(f"Saving to temp folder {folder} in the Transformers format.")
    tmp_folder = os.path.join(folder, "tmp")
    model.save_pretrained(tmp_folder)
    tgt_url = parse_target_url(args)
    os.system(f'/mnt/cache/share/sensesync cp {tmp_folder}/ {tgt_url}')
    shutil.rmtree(folder)