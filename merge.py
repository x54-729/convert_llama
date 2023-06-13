"""
将 colossalai 流水线的权重合并为 llama 的模型并行格式
"""
import os
import re
import random

import torch
from tqdm import tqdm

def merge_pp(folder, driver, tp, max_pp):
    """
    给定一个 folder ，merge 下面的 pipeline model
    """
    layer_shift = 0

    tp_states = {}
    for pp in tqdm(range(max_pp)):
        _layer_shift = 0
        model_name = f'model_tp{tp}_pp{pp}.pt'
        states = driver.torch_load(os.path.join(folder, model_name))
        keys = list(states.keys())
        for key in keys:
            match = re.search('\.\d+\.', key)
            if match is not None:  # 说明是 layer 相关的, 需要shift
                s, e = match.span()
                layer_idx = int(key[s+1:e-1]) + layer_shift
                _layer_shift = max(_layer_shift, int(key[s+1:e-1]))
                name = key[:s] + f'.{layer_idx}.' + key[e:]
                tp_states[name] = states[key]
            else:
                tp_states[key] = states[key]
        layer_shift += _layer_shift + 1
    
    return tp_states

basic_config = dict(
    num_chunks=1, checkpoint=False, dtype=torch.half, embed_split_hidden=False, 
    num_layers=40, hidden_size=5120, vocab_size=150494, embed_grad_scale=1,
    parallel_output=False, num_attention_heads=40, mlp_ratio=8/3, apply_post_layer_norm=False, 
    no_bias=True, deepnorm=False, residual_in_fp32=False,
    norm_type='rmsnorm', drop_rate=0, attn_drop_rate=0, model_type='llama'
)

def merge(args, src_driver, tgt_driver):
    src = args.src
    tgt = args.tgt
    print(f"Merge from {src}...")

    # Config file
    print('Config loading', flush=True)
    config_file = os.path.join(src, 'model_config.pt')
    assert src_driver.isfile(config_file), "Need config file!"
    update_config = src_driver.torch_load(config_file)
    model_config = basic_config
    model_config.update(update_config)
    print('Config loaded.', flush=True)

    confs = {"dim": model_config['hidden_size'], "multiple_of": 256, "n_heads": model_config['num_attention_heads'], "n_layers": model_config['num_layers'], "norm_eps": 1e-05, "vocab_size": -1}
    tgt_driver.json_save(confs, os.path.join(tgt, 'params.json'))

    assert src_driver.isdir(src), 'not a folder.'
    fns = src_driver.list(src)
    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'):
            # 加入 _t 是为了避免和model_config.py冲突
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_tp = max(max_tp, int(tp[2:])+1)
        max_pp = max(max_pp, int(pp[2:])+1)

    print('Start merging.', flush=True)
    print("Ready to save weights...", flush=True)

    assert len(tgt.split("://")) == 2
    prefix, path = tgt.split("://")
    parts = path.split(os.sep)
    bucket_name = parts[0]
    path = os.sep.join(parts[1:]).strip('/')
    folder = f'/dev/shm/wait_to_upload_weight_tmp_{random.random()}/'
    os.makedirs(folder, exist_ok=True)
    try:
        for tp in tqdm(range(max_tp)):
            idx = tp
            state = merge_pp(src, src_driver, tp, max_pp)
            state = {key[6:]:value for key,value in state.items()}

            current_states = {}

            for i in range(confs['n_layers']):
                name = f'layers.{i}.attention.Wqkv.weight'
                wqkv = state.pop(name).reshape(confs['n_heads']//max_tp, 3, -1, confs['dim'])

                current_states[f'layers.{i}.attention.wq.weight'] = wqkv[:, 0].reshape(-1, confs['dim'])
                current_states[f'layers.{i}.attention.wk.weight'] = wqkv[:, 1].reshape(-1, confs['dim'])
                current_states[f'layers.{i}.attention.wv.weight'] = wqkv[:, 2].reshape(-1, confs['dim'])
                state.pop(f'layers.{i}.attention.rotary_emb.inv_freq')

            current_states.update(state)

            tmp_fp = os.path.join(folder, f'tp_{idx}.pt')
            torch.save(current_states, tmp_fp)
            # sensesync copy
            if args.ak is not None and args.sk is not None:
                tgt_url = f"{prefix}://{args.ak}:{args.sk}@{bucket_name}.{args.bucket_ip}/{path}/"
            else:
                tgt_url = f"{prefix}://{bucket_name}.{args.bucket_ip}/{path}/"
            os.system(f'/mnt/cache/share/sensesync cp {folder} {tgt_url}')
            os.remove(tmp_fp)
    finally:
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)
