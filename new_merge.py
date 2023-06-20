"""
将 colossalai 流水线的权重合并为 llama 的模型并行格式。用于新版的权重。
"""
import os, random, re

import torch
from tqdm import tqdm

from utils import parse_target_url

def merge_pp(folder, driver):
    """
    给定一个 folder ，merge 下面的 pipeline model

    """
    assert driver.isdir(folder), 'not a folder.'
    fns = driver.list(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'): # 加入 _t 是为了避免和model_config.py冲突
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_tp = max(max_tp, int(tp[2:])+1)
        max_pp = max(max_pp, int(pp[2:])+1)

    full_states = []
    for tp in tqdm(range(max_tp)):
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
        full_states.append({(key[6:] if key.startswith('model.') else key):value for key,value in tp_states.items()})
    return full_states  # List[{}]，其中元素的长度是 tp 的数量

basic_config = dict(
    num_chunks=1, checkpoint=False, dtype=torch.half, embed_split_hidden=False, 
    num_layers=40, hidden_size=5120, vocab_size=150494, embed_grad_scale=1,
    parallel_output=False, num_attention_heads=40, mlp_ratio=8/3, apply_post_layer_norm=False, 
    no_bias=True, deepnorm=False, residual_in_fp32=False,
    norm_type='rmsnorm', drop_rate=0, attn_drop_rate=0, model_type='llama'
)

def new_merge(args, src_driver, tgt_driver):
    src = args.src
    tgt = args.tgt
    print(f"Merge from {src}...")

    # Config file
    print('Config loading', flush=True)
    # read config, needed to split Wqkv
    config_file = os.path.join(src, 'model_config.pt')
    assert src_driver.isfile(config_file), "Need config file!"
    update_config = src_driver.torch_load(config_file)
    model_config = basic_config
    model_config.update(update_config)
    print('Config loaded.', flush=True)
    model_config['dtype'] = torch.half if str(model_config['dtype']) == 'torch.float16' else torch.bfloat16
    model_config['parallel_output'] = False

    tmp_folder = f'/dev/shm/wait_to_upload_weight_tmp/{random.random()}/'
    os.makedirs(tmp_folder, exist_ok=True)

    param_obj = {
        'dim': model_config['hidden_size'], 'multiple_of': 256, 'n_heads': model_config['num_attention_heads'],
        'n_layers': model_config['num_layers'], 'norm_eps': 1e-06, 'vocab_size': model_config['vocab_size'], 'bias':True
    }

    try:
        statess = merge_pp(src, src_driver)
        world_size = len(statess)
        assert world_size > 0, '?'
        
        if 'embedding.word_embeddings.weight' in statess[0]:
            embedding_key = 'embedding.word_embeddings.weight'
        elif 'embedding.weight' in statess[0]:
            embedding_key = 'embedding.weight'
        else:
            print('Check embedding states\'names in below:', flush=True)
            print(list(statess[0].keys()), flush=True)

        size_0, size_1 = statess[0][embedding_key].shape
        embdim_pertp = size_1 // world_size
        tok_emb_list = [
                            torch.concat(
                                [   
                                    statess[tp][embedding_key]\
                                        [:,embdim_pertp*local_rank:embdim_pertp*(local_rank + 1)]
                                    for tp in range(world_size)
                                ], 
                                dim=0
                            )
                            for local_rank in range(world_size)
                        ]
        
        share_biass = [
            statess[0].pop(f'blocks.{i}.mixer.out_proj.bias')
            for i in range(param_obj['n_layers'])
        ]

        for tp in range(world_size):
            current_states = {}
            current_states['tok_embeddings.weight'] = tok_emb_list[tp]
            statess[tp].pop(embedding_key)
            current_states['output.weight'] = statess[tp].pop('head.weight')
        
            for i in range(param_obj['n_layers']):
                wqkv = statess[tp].pop(f'blocks.{i}.mixer.Wqkv.weight').\
                    reshape(3, param_obj['n_heads']//world_size, -1, param_obj['dim'])
                bqkv = statess[tp].pop(f'blocks.{i}.mixer.Wqkv.bias').\
                    reshape(3, param_obj['n_heads']//world_size, -1)
                
                current_states[f'layers.{i}.attention.wq.weight'] = wqkv[0].reshape(-1, param_obj['dim'])
                current_states[f'layers.{i}.attention.wq.bias'] = bqkv[0].reshape(-1)
                current_states[f'layers.{i}.attention.wk.weight'] = wqkv[1].reshape(-1, param_obj['dim'])
                current_states[f'layers.{i}.attention.wk.bias'] = bqkv[1].reshape(-1)
                current_states[f'layers.{i}.attention.wv.weight'] = wqkv[2].reshape(-1, param_obj['dim'])
                current_states[f'layers.{i}.attention.wv.bias'] = bqkv[2].reshape(-1)

                current_states[f'layers.{i}.attention.wo.weight'] = statess[tp].pop(f'blocks.{i}.mixer.out_proj.weight')
                current_states[f'layers.{i}.attention.wo.bias'] = share_biass[i]

                statess[tp].pop(f'blocks.{i}.mixer.rotary_emb.inv_freq')

                current_states[f'layers.{i}.feed_forward.w1.weight'] = statess[tp].pop(f'blocks.{i}.mlp.w1.weight')
                current_states[f'layers.{i}.feed_forward.w2.weight'] = statess[tp].pop(f'blocks.{i}.mlp.w3.weight')
                current_states[f'layers.{i}.feed_forward.w3.weight'] = statess[tp].pop(f'blocks.{i}.mlp.w2.weight')

                current_states[f'layers.{i}.attention_norm.weight'] = statess[tp].pop(f'blocks.{i}.norm1.weight')
                current_states[f'layers.{i}.ffn_norm.weight'] = statess[tp].pop(f'blocks.{i}.norm2.weight')

            current_states.update(statess[tp])
            tmp_fp = os.path.join(tmp_folder, f'tp_{tp}.pt')
            tgt_driver.torch_save(current_states, tmp_fp)

        tmp_fp = os.path.join(tmp_folder, 'params.json')
        tgt_driver.json_save(param_obj, tmp_fp)
                
        tgt_url = parse_target_url(args)
        os.system(f'/mnt/cache/share/sensesync cp {tmp_folder} {tgt_url}')
    finally:
        if os.path.exists(tmp_folder):
            import shutil
            shutil.rmtree(tmp_folder)
