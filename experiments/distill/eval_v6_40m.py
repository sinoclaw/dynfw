#!/usr/bin/env python3
"""破 HF 注册卡点: 加载 v6 真实预训练 checkpoint -> HF 目录 -> lm_eval arc_easy 出分。
验证 '自定义架构能被 model=hf 加载 + 标准评测出分' 链路。
"""
import os, sys, json
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/data/dynfw')

import torch
import numpy as np

# 1) 注册 V6 架构 (AutoConfig + AutoModelForCausalLM 双注册必须在 lm_eval 之前)
import experiments.distill.v6hf_adapter as ad  # noqa  (side-effect: registers)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

SRC = '/data/dynfw/results/pretrain_v6_40m_smoke'
cfg = json.load(open(os.path.join(SRC, 'config.json')))
print("cfg:", cfg)

# 2) 构造 HF 目录
hd = os.path.join(SRC, 'hf')
os.makedirs(hd, exist_ok=True)
from experiments.distill.v6hf_adapter import V6HFConfig, V6HFModel
hfc = V6HFConfig(vocab_size=cfg['vocab'], D=cfg['D'], n_layer=cfg['n_layer'],
                 nh=cfg['nh'], mlp_mult=cfg['mlp_mult'], fw_arch='v6')
model = V6HFModel(hfc)
sd = torch.load(os.path.join(SRC, 'student.pt'), map_location='cpu')
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
model.save_pretrained(hd)   # 写 config.json(model_type=v6_dynfw) + pytorch_model.bin
print("saved HF dir:", hd, sorted(os.listdir(hd)))

# 3) 验证 AutoModelForCausalLM 能按 model_type 加载
m2 = AutoModelForCausalLM.from_pretrained(hd, trust_remote_code=True)
print("加载成功, type:", type(m2).__name__)
print("loaded config model_type:", m2.config.model_type)

# 4) lm_eval 评测 (arc_easy, limit=8 冒烟)
from lm_eval import simple_evaluate
res = simple_evaluate(
    model="hf",
    model_args=f"pretrained={hd},tokenizer=/data/models/Qwen3-0.6B,dtype=float32",
    tasks=["arc_easy"], num_fewshot=0, limit=8, device="cuda", batch_size=2,
)
r = res["results"]["arc_easy"]
print("=== arc_easy ===")
print("acc,none   =", r.get("acc,none"))
print("acc_norm,none =", r.get("acc_norm,none"))
print("[EVAL_DONE]")
