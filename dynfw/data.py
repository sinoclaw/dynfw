"""数据加载（字符级语料）。v0.0.1 冻结基线用 /data/bdh/input.txt（可换路径）。"""
import numpy as np
import torch


def load_data(path='/data/bdh/input.txt'):
    """mmap 读取字节语料（字符级）。"""
    return np.memmap(path, dtype=np.uint8, mode='r')


def get_batch(data, block=256, batch=16, rng=np.random.RandomState(0), split='train'):
    """随机抽 batch：x=[B,block] token ids, y=shifted 下一个 token。"""
    n = len(data); cut = int(0.9*n); d = data[:cut] if split == 'train' else data[cut:]
    ix = rng.randint(0, len(d)-block, (batch,))
    x = torch.stack([torch.from_numpy(d[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x, y
