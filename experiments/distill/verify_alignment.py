"""验证: 用候选 --data 生成 teacher logits, 与 shared_logits/teacher_logits.npy 对比, 找生成时的真实 --data。
如果 max diff 小 → 该 --data 就是生成 teacher_logits 的源(复用正确); 大 → 错位, 之前数字不可信。"""
import numpy as np, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

TEACHER = '/data/models/Qwen3-0.6B'
LOGITS_NPY = '/data/dynfw/results/shared_logits/teacher_logits.npy'

def gen(arr, n, t):
    return np.abs(arr.astype(np.float32)[:n] - t.astype(np.float32)).max()

tok = AutoTokenizer.from_pretrained(TEACHER)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16, device_map='cuda')
teacher.eval()
saved = np.load(LOGITS_NPY, mmap_mode='r')
print('saved teacher_logits shape:', saved.shape)

for data in ['/data/dynfw/data/wikitext103.txt']:
    lines = [l.strip() for l in open(data) if l.strip()][:2000]
    ids = tok(lines, return_tensors='pt', padding=True, truncation=True, max_length=256)['input_ids']
    x = ids.to('cuda')
    n_blocks = saved.shape[0] // 8
    batches = list(x.split(8))[:n_blocks+1]  # batch=8 减少循环
    all_logits = []
    for bx in batches:
        with torch.no_grad():
            lg = teacher(bx).logits.float()
        all_logits.append(lg.cpu().numpy().astype(np.float16))
    arr = np.concatenate(all_logits, axis=0)
    n = saved.shape[0]
    m = arr.shape[0]
    if m >= n:
        d = np.abs(arr.astype(np.float32)[:n] - saved[:n].astype(np.float32)).max()
        print(f'{data}: 生成{arr.shape} vs saved 前{n} max_diff = {d:.4f}  {"✅匹配" if d < 0.5 else "❌不匹配"}')
