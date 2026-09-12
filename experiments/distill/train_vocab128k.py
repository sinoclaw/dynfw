"""训家族通用词表 128K —— 用 tokenizers 文件 train() (C加速, 完成更稳) + 数据抽样。
英文 wikitext-103 + 中文 zh-wiki + SkyPile 各抽样合并成一个语料文件，再 train()。
注意: train() 走 C 实现，比 train_from_iterator 快 10-100 倍。"""
import os, random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB = 131072
SPECIAL = ["<|endoftext|>", "<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>",
           "<|think|>", "<|/think|>", "<|tool_call|>", "<|/tool_call|>"]
SPECIAL += [f"<|reserved_{i}|>" for i in range(20)]

# ---- 抽样合并语料到一个文件 (每个源抽 20万行, 控制规模) ----
CORPUS = "/data/dynfw/data/corpus_128k_prep.txt"
random.seed(42)
with open(CORPUS, "w", encoding="utf-8") as out:
    # 英文: 抽样 40 万行
    n = 0
    with open("/data/dynfw/data/wikitext103.txt", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]
    for l in random.sample(lines, min(400000, len(lines))):
        out.write(l + "\n"); n += 1
    # 中文 zh-wiki: 全部 25 万条
    import json
    with open("/data/dynfw/data/zh-wiki.json", encoding="utf-8") as f:
        data = json.load(f)
    for item in data[:255000]:
        out.write(item["completion"] + "\n"); n += 1
    # 中文 SkyPile: 抽样 40 万条
    sp = []
    for jf in ["skypile_zh_0.jsonl", "skypile_zh_1.jsonl"]:
        with open(f"/data/dynfw/data/{jf}", encoding="utf-8") as f:
            for line in f:
                try:
                    sp.append(json.loads(line)["text"])
                except Exception:
                    pass
    for l in random.sample(sp, min(400000, len(sp))):
        out.write(l + "\n"); n += 1
print(f"语料已合并: {CORPUS} 共 {n} 行 ≈ {os.path.getsize(CORPUS)/1e9:.2f}GB", flush=True)

# ---- 训词表 (文件 train, C加速) ----
tok = Tokenizer(BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = Sequence([ByteLevel(add_prefix_space=True), Digits(individual_digits=True)])
tok.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, min_frequency=2, show_progress=True)
print(f"开始训 128K 词表 (文件 train, C加速)...", flush=True)
tok.train([CORPUS], trainer=trainer)

OUT = "/data/dynfw/data/our_vocab_128k"
os.makedirs(OUT, exist_ok=True)
tok.save(f"{OUT}/tokenizer.json")
print(f"词表已保存: {OUT}/tokenizer.json, vocab={tok.get_vocab_size()}", flush=True)
