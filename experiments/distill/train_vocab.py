"""训家族通用词表 V1 —— 中英双语 byte-level BPE (65K)。
- 英文: /data/dynfw/data/wikitext103.txt (513MB, 已落盘)
- 中文: hf-mirror streaming 拉取 pleisto/wikipedia-cn (不落盘)
- 用 tokenizers.BpeTrainer，流式迭代器，内存高效
配置(参照 Qwen/DeepSeek 范本): byte-level、单数字切分(SplitDigits)、byte fallback、控制token+预留
"""
import os, io, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

# ---- 语料迭代器 ----
def en_iterator(batch_size=1000):
    """英文: wikitext-103 逐行读"""
    with open("/data/dynfw/data/wikitext103.txt", "r", encoding="utf-8", errors="ignore") as f:
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

def zh_iterator(batch_size=1000):
    """中文: zh-wiki.json + SkyPile（2.5GB，主力），逐个文件读"""
    import json
    # 1) 旧 zh-wiki.json
    try:
        with open("/data/dynfw/data/zh-wiki.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        batch = []
        for item in data:
            batch.append(item["completion"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    except Exception as e:
        print(f"[zh-wiki] skip: {str(e)[:60]}")
    # 2) SkyPile 主力中文
    for jf in ["skypile_zh_0.jsonl", "skypile_zh_1.jsonl"]:
        try:
            with open(f"/data/dynfw/data/{jf}", "r", encoding="utf-8") as f:
                batch = []
                for line in f:
                    try:
                        batch.append(json.loads(line)["text"])
                    except Exception:
                        continue
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
        except Exception as e:
            print(f"[{jf}] skip: {str(e)[:60]}")

def mixed_iterator(en_n=1, zh_n=6):
    """中英交叉喂。wikitext-103 116万行≈1.6G char, zh-wiki 25万条≈0.26G char
    中英 char 总量比≈6:1 → 用 en_n:zh_n=1:6 让喂入的 token 数接近均衡
    (每批: 英文1000行×1.4K≈1.4M char, 中文1000条×1K≈1M char → 2批≈平衡)"""
    en_it, zh_it = en_iterator(), zh_iterator()
    for _ in range(300000):  # 足够多的批次上限
        for _ in range(en_n):
            try:
                yield next(en_it)
            except StopIteration:
                return
        for _ in range(zh_n):
            try:
                yield next(zh_it)
            except StopIteration:
                return

# ---- 词表配置 ----
VOCAB = 131072
SPECIAL = [
    "<|endoftext|>", "<|pad|>", "<|unk|>",   # 基础
    "<|im_start|>", "<|im_end|>",            # ChatML 角色
    "<|think|>", "<|/think|>",               # 推理模式
    "<|tool_call|>", "<|/tool_call|>",       # 工具调用
]
# 预留后续扩展空间
RESERVED = [f"<|reserved_{i}|>" for i in range(20)]
SPECIAL += RESERVED

# ---- 建 tokenizer (byte-level BPE + 单数字切分) ----
tok = Tokenizer(BPE(unk_token="<|unk|>"))
# 预切分: ByteLevel + Digits(individual) —— 数字按单数位切，提升数学能力
tok.pre_tokenizer = Sequence([ByteLevel(add_prefix_space=True), Digits(individual_digits=True)])
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=SPECIAL,      # 会占用 vocab 前 N 个 id
    min_frequency=2,
    show_progress=True,
)

print(f"开始训练词表 (vocab={VOCAB}, 中英双流) ...", flush=True)
tok.train_from_iterator(mixed_iterator(), trainer=trainer)

# 保存
OUT = "/data/dynfw/data/our_vocab_128k"
os.makedirs(OUT, exist_ok=True)
tok.save(f"{OUT}/tokenizer.json")
print(f"词表已保存: {OUT}/tokenizer.json, vocab={tok.get_vocab_size()}", flush=True)
