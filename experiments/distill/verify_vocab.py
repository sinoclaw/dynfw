"""公平对比: 自训词表 vs Qwen 词表 (同一批测试文本, 一致的方法)。
用 tokenizers.Encoding 的 ids 字段, 保证两边一致。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tokenizers import Tokenizer

def load_tok(path):
    return Tokenizer.from_file(path)

def metrics_toklib(tok, texts):
    """用 tokenizers 库, 统一 encode 接口"""
    total_tok = total_char = total_byte = 0
    single_tok_words = total_words = 0
    for t in texts:
        enc = tok.encode(t)
        ids = enc.ids  # tokenizers.Encoding.ids
        total_tok += len(ids)
        total_char += len(t)
        total_byte += len(t.encode("utf-8"))
        for w in t.split():
            total_words += 1
            if len(tok.encode(w).ids) == 1:
                single_tok_words += 1
    fert = total_tok / max(total_char, 1)
    bpt = total_byte / max(total_tok, 1)
    strr = single_tok_words / max(total_words, 1)
    return fert, bpt, strr

TEST_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world of technology.",
    "Language models learn patterns from large amounts of text data.",
    "In 2023, researchers published 12345 papers on machine learning.",
    "The city of Beijing has a population of over 21 million people.",
]
TEST_ZH = [
    "人工智能正在改变科技世界的面貌。",
    "语言模型从海量文本中学习规律和知识。",
    "北京是中国的首都，人口超过两千万。",
    "2023年研究人员发表了12345篇机器学习论文。",
    "机器学习是人工智能的一个重要分支。",
]

# 我们的词表
our = load_tok("/data/dynfw/data/our_vocab_65k/tokenizer.json")
print(f"=== 自训词表 (vocab={our.get_vocab_size()}) ===")
fe, be, se = metrics_toklib(our, TEST_EN)
fz, bz, sz = metrics_toklib(our, TEST_ZH)
print(f"  英文: fertility={fe:.3f} bytes/tok={be:.2f} STRR={se:.2f}")
print(f"  中文: fertility={fz:.3f} bytes/tok={bz:.2f} STRR={sz:.2f}")

# Qwen tokenizer.json (从 model 目录加载 tokenizers 库对象)
try:
    qwen = load_tok("/data/models/Qwen3-0.6B/tokenizer.json")
    print(f"\n=== Qwen 词表 (vocab={qwen.get_vocab_size()}) ===")
    fe2, be2, se2 = metrics_toklib(qwen, TEST_EN)
    fz2, bz2, sz2 = metrics_toklib(qwen, TEST_ZH)
    print(f"  英文: fertility={fe2:.3f} bytes/tok={be2:.2f} STRR={se2:.2f}")
    print(f"  中文: fertility={fz2:.3f} bytes/tok={bz2:.2f} STRR={sz2:.2f}")
    print(f"\n  === 对比 ===  英文fert {fe:.3f} vs {fe2:.3f} | 中文fert {fz:.3f} vs {fz2:.3f} | 中文bytes/tok {bz:.2f} vs {bz2:.2f} | 英文STRR {se:.2f} vs {se2:.2f}")
except Exception as e:
    print(f"\n(Qwen load via tokenizers: {str(e)[:120]})")
