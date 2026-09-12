"""把文本语料用 Qwen3 tokenizer 编码成 .bin（uint32），供长 T 蒸馏使用。

为什么需要：`/data/corpus/tinystories/train.bin` 是 GPT2 词表(50257)，而教师 Qwen3-0.6B 是 151936
→ 词表不匹配无法共享 logits。此处用教师自己的 tokenizer 重新编码，保证同源。

用法: python experiments/distill/prep_corpus_qwen.py --src <txt> --out <bin> --mb 200
输出: <bin>（uint32 token 数组）+ <bin>.meta.json（tokenizer/词表/长度）
"""
import argparse
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/data/corpus/tinystories/TinyStoriesV2-GPT4-train.txt')
    ap.add_argument('--tok', default='/data/models/Qwen3-0.6B/tokenizer.json')
    ap.add_argument('--out', default='/data/dynfw/data/tinystories_qwen.bin')
    ap.add_argument('--mb', type=int, default=200, help='只取源文件前 N MB（0=全部）')
    ap.add_argument('--chunk-lines', type=int, default=20000)
    args = ap.parse_args()

    limit = args.mb * 1024 * 1024 if args.mb > 0 else None
    tk = Tokenizer.from_file(args.tok)
    print(f'tokenizer vocab = {tk.get_vocab_size()}  源: {args.src}  取前 {args.mb} MB')

    t0 = time.time()
    n_tok = 0
    buf = []
    with open(args.src, 'r', encoding='utf-8', errors='ignore') as f, \
            open(args.out, 'wb') as out:
        read_bytes = 0
        lines = []
        for line in f:
            lines.append(line)
            read_bytes += len(line.encode('utf-8', errors='ignore'))
            if len(lines) >= args.chunk_lines:
                enc = tk.encode('\n'.join(lines))
                arr = np.asarray(enc.ids, dtype=np.uint32)
                out.write(arr.tobytes())
                n_tok += arr.size
                buf.append(arr.size)
                lines = []
                print(f'  {n_tok/1e6:8.2f}M tokens  ({time.time()-t0:.0f}s)', flush=True)
            if limit and read_bytes >= limit:
                break
        if lines:
            enc = tk.encode('\n'.join(lines))
            arr = np.asarray(enc.ids, dtype=np.uint32)
            out.write(arr.tobytes())
            n_tok += arr.size

    meta = {'src': args.src, 'tokenizer': args.tok, 'vocab': tk.get_vocab_size(),
            'n_tokens': int(n_tok), 'dtype': 'uint32', 'mb_limit': args.mb,
            'char_per_tok': round(read_bytes / max(n_tok, 1), 3)}
    json.dump(meta, open(args.out + '.meta.json', 'w'), indent=2)
    size = os.path.getsize(args.out)
    print(f'完成: {n_tok:,} tokens → {args.out} ({size/1e6:.1f} MB)  用时 {time.time()-t0:.0f}s')
    print(f'可切 T=8192 的块数: {n_tok // 8192}')


if __name__ == '__main__':
    main()
