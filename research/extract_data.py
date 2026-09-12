import urllib.request, re, html, sys
UA={'User-Agent':'Mozilla/5.0'}
def get(u):
    r=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=90)
    t=r.read().decode('utf-8','ignore')
    t=re.sub(r'<script.*?</script>','',t,flags=re.S); t=re.sub(r'<style.*?</style>','',t,flags=re.S)
    t=re.sub(r'<[^>]+>',' ',t); t=html.unescape(t)
    t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    return t
KW=[r'\bGB\b',r'\bTB\b',r'token',r'tokens',r'UltraFineWeb',r'Ultra-FineWeb',r'data (set|mixture|recipe|proportion|ratio)',
    r'filter',r'FastText',r'classifier',r'edu',r'Web',r'code',r'math',r'Chinese',r'English',r'pretrain',
    r'stage',r'anneal',r'WSD',r'cosine',r'trillion',r'billion',r'\d+B\b',r'\d+T\b',r'OpenBMB',r'openbmb']
for aid,url in [('2404.06395','https://arxiv.org/html/2404.06395v3'),('2505.05427','https://arxiv.org/html/2505.05427v1')]:
    try: t=get(url)
    except Exception as e:
        print(f"### {aid} FAIL {e}"); continue
    print("="*104); print(f"### {aid}  len={len(t)}"); print("="*104)
    sents=re.split(r'(?<=[.;:])\s+', t)
    seen=set(); out=[]
    for s in sents:
        s=s.strip()
        if not (50<len(s)<400): continue
        sc=sum(1 for p in KW if re.search(p,s))
        if sc>=2:
            k=s[:70]
            if k in seen: continue
            seen.add(k); out.append((sc,s))
    out.sort(key=lambda x:-x[0])
    for sc,s in out[:55]:
        print(f"[{sc}] {s}\n")
