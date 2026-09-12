import urllib.request, re, html, sys
UA={'User-Agent':'Mozilla/5.0'}
def get(u):
    r=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60)
    t=r.read().decode('utf-8','ignore')
    t=re.sub(r'<script.*?</script>','',t,flags=re.S)
    t=re.sub(r'<style.*?</style>','',t,flags=re.S)
    t=re.sub(r'<[^>]+>',' ',t)
    t=html.unescape(t); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    return t
pats = {
 '2407.13623': [r'N[_ ]?opt', r'V[_ ]?opt', r'\bC\^?\{?[0-9.]+\}?\b', r'compute-optimal', r'IsoFLOP', r'216K', r'32K', r'43K', r'power law', r'\.\d+\s*\\?alpha', r'vocabulary size.*?scaling', r'fit', r'exponent'],
 '2501.16975': [r'log-linear', r'input vocabulary', r'double-sized', r'perplexity', r'loss', r'\.\d+\s*nats', r'over-tokeniz', r'multi-gram'],
}
for aid, ps in pats.items():
    url=f'https://arxiv.org/html/{aid}v1'
    try: t=get(url)
    except Exception as e:
        print(f"### {aid}: fetch fail {e}"); continue
    print("="*100); print(f"### {aid}  (len {len(t)})"); print("="*100)
    sents=re.split(r'(?<=[.;])\s+', t)
    seen=set(); out=[]
    for s in sents:
        s=s.strip()
        if not (40<len(s)<420): continue
        sc=sum(1 for p in ps if re.search(p,s,re.I))
        if sc>=1:
            key=s[:90]
            if key in seen: continue
            seen.add(key); out.append((sc,s))
    out.sort(key=lambda x:-x[0])
    for sc,s in out[:45]:
        print(f"[{sc}] {s}\n")
