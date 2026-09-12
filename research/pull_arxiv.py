import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, json, sys
ns={'a':'http://www.w3.org/2005/Atom','arxiv':'http://arxiv.org/schemas/atom'}
UA={'User-Agent':'Mozilla/5.0 (research)'}
def q(query, n=12, sort='relevance'):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
        {'search_query':query,'max_results':n,'sortBy':sort,'sortOrder':'descending'})
    for attempt in range(3):
        try:
            r=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40)
            return ET.fromstring(r.read())
        except Exception as e:
            print(f"   [retry {attempt+1}: {type(e).__name__}]"); time.sleep(4)
    return None
def show(root,tag):
    out=[]
    if root is None: print(f"### {tag}: FAILED"); return out
    for e in root.findall('a:entry',ns):
        t=e.find('a:title',ns).text.strip().replace('\n',' ')
        aid=e.find('a:id',ns).text.strip().split('/abs/')[-1]
        pub=e.find('a:published',ns).text[:10]
        au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
        cat=e.find('arxiv:primary_category',ns)
        cat=cat.get('term') if cat is not None else ''
        com=e.find('arxiv:comment',ns)
        com=(com.text or '').strip().replace('\n',' ')[:110] if com is not None else ''
        out.append(dict(id=aid,title=t,date=pub,authors=au[:4],n_au=len(au),cat=cat,comment=com,
                        abs=e.find('a:summary',ns).text.strip().replace('\n',' ')))
    print(f"\n{'='*100}\n### {tag}  ({len(out)} hits)\n{'='*100}")
    for i,o in enumerate(out,1):
        print(f"{i:>2}. [{o['id']}] {o['date']} ({o['cat']})")
        print(f"    {o['title']}")
        print(f"    {', '.join(o['authors'])}{' et al.' if o['n_au']>4 else ''}")
        if o['comment']: print(f"    comment: {o['comment']}")
    return out
QUERIES=[
 ('vocabulary scaling law language model','all:"vocabulary size" AND all:"scaling law"'),
 ('tokenizer-free / byte-level','all:"tokenizer-free" OR all:"token-free"'),
 ('byte latent transformer','ti:"byte latent transformer"'),
 ('ByT5 byte-to-byte','ti:"ByT5"'),
 ('tokenizer transfer / transplantation','all:"tokenizer transfer" OR all:"tokenizer transplantation"'),
 ('vocabulary transfer cross-lingual','all:"vocabulary transfer" OR all:"vocabulary expansion" AND all:"language model"'),
 ('tokenizer quality downstream','all:"tokenizer" AND all:"downstream performance" AND all:"language model"'),
 ('byte-level BPE vs subword','all:"byte-level" AND all:"BPE" AND all:"language model"'),
]
ALL={}
for tag,query in QUERIES:
    r=q(query,12)
    ALL[tag]=show(r,tag)
    time.sleep(3.2)
json.dump(ALL,open('arxiv_vocab_route.json','w'),ensure_ascii=False,indent=1)
print("\n\nsaved -> /data/dynfw/research/arxiv_vocab_route.json")
