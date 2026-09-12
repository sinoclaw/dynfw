import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, json
ns={'a':'http://www.w3.org/2005/Atom','arxiv':'http://arxiv.org/schemas/atom'}
UA={'User-Agent':'Mozilla/5.0'}
def q(query,n=12,sort='relevance'):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
        {'search_query':query,'max_results':n,'sortBy':sort,'sortOrder':'descending'})
    for _ in range(3):
        try:
            return ET.fromstring(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40).read())
        except Exception: time.sleep(4)
    return None
def show(root,tag):
    print(f"\n{'='*100}\n### {tag}\n{'='*100}")
    if root is None: print("FAILED"); return
    for i,e in enumerate(root.findall('a:entry',ns),1):
        t=e.find('a:title',ns).text.strip().replace('\n',' ')
        aid=e.find('a:id',ns).text.strip().split('/abs/')[-1]
        d=e.find('a:published',ns).text[:10]
        au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
        c=e.find('arxiv:comment',ns)
        c=(c.text or '').strip().replace('\n',' ')[:80] if c is not None else ''
        print(f"{i:>2}. [{aid}] {d} {t}")
        print(f"    {', '.join(au[:3])}{' et al.' if len(au)>3 else ''}"+(f" | {c}" if c else ""))
QS=[("涌现能力 原始","ti:emergent AND ti:abilities"),
    ("涌现是幻象","all:\"emergent abilities\" AND all:mirage"),
    ("小模型/受限领域 TinyStories","ti:TinyStories"),
    ("最小规模推理能力","all:reasoning AND all:\"small language models\" AND all:emergence"),
    ("grokking 顿悟","ti:grokking"),
    ("能力密度/参数效率","all:\"intelligence density\" OR all:\"capability density\"")]
for tag,query in QS:
    show(q(query,10),tag); time.sleep(3.2)
