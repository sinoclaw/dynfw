import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, json
ns={'a':'http://www.w3.org/2005/Atom','arxiv':'http://arxiv.org/schemas/atom'}
UA={'User-Agent':'Mozilla/5.0 (research)'}
def q(query,n=15,sort='relevance'):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
        {'search_query':query,'max_results':n,'sortBy':sort,'sortOrder':'descending'})
    for i in range(3):
        try:
            r=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40)
            return ET.fromstring(r.read())
        except Exception as e:
            time.sleep(4)
    return None
def show(root,tag):
    out=[]
    if root is None: print(f"\n### {tag}: FAILED"); return out
    for e in root.findall('a:entry',ns):
        out.append(dict(id=e.find('a:id',ns).text.strip().split('/abs/')[-1],
            title=e.find('a:title',ns).text.strip().replace('\n',' '),
            date=e.find('a:published',ns).text[:10],
            au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)],
            cat=(e.find('arxiv:primary_category',ns).get('term') if e.find('arxiv:primary_category',ns) is not None else ''),
            com=((e.find('arxiv:comment',ns).text or '').strip().replace('\n',' ')[:100] if e.find('arxiv:comment',ns) is not None else ''),
            abs=e.find('a:summary',ns).text.strip().replace('\n',' ')))
    print(f"\n{'='*100}\n### {tag} ({len(out)})\n{'='*100}")
    for i,o in enumerate(out,1):
        print(f"{i:>2}. [{o['id']}] {o['date']} ({o['cat']}) {o['title']}")
        print(f"    {', '.join(o['au'][:3])}{' et al.' if len(o['au'])>3 else ''} | {o['com']}")
    return out
QS=[("MiniCPM 系列","ti:MiniCPM"),
    ("Ultra-FineWeb 数据","all:Ultra-FineWeb"),
    ("UltraData 数据","all:UltraData AND all:dataset"),
    ("LLM 预训练数据配方","ti:data AND ti:recipe AND cat:cs.CL"),
    ("WSD 学习率 / 数据退火","all:annealing AND all:pretraining AND all:data AND cat:cs.CL")]
ALL={}
for tag,query in QS:
    ALL[tag]=show(q(query,15),tag); time.sleep(3.2)
json.dump(ALL,open('arxiv_facewall_data.json','w'),ensure_ascii=False,indent=1)
print("\nsaved.")
