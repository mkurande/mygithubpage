import sys,hashlib,urllib.request,urllib.parse,os,re,time
out=sys.argv[1]; titles=[l.strip() for l in open(sys.argv[2]) if l.strip()]
UA="yf23-cad-research/1.0 (https://github.com/mkurande/mygithubpage)"
for t in titles:
    name=t[5:].replace(" ","_"); h=hashlib.md5(name.encode()).hexdigest()
    slug=re.sub(r"[^A-Za-z0-9]+","_",t[5:-4])[:60]+"_"+hashlib.md5(t.encode()).hexdigest()[:6]
    fn=os.path.join(out,slug+".jpg")
    if os.path.exists(fn) and os.path.getsize(fn)>0: print("have",t); continue
    for w in (1920,1280,960,500):
        u=f"https://thumb.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{urllib.parse.quote(name)}/{w}px-{urllib.parse.quote(name)}"
        try:
            d=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=30).read()
            open(fn,"wb").write(d); print("OK",w,t,flush=True); break
        except Exception as e:
            print("  ",w,str(e)[:50],flush=True); time.sleep(1)
    time.sleep(1)
