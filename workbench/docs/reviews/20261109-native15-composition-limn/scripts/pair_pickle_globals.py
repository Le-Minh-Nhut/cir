import zipfile, pickletools, io, sys
LOG='/tmp/pair_globals.out'
log=open(LOG,'w')
class T:
    def __init__(s,*f): s.f=f
    def write(s,x):
        for q in s.f: q.write(x)
    def flush(s):
        for q in s.f: q.flush()
sys.stdout=T(sys.__stdout__,log); sys.stderr=T(sys.__stderr__,log)
CKPT='/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/pair/pair-B1.pt'
z=zipfile.ZipFile(CKPT)
name=[n for n in z.namelist() if n.endswith('data.pkl')][0]
data=z.read(name)
globs=[]
stack=[]
for op,arg,pos in pickletools.genops(data):
    if op.name=='GLOBAL':
        globs.append(arg)
    elif op.name=='STACK_GLOBAL':
        if len(stack)>=2:
            globs.append((stack[-2],stack[-1]))
    if op.name in ('SHORT_BINUNICODE','BINUNICODE','UNICODE','SHORT_BINSTRING','BINSTRING'):
        stack.append(arg)
print("GLOBAL_COUNT", len(globs))
seen=set()
for g in globs:
    if g not in seen:
        seen.add(g); print("GLOBAL", g)
# module names referenced
mods=set()
for g in globs:
    if isinstance(g,tuple) and len(g)==2: mods.add(g[0])
    elif isinstance(g,str) and ' ' in g: mods.add(g.split()[0])
print("MODULES:", sorted(mods))
log.flush()
