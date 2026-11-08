import sys, json, os, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.composition.runner import _load_memmap, _write_memmap_metadata
d=Path(tempfile.mkdtemp())
d.mkdir(parents=True,exist_ok=True)
p=d/'gallery_x.npy'
# simulate: gallery encoded for a 100-row gallery under fp A
arr=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(100,4))
arr[:]=1.0; arr.flush()
fpA='A'*64
_write_memmap_metadata(p,fpA,arr)
got=_load_memmap(p,fpA)                 # no shape arg, exactly as _encode_gallery calls it
print('gallery cache reused with NO shape check:', got is not None, 'shape', None if got is None else got.shape)
# now the same file, checked the way _encode_queries checks
got2=_load_memmap(p,fpA,(200,4))
print('query-style call with expected (200,4):', got2 is not None)
# different fingerprint -> refused (so model/gallery changes DO invalidate)
print('different fingerprint refused:', _load_memmap(p,'B'*64) is None)
