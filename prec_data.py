import numpy as np
from krylov_run import *
ts=np.linspace(0,1000,2001)
out={}
for l,lf in [('1/2',0.5),('3/10',0.3)]:
    S,P=model("full",12)
    a,b=coeffs("full",12,'1/10','2/5',l,S.D,bits=6000)
    Ch,_,_=complexity(a,b,ts)
    H=S.W(0.1,0.4,lf); v=S.vacuum()
    ad,bd,_=lanczos_sparse(H,v,S.D-1)
    Cd,_,_=complexity(ad,bd,ts)
    m=min(len(b),len(bd))
    out[l]=dict(Ch=Ch,Cd=Cd,b=b,bd=bd)
    print(l,"exact depth",len(a),"double depth",len(ad),"max|dC|",np.max(abs(Ch-Cd)),"first n |db|>1e-6:",np.argmax(abs(b[:m]-bd[:m])>1e-6))
np.savez("../data/precision.npz",ts=ts,**{f"{k}_{q}":v[q] for k,v in out.items() for q in v})
