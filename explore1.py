import numpy as np, time
from krylov_run import *
ts=np.linspace(0,3000,6001)
for l in ['0','3/10','9/20','1/2','11/20','3/5','7/10','9/10']:
    t0=time.time()
    a,b=coeffs("full",12,'1/10','2/5',l,900)
    C,Pn,tail=complexity(a,b,ts)
    ok=ts[tail<1e-8]; tmax=ok.max() if len(ok) else 0
    print("l=%5s depth=%d  C(100)=%.4f C(500)=%.4f C(3000)=%.4f  mean[0,3000]=%.4f  max=%.3f  tail_max=%.1e  (%.1fs)"%(l,len(a),C[200],C[1000],C[-1],C.mean(),C.max(),tail.max(),time.time()-t0),flush=True)
