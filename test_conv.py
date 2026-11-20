import numpy as np, time
from krylov_run import *
from scipy.linalg import eigh_tridiagonal
ts=np.linspace(0,4000,2001)
for N,ls in [(10,['3/10','1/2','9/10']),(12,['3/10','1/2','9/10'])]:
    S,P=model("full",N)
    for l in ls:
        out=[]
        for bits in [int(2.6*S.D)+600, int(6*S.D)+1000]:
            a,b=coeffs("full",N,'1/10','2/5',l,S.D,bits=bits)
            C,Pn,tail=complexity(a,b,ts)
            ev,U=eigh_tridiagonal(a,b); Ci=float(np.arange(len(a))@((U**2)@(U[0,:]**2)))
            out.append((len(a),C,Ci,b))
        (d1,C1,i1,b1),(d2,C2,i2,b2)=out
        m=min(len(b1),len(b2)); db=abs(b1[:m]-b2[:m]); first=np.argmax(db>1e-12) if np.any(db>1e-12) else m
        print(N,l,"depths",d1,d2," max|dC|=%.2e  Cinf %.10f %.10f  first n with db>1e-12: %d"%(np.max(abs(C1-C2)),i1,i2,first),flush=True)
