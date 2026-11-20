# second-order (in x) shifted resonance condition for vacuum <-> n-bubble resonances
import numpy as np
from scipy.optimize import brentq
from schwinger import Schwinger
def E2(S,i,x,mu,l):
    d=S.diag(mu,l); row=S.Hop.getrow(i)
    return sum((x*v)**2/(d[i]-d[j]) for j,v in zip(row.indices,row.data) if abs(d[i]-d[j])>1e-12)
def bubble_states(S,n):
    out=[]
    for j in range(1,S.N,2):        # antiparticle at odd j
        k=j+n                        # particle at even k
        if k<S.N:
            s=0
            for q in range(1,S.N,2): s|=1<<q
            s^=(1<<j); s|=(1<<k)
            out.append(S.index[s])
    return out
import sys
N=int(sys.argv[1]) if len(sys.argv)>1 else 10; S=Schwinger(N); x=0.1; mu=0.4
for n in [1,3,5]:
    res=[]
    for bi in bubble_states(S,n):
        f=lambda l: (S.diag(mu,l)[bi]+E2(S,bi,x,mu,l))-(S.diag(mu,l)[S.vac]+E2(S,S.vac,x,mu,l))
        l0=0.5+mu/n
        try: res.append(brentq(f,l0-0.03,l0+0.03))
        except Exception as e: res.append(np.nan)
    print("n=%d  bare l_n=%.4f  shifted (per bubble position):"%(n,0.5+mu/n),np.round(res,4))
