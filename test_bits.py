import numpy as np, time, sys
from schwinger import *
from hplanczos import hp_lanczos, partners
def partnersQ(S):
    P=[]
    for n in range(S.N-1):
        p=-np.ones(S.D,dtype=np.int64)
        for i,s in enumerate(S.states):
            s=int(s)
            if ((s>>n)&1)!=((s>>(n+1))&1):
                t=s^((1<<n)|(1<<(n+1)))
                if t in S.index: p[i]=S.index[t]
        P.append(p)
    return P
cases=[("full",12,'1/10','2/5','1/2'),("full",12,'1/10','2/5','3/10'),("full",12,'1/4','1','1/2'),("qlm",20,'1/10','2/5','1/2'),("qlm",20,'1/10','2/5','3/10')]
for kind,N,x,mu,l in cases:
    S=Schwinger(N) if kind=="full" else SchwingerQLM(N)
    P=partners(S) if kind=="full" else partnersQ(S)
    out={}
    for bits in [500,1000]:
        t0=time.time(); a,b,i=hp_lanczos(S,x,mu,l,400,bits=bits,P=P,check_every=0); out[bits]=b; dt=time.time()-t0
    m=min(len(out[500]),len(out[1000])); d=abs(out[500][:m]-out[1000][:m])
    print(kind,N,x,mu,l,"D=",S.D,"depth",m,"first n |db|>1e-10:",(np.argmax(d>1e-10) if np.any(d>1e-10) else 'none'),"time(1000b) %.1fs"%dt, flush=True)
