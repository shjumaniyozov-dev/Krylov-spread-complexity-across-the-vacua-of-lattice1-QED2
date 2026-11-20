import numpy as np, time
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
_cache={}
def model(kind,N):
    key=(kind,N)
    if key not in _cache:
        S=Schwinger(N) if kind=="full" else SchwingerQLM(N)
        P=partners(S) if kind=="full" else partnersQ(S)
        _cache[key]=(S,P)
    return _cache[key]
def coeffs(kind,N,x,mu,l,nmax,bits=None):
    S,P=model(kind,N)
    if bits is None: bits=max(600,int(3*nmax)+200)
    a,b,info=hp_lanczos(S,x,mu,l,nmax,bits=bits,P=P,check_every=0)
    return a,b
def complexity(a,b,ts):
    Pn=chain_evolve(a,b,ts)
    n=np.arange(len(a))
    C=Pn@n
    tail=Pn[:,int(0.9*len(a)):].sum(1)
    return C,Pn,tail
