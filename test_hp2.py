import numpy as np, time
from schwinger import *
from hplanczos import hp_lanczos, partners
N=12; S=Schwinger(N); P=partners(S)
res={}
for bits in [300,600,1200]:
    t0=time.time(); a,b,i=hp_lanczos(S,'1/2','4/5','3/10',500,bits=bits,P=P,check_every=25)
    res[bits]=(a,b); print(bits,"time %.1f"%(time.time()-t0)," orth:",[ (n,'%.0e'%o) for n,o in i['orth'] if n%100==0 or o>1e-30][:12])
for (p,q) in [(300,600),(600,1200)]:
    d=abs(res[p][1]-res[q][1]); print(p,q,"first n with |db|>1e-10:",np.argmax(d>1e-10))
a,b=res[1200]
ts=np.linspace(0,400,801)
for ncut in [150,200,250,300]:
    Pn=chain_evolve(a[:ncut+1],b[:ncut],ts); C=Pn@np.arange(ncut+1)
    print(ncut,"C(10,50,100,400)=",C[20],C[100],C[200],C[-1]," max p_last:",Pn[:,-20:].sum(1).max())
