import numpy as np, time
from schwinger import *
from hplanczos import hp_lanczos
for N in [12,16,20,24]:
    Q=SchwingerQLM(N); print(N,"QLM dim",Q.D)
N=14; Q=SchwingerQLM(N)
# hp lanczos needs partners restricted to QLM basis
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
P=partnersQ(Q)
a_h,b_h,info=hp_lanczos(Q,'1/2','4/5','3/10',Q.D,bits=2000,P=P,check_every=50)
print("hp depth",len(a_h),"orth",info['orth'][-3:])
H=Q.W(0.5,0.8,0.3); v=Q.vacuum()
a_d,b_d,_=lanczos_sparse(H,v,Q.D-1)
m=min(len(b_h),len(b_d)); d=abs(b_h[:m]-b_d[:m])
print("double FRO depth",len(a_d),"first n |db|>1e-8:",np.argmax(d>1e-8) if np.any(d>1e-8) else 'none', "max",d.max())
ts=np.linspace(0,1000,2001)
Ch=chain_evolve(a_h,b_h,ts)@np.arange(len(a_h)); Cd=chain_evolve(a_d,b_d,ts)@np.arange(len(a_d))
print("max |C_hp - C_double| :",np.max(abs(Ch-Cd)), " C range",Ch.min(),Ch.max())
E,w=spectral_measure(H.toarray(),v,tol=1e-28); a,b,K=krylov_from_measure(E,w)
Cm=spread_complexity(E,w,K,ts); print("measure route diff:",np.max(abs(Cm-Ch)), "levels",len(E))
print("spectral width W:",E.min(),E.max())
