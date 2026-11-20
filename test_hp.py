import numpy as np, time
from fractions import Fraction as F
from schwinger import *
from hplanczos import hp_lanczos, partners
N=12; S=Schwinger(N); P=partners(S)
t0=time.time()
a1,b1,i1=hp_lanczos(S,'1/2','4/5','3/10',1000,bits=300,P=P)
t1=time.time()
a2,b2,i2=hp_lanczos(S,'1/2','4/5','3/10',1000,bits=600,P=P)
print("time",t1-t0,time.time()-t1, "depths",len(a1),len(a2))
m=min(len(b1),len(b2))
print("max |b(300)-b(600)| over common depth:",np.max(abs(b1[:m]-b2[:m])), " smallest b:",b2.min())
print("orth checks:",i2['orth'][:5], i2['orth'][-3:])
ts=np.linspace(0,400,801)
Pn=chain_evolve(a2,b2,ts); C_hp=Pn@np.arange(len(a2))
np.save('/home/claude/fdp/data/test_N12_Chp.npy',C_hp)
H=S.W(0.5,0.8,0.3); v=S.vacuum(); Hd=H.toarray()
for tol in [1e-16,1e-24,1e-28,0]:
    E,w=spectral_measure(Hd,v,tol=tol); a,b,K=krylov_from_measure(E,w)
    C=spread_complexity(E,w,K,ts)
    d=abs(C-C_hp); print("measure tol",tol,len(E),[float(d[ts<=T].max()) for T in [10,50,400]])
a3,b3,Q=lanczos_sparse(H,v,923)
Pn3=chain_evolve(a3,b3,ts); C3=Pn3@np.arange(len(a3))
d=abs(C3-C_hp); print("double FRO Lanczos:",[float(d[ts<=T].max()) for T in [10,50,400]])
print("b_hp[:10]",b2[:10]); print("b_FRO[:10]",b3[:10])
print("first index where |b_FRO-b_hp|>1e-6:", np.argmax(abs(b3[:len(b2)]-b2[:len(b3)])>1e-6))
