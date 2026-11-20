import numpy as np
from krylov_run import *
ts=np.linspace(0,1000,2001)
S,P=model("full",12)
res={}
for bits in [6000,9000]:
    a,b=coeffs("full",12,'1/2','4/5','3/10',S.D,bits=bits); C,_,_=complexity(a,b,ts); res[bits]=(a,b,C)
print("depths",len(res[6000][0]),len(res[9000][0]),"max|C6000-C9000|",np.max(abs(res[6000][2]-res[9000][2])))
H=S.W(0.5,0.8,0.3); v=S.vacuum()
ad,bd,_=lanczos_sparse(H,v,S.D-1); Cd,_,_=complexity(ad,bd,ts)
E,w=spectral_measure(H.toarray(),v,tol=1e-16); aa,bb,K=krylov_from_measure(E,w); Cm=spread_complexity(E,w,K,ts)
Ch=res[9000][2]
print("double FRO max|dC|",np.max(abs(Cd-Ch)),"measure(1e-16) max|dC|",np.max(abs(Cm-Ch)))
np.savez("../data/precision2.npz",ts=ts,Ch=Ch,Cd=Cd,Cm=Cm,b=res[9000][1],bd=bd)
