import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, time
from functools import reduce
from schwinger import *

# 1. Cross-check against brute-force Pauli construction (full 2^N space) for N=6
N=6; x=0.7; mu=0.3; l=0.37
I2=np.eye(2); sz=np.diag([1.,-1.]); spl=np.array([[0,1.],[0,0]]); smi=spl.T
def op(o,k): return reduce(np.kron,[o if j==k else I2 for j in range(N)])
# kron ordering: site 0 is leftmost factor; basis index bit (N-1-k) <-> site k; state |up>=index0 => sz=+1
H=np.zeros((2**N,2**N))
for n in range(N-1):
    H+=x*(op(spl,n)@op(smi,n+1)+op(smi,n)@op(spl,n+1))
for n in range(N):
    H+=mu/2*(np.eye(2**N)+(-1)**n*op(sz,n))
for n in range(N-1):
    L=l*np.eye(2**N)+0.5*sum(((-1)**k)*np.eye(2**N)+op(sz,k) for k in range(n+1))
    H+=L@L
Q=sum(op(sz,k) for k in range(N))
idx=[i for i in range(2**N) if abs(Q[i,i])<1e-12]
Hb=H[np.ix_(idx,idx)]
Eb=np.linalg.eigvalsh(Hb)
S=Schwinger(N); Es=np.linalg.eigvalsh(S.W(x,mu,l).toarray())
print("N=6 spectrum max diff vs brute force:", np.max(abs(np.sort(Eb)-np.sort(Es))))

# 2. vacuum energy & b1
for N in [8,10,12]:
    S=Schwinger(N); x=0.4; mu=0.6; l=0.2
    H=S.W(x,mu,l); v=S.vacuum()
    e0=v@(H@v); b1=np.sqrt((H@v)@(H@v)-e0**2)
    print(N,"<W>_vac =",e0,"(N-1)l^2=",(N-1)*l**2," b1^2=",b1**2," x^2(N-1)=",x**2*(N-1))

# 3. Householder vs sparse Lanczos
N=12; S=Schwinger(N); x=0.5; mu=0.8; l=0.3
H=S.W(x,mu,l); v=S.vacuum()
E,w=spectral_measure(H.toarray(),v)
a,b,K=krylov_from_measure(E,w)
a2,b2,Q=lanczos_sparse(H,v,60)
print("D_eff =",len(E)," max|b-b_sparse| (first 60):",np.max(abs(b[:60]-b2[:60]))," max|a-a|:",np.max(abs(a[:61]-a2[:61])))
print("orthonormality of K:",np.max(abs(K.T@K-np.eye(len(E)))))

# 4. massless Schwinger boson mass (continuum g/sqrt(pi)=0.5642)
for N,x in [(12,1.0),(14,2.0),(16,2.0)]:
    S=Schwinger(N); H=S.W(x,0.0,0.0)
    ev=sla.eigsh(H,k=3,which='SA')[0]
    print(N,x,"gap*g/(2sqrt x) =",(ev[1]-ev[0])/(2*np.sqrt(x)))
