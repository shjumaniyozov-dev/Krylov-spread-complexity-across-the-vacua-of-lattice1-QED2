"""Independent re-verification of every quantitative claim in the manuscript."""
import numpy as np, scipy.sparse as sp
from functools import reduce
from math import comb
from scipy.linalg import eigh_tridiagonal
from schwinger import *
from krylov_run import coeffs, complexity, model
R=[]
def check(name,ok,info=""):
    R.append(ok); print(("PASS " if ok else "FAIL ")+name+("  ["+info+"]" if info else ""),flush=True)
# 1 Hamiltonian vs brute-force Pauli construction
N=6;x,mu,l=0.7,0.3,0.37
I2=np.eye(2);sz=np.diag([1.,-1.]);spl=np.array([[0,1.],[0,0]]);smi=spl.T
op=lambda o,k: reduce(np.kron,[o if j==k else I2 for j in range(N)])
H=sum(x*(op(spl,n)@op(smi,n+1)+op(smi,n)@op(spl,n+1)) for n in range(N-1))+sum(mu/2*(np.eye(64)+(-1)**n*op(sz,n)) for n in range(N))
for n in range(N-1):
    L=l*np.eye(64)+0.5*sum((-1)**k*np.eye(64)+op(sz,k) for k in range(n+1)); H+=L@L
Q=sum(op(sz,k) for k in range(N)); idx=[i for i in range(64) if abs(Q[i,i])<1e-12]
d=np.max(abs(np.linalg.eigvalsh(H[np.ix_(idx,idx)])-np.linalg.eigvalsh(Schwinger(N).W(x,mu,l).toarray())))
check("Eq.(1) Hamiltonian equals Pauli construction (N=6)",d<1e-12,f"{d:.1e}")
# 2 reflection symmetry R for generic l
for N in [8,10]:
    S=Schwinger(N); P=np.zeros((S.D,S.D))
    for i,s in enumerate(S.states):
        t=0
        for k in range(N):
            if not (int(s)>>(N-1-k))&1: t|=1<<k   # sigma_k' = -sigma_{N-1-k}
        P[S.index[t],i]=1
    Hd=S.W(0.37,0.61,0.2718).toarray(); c=np.max(abs(P@Hd-Hd@P))
    check(f"reflection R commutes with W at generic l (N={N})",c<1e-12,f"{c:.1e}")
    v=S.vacuum(); check(f"R|Omega>=|Omega> (N={N})",np.allclose(P@v,v))
# 3 closed forms
mx=0
for N in [8,10,12]:
    for (xs,ms,ls) in [('1/10','2/5','3/10'),('1/4','1','1/2'),('7/10','3/10','4/5')]:
        a,b=coeffs("full",N,xs,ms,ls,4,bits=300); xf,mf,lf=[eval(z) for z in (xs,ms,ls)]
        Np,Nm=N//2,N//2-1
        cl=[(N-1)*lf**2, xf**2*(N-1), (N-1)*lf**2+2*mf+1+2*lf/(N-1), 16*lf**2*Np*Nm/(N-1)**2+2*xf**2*(N-2)*(N-3)/(N-1)]
        mx=max(mx,np.max(abs(np.array([a[0],b[0]**2,a[1],b[1]**2])-cl)))
check("closed forms a0,b1^2,a1,b2^2 (Eqs.11-12)",mx<1e-12,f"max dev {mx:.1e}")
# 4 short-time expansion
N=12; a,b=coeffs("full",N,'1/10','2/5','1/2',30,bits=600)
b1s=0.01*11; Db=0.8+1+1/11; b2s=16*0.25*6*5/121+2*0.01*10*9/11; k4=-b1s*(Db**2+4*b1s-2*b2s)/12
ts=np.array([0.05,0.1,0.2]); C=complexity(a,b,ts)[0]; rel=abs(C-(b1s*ts**2+k4*ts**4))/C
check("short-time Eq.(13): relative error ~ t^4",np.allclose(rel[1:]/rel[:-1],16,rtol=0.15),str(np.round(rel,9)))
# 5 Krylov dimension and precision convergence
for N in [8,10,12]:
    S,_=model("full",N); a,b=coeffs("full",N,'1/10','2/5','3/10',S.D,bits=6*S.D+1000)
    DK=(comb(N,N//2)+2**(N//2))//2
    check(f"Krylov dimension = D_K (N={N})",len(a)==DK,f"{len(a)} vs {DK}")
S,_=model("full",10); ts=np.linspace(0,4000,801)
C1=complexity(*coeffs("full",10,'1/10','2/5','1/2',S.D,bits=int(2.6*S.D)+600),ts)[0]
C2=complexity(*coeffs("full",10,'1/10','2/5','1/2',S.D,bits=6*S.D+1000),ts)[0]
check("precision convergence of C(t) (N=10, l=1/2)",np.max(abs(C1-C2))<1e-12,f"{np.max(abs(C1-C2)):.1e}")
# 6 resonance shifts
import subprocess; out=subprocess.run(["python3","lamb.py","12"],capture_output=True,text=True).stdout
check("second-order shifted l3=0.6398, l5=0.5836",("0.6398" in out) and ("0.5836" in out))
d3=np.load("../data/fine_res3_12.npz"); p3=d3["l"][np.argmax(d3["Cinf"])]
check("n=3 peak (N=12) within 1e-3 of 0.6398",abs(p3-0.6398)<1e-3,f"{p3:.4f}")
d5=np.load("../data/fine_res5_10.npz"); p5=d5["l"][np.argmax(d5["Cinf"])]
check("n=5 peak (N=10) within 2e-4 of 0.5836",abs(p5-0.5836)<2e-4,f"{p5:.5f}")
# 7 n=1 ladder
for tag in ["scan_full10","scan_full12"]:
    s=np.load(f"../data/{tag}.npz"); k=np.argmin(abs(s["l"]-0.9)); n=s["depth"][k]
    bb=s["b"][k][:n-1]; bb=np.where(bb>0,bb,1e-300); ev,U=eigh_tridiagonal(s["a"][k][:n],bb); w=U[0]**2
    top=np.sort(ev[np.argsort(-w)[:3]]); sp_=np.diff(top)
    check(f"Rabi ladder spacing ~2x ({tag})",np.allclose(sp_,0.2,atol=0.002),str(np.round(sp_,4)))
# 8 theta=pi width and N scaling
for tag,x in [("pi8",0.1),("pi10",0.1),("pi12",0.1),("pi10x05",0.05)]:
    f=np.load(f"../data/fine_{tag}.npz"); y=f["Cinf"]; base=min(y[0],y[-1]); a_=f["l"][y>=base+0.5*(y.max()-base)]
    r=(a_.max()-a_.min())/(x*x/0.8); check(f"FWHM/Jeff in [0.7,1.4] ({tag})",0.7<=r<=1.4,f"{r:.2f}")
def win(tag,lv):
    q=np.load(f"../data/{tag}.npz"); k=np.argmin(abs(q["l"]-lv)); return q["C"][k][q["ts"]<=200].mean()
r05=[win(t,0.5)/(N-1) for t,N in [("scan_full8",8),("scan_full10",10),("scan_full12",12),("pts_full14",14)]]
r45=[win(t,0.45)/(N-1) for t,N in [("scan_full8",8),("scan_full10",10),("scan_full12",12),("pts_full14",14)]]
check("per-link complexity at l=1/2 quoted (0.080,0.136,0.209,0.276)",np.allclose(r05,[0.080,0.136,0.209,0.276],atol=0.001),str(np.round(r05,4)))
check("per-link complexity at l=0.45 quoted (0.022,0.025,0.027,0.030)",np.allclose(r45,[0.022,0.025,0.027,0.030],atol=0.001),str(np.round(r45,4)))
s=np.load("../data/scan_full12.npz"); w=s["C"][:,s["ts"]<=200].mean(1)
vals=[w[np.argmin(abs(s["l"]-v))] for v in (0.45,0.5,0.55)]
check("N=12 window values 0.30/2.29/0.47",np.allclose(vals,[0.30,2.29,0.47],atol=0.006),str(np.round(vals,3)))
print(f"\n{sum(R)}/{len(R)} checks passed")
