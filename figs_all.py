import numpy as np
from fig_common import *
from scipy.linalg import eigh_tridiagonal
# ---------- Fig 1: C(t) and b_n, N=12
d=np.load("../data/scan_full12.npz"); ts=d["ts"]; l=d["l"]
sel=[(0.30,S1,"-",r"$\ell=0.30$ (confined)"),(0.50,S2,"-",r"$\ell=0.50$ ($\theta=\pi$)"),(0.64,S3,"-",r"$\ell=0.64\simeq\ell_3$"),(0.90,S4,"-",r"$\ell=0.90=\ell_1$")]
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for lv,c,ls,lab in sel:
    k=np.argmin(abs(l-lv)); C=d["C"][k]; m=ts>0
    ax[0].loglog(ts[m],C[m],ls,color=c,lw=1.1,label=lab)
    n=d["depth"][k]; b=d["b"][k][:n-1]
    ax[1].plot(np.arange(1,len(b)+1)[:494],b[:494],color=c,lw=0.9)
tt=np.logspace(-1,0.3,20); ax[0].loglog(tt,0.11*tt**2,":",color="#52514e",lw=1); ax[0].text(0.35,0.004,r"$b_1^2t^2$",fontsize=8,color="#52514e")
ax[0].set_xlim(0.1,4000); ax[0].set_ylim(1e-3,100)
ax[0].set_xlabel(r"$t\;[2/(ag^2)]$"); ax[0].set_ylabel(r"$C(t)$"); ax[0].legend(frameon=False,loc="upper left",fontsize=7)
ax[1].set_xlabel(r"$n$"); ax[1].set_ylabel(r"$b_n$"); ax[1].set_xlim(0,494)
for a,t in zip(ax,"ab"): a.grid(color=GRID,lw=0.5); a.text(0.97,0.93,f"({t})",transform=a.transAxes,ha="right",fontsize=9)
plt.tight_layout(); plt.savefig("../figs/fig_Ct_bn.pdf"); plt.savefig("../figs/fig_Ct_bn.png",dpi=150)
# ---------- Fig 2: precision
p=np.load("../data/precision2.npz"); p1=np.load("../data/precision.npz")
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
ax[0].plot(p["ts"],p["Ch"],color=S1,label="multiprecision (exact)")
ax[0].plot(p["ts"],p["Cd"],"--",color=S2,label="double, full reorth.")
ax[0].plot(p["ts"],p["Cm"],":",color=S3,lw=1.6,label=r"measure cut $10^{-16}$")
ax[0].set_xlim(0,300); ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"$C(t)$"); ax[0].legend(frameon=False,fontsize=7,loc="lower right")
for (b,bd,c,lab) in [(p["b"],p["bd"],S1,r"$x=1/2,\ \mu=4/5,\ \ell=3/10$"),(p1["1/2_b"],p1["1/2_bd"],S2,r"$x=1/10,\ \mu=2/5,\ \ell=1/2$")]:
    m=min(len(b),len(bd)); dev=np.abs(b[:m]-bd[:m]); dev[dev==0]=1e-17
    ax[1].semilogy(np.arange(1,m+1),dev,color=c,lw=0.9,label=lab)
ax[1].set_xlabel(r"$n$"); ax[1].set_ylabel(r"$|b_n^{\rm double}-b_n^{\rm exact}|$"); ax[1].set_ylim(1e-17,1e2); ax[1].legend(frameon=False,fontsize=7,loc="lower right")
for a,t in zip(ax,"ab"): a.grid(color=GRID,lw=0.5); a.text(0.03,0.93,f"({t})",transform=a.transAxes,fontsize=9)
plt.tight_layout(); plt.savefig("../figs/fig_precision.pdf"); plt.savefig("../figs/fig_precision.png",dpi=150)
# ---------- Fig 3: theta scan
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for tag,N,c,ls in [("full8",8,S1,"-"),("full10",10,S2,"--"),("full12",12,S3,"-.")]:
    d=np.load(f"../data/scan_{tag}.npz"); l=d["l"]; ts=d["ts"]
    ax[0].semilogy(l,d["C"][:,ts<=200].mean(1),ls,color=c,label=f"$N={N}$")
    ax[1].semilogy(l,d["Cinf"],ls,color=c,label=f"$N={N}$")
for a in ax:
    a.axvline(0.5,color="#8a8984",lw=0.8,ls=":")
    for xv in [0.9,0.5+0.4/3]: a.axvline(xv,color="#c3c2b7",lw=0.7,ls="--")
    a.set_xlabel(r"$\ell=\theta/2\pi$"); a.grid(color=GRID,lw=0.5)
ax[0].set_ylabel(r"$\overline{C}_{[0,200]}$"); ax[1].set_ylabel(r"$\overline{C}_\infty$")
ax[1].legend(frameon=False,loc="upper left",bbox_to_anchor=(0.03,0.88))
for a,t in zip(ax,"ab"):
    a.text(0.03,0.93,f"({t})",transform=a.transAxes,fontsize=9)
    yl=a.get_ylim(); a.text(0.5,yl[1]*0.55,r"$\theta=\pi$",ha="center",fontsize=7,color="#52514e",bbox=dict(fc="white",ec="none",pad=0.5))
    a.text(0.9,yl[0]*1.6,r"$\ell_1$",ha="center",fontsize=8,color="#52514e"); a.text(0.6333,yl[0]*1.6,r"$\ell_3$",ha="center",fontsize=8,color="#52514e")
plt.tight_layout(); plt.savefig("../figs/fig_theta_scan.pdf"); plt.savefig("../figs/fig_theta_scan.png",dpi=150)
# ---------- Fig 4: theta=pi peak collapse + N scaling
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for tag,N,x,c,ls in [("pi8",8,0.1,S1,"-"),("pi10",10,0.1,S2,"-"),("pi12",12,0.1,S3,"-"),("pi10x05",10,0.05,S4,"--")]:
    d=np.load(f"../data/fine_{tag}.npz"); J=x*x/0.8; y=d["Cinf"]; base=np.interp(-2.0,(d["l"]-0.5)/J,y)
    ax[0].plot((d["l"]-0.5)/J,y/base,ls,color=c,label=f"$N={N},\ x={x}$")
ax[0].set_xlim(-2.0,2.4); ax[0].set_xlabel(r"$(\ell-1/2)/J_{\rm eff}$"); ax[0].set_ylabel(r"$\overline{C}_\infty(\ell)\,/\,\overline{C}_\infty(\frac{1}{2}-2J_{\rm eff})$")
ax[0].legend(frameon=False,fontsize=7,loc="upper right")
Ns=np.array([8,10,12,14])
def win(tag,lv):
    d=np.load(f"../data/{tag}.npz"); k=np.argmin(abs(d["l"]-lv)); return d["C"][k][d["ts"]<=200].mean()
v05=[win("scan_full8",.5),win("scan_full10",.5),win("scan_full12",.5),win("pts_full14",.5)]
v45=[win("scan_full8",.45),win("scan_full10",.45),win("scan_full12",.45),win("pts_full14",.45)]
v30=[win("scan_full8",.3),win("scan_full10",.3),win("scan_full12",.3)]
ax[1].plot(Ns,np.array(v05)/(Ns-1),"o-",color=S2,ms=4,label=r"$\ell=0.50$")
ax[1].plot(Ns,np.array(v45)/(Ns-1),"s-",color=S1,ms=4,label=r"$\ell=0.45$")
ax[1].plot(Ns[:3],np.array(v30)/(Ns[:3]-1),"^-",color=S3,ms=4,label=r"$\ell=0.30$")
ax[1].set_xlabel(r"$N$"); ax[1].set_ylabel(r"$\overline{C}_{[0,200]}/(N-1)$"); ax[1].legend(frameon=False,fontsize=7,loc="upper left"); ax[1].set_xticks(Ns)
ax[0].text(0.03,0.93,"(a)",transform=ax[0].transAxes,fontsize=9); ax[1].text(0.95,0.45,"(b)",transform=ax[1].transAxes,fontsize=9,ha="right")
for a in ax: a.grid(color=GRID,lw=0.5)
plt.tight_layout(); plt.savefig("../figs/fig_pi.pdf"); plt.savefig("../figs/fig_pi.png",dpi=150)
print("v05",v05,"\nv45",v45,"\nv30",v30)
