import sympy as sp
t=sp.symbols('t',real=True)
a0,a1,a2,a3=sp.symbols('a0:4',real=True); b1,b2,b3=sp.symbols('b1:4',positive=True)
# Krylov chain truncated at 4 sites is exact to O(t^6) for p_1,p_2 (p_3 ~ t^6)
L=sp.Matrix([[a0,b1,0,0],[b1,a1,b2,0],[0,b2,a2,b3],[0,0,b3,a3]])
order=7
U=sp.zeros(4,1); U[0]=1
psi=sp.zeros(4,1); term=U.copy()
for k in range(order):
    psi+=term
    term=(-sp.I*t/(k+1))*L*term
p=[sp.expand(sp.simplify(psi[n]*sp.conjugate(psi[n]))) for n in range(4)]
C=sp.series(sum(n*p[n] for n in range(4)),t,0,7).removeO()
C=sp.expand(C)
c2=sp.factor(C.coeff(t,2)); c4=sp.factor(C.coeff(t,4)); c6=sp.factor(C.coeff(t,6))
print("t^2:",c2); print("t^4:",c4); print("t^6:",c6)
