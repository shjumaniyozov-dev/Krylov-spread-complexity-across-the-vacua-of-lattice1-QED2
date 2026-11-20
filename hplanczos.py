"""
High-precision Lanczos for the lattice Schwinger model (gmpy2 mpfr in numpy
object arrays).  The Lanczos map (H, psi0) -> (a_n, b_n) is ill-conditioned
for this model because the electric energy gives a broad spectrum whose
exponentially small spectral weights dominate high moments; double precision
is therefore not sufficient.  Inputs x, mu, l are exact rationals.
"""
import numpy as np
import gmpy2
from gmpy2 import mpfr, mpq


def partners(S):
    """For each link n an index array p_n with p_n[i] = index of the state
    reached by hopping across link n, or -1 if not allowed."""
    P = []
    for n in range(S.N - 1):
        p = -np.ones(S.D, dtype=np.int64)
        for i, s in enumerate(S.states):
            s = int(s)
            if ((s >> n) & 1) != ((s >> (n + 1)) & 1):
                p[i] = S.index[s ^ ((1 << n) | (1 << (n + 1)))]
        P.append(p)
    return P


def hp_lanczos(S, x, mu, l, nmax, bits=400, psi0_index=None, check_every=50, P=None):
    """x, mu, l : python Fractions or strings convertible by mpq.
    Returns (a, b, info) with a, b as numpy float64 arrays and the mpfr lists."""
    ctx = gmpy2.get_context()
    ctx.precision = bits
    xq, muq, lq = mpq(x), mpq(mu), mpq(l)
    # exact diagonal: mu*mass_count + sum (l + L_n)^2 ; mass_count, L_n are half-integers/integers
    mc = S.mass_count
    Lm = S.Lmat
    d = np.empty(S.D, dtype=object)
    for i in range(S.D):
        acc = muq * mpq(int(round(mc[i])))
        for Ln in Lm[i]:
            v = lq + mpq(int(round(2 * Ln)), 2)
            acc += v * v
        d[i] = mpfr(acc)
    if P is None:
        P = partners(S)
    masks = [p >= 0 for p in P]
    xf = mpfr(xq)
    zero = mpfr(0)

    def matvec(v):
        y = d * v
        for p, m in zip(P, masks):
            y[m] += xf * v[p[m]]
        return y

    def dot(u, v):
        return gmpy2.fsum(u * v) if hasattr(gmpy2, "fsum") else sum(u * v)

    q_prev = np.array([zero] * S.D, dtype=object)
    q = np.array([zero] * S.D, dtype=object)
    i0 = S.vac if psi0_index is None else psi0_index
    q[i0] = mpfr(1)
    q0 = q.copy()
    a_list, b_list = [], []
    b_prev = zero
    orth = []
    for n in range(nmax + 1):
        r = matvec(q)
        an = dot(q, r)
        a_list.append(an)
        if n == nmax:
            break
        r = r - an * q - b_prev * q_prev
        bn = gmpy2.sqrt(dot(r, r))
        if bn < mpfr(10) ** (-(bits // 5)):
            break
        b_list.append(bn)
        q_prev, q = q, r / bn
        b_prev = bn
        if check_every and (n + 1) % check_every == 0:
            orth.append((n + 1, float(abs(dot(q, q0)))))
    a = np.array([float(v) for v in a_list])
    b = np.array([float(v) for v in b_list])
    return a, b, {"orth": orth, "a_mp": a_list, "b_mp": b_list}
