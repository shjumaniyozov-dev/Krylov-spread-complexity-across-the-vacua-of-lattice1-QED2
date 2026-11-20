"""
Lattice Schwinger model (Kogut-Susskind staggered fermions, open boundaries,
gauge field eliminated by Gauss's law) and Krylov spread complexity.

Hamiltonian (dimensionless, W = 2H/(a g^2)), sites n = 0..N-1, links n = 0..N-2
[Banuls, Cichy, Jansen, Cirac, JHEP 11 (2013) 158, Eq. (2.4)]:

  W = x  sum_n (s+_n s-_{n+1} + h.c.)
    + mu/2 sum_n (1 + (-1)^n sz_n)
    + sum_n ( l + L_n )^2 ,     L_n = 1/2 sum_{k<=n} ( (-1)^k + sz_k )

x = 1/(a g)^2 ,  mu = 2 sqrt(x) m/g ,  l = theta/(2 pi).
Physical energies: E = W * g / (2 sqrt(x)).

Bit convention: bit k = 1  <->  sz_k = +1 (site occupied).
Bare (strong-coupling) vacuum: even sites empty, odd sites occupied.
"""
import numpy as np
import scipy.sparse as sp
from itertools import combinations
from scipy.linalg import eigh, hessenberg


def basis_zero_charge(N):
    """All N-bit configurations with N/2 occupied sites (total charge Q=0)."""
    states = []
    for occ in combinations(range(N), N // 2):
        s = 0
        for k in occ:
            s |= 1 << k
        states.append(s)
    states = np.array(sorted(states), dtype=np.int64)
    index = {int(s): i for i, s in enumerate(states)}
    return states, index


def sz_array(states, N):
    return np.array([[1 if (s >> k) & 1 else -1 for k in range(N)] for s in states], dtype=np.int8)


def diagonal_parts(states, N):
    """Return (mass_count, Lmat) with mass term = mu * mass_count and
    electric = sum_n (l + L_n)^2 ; Lmat[i, n] = L_n for configuration i."""
    sz = sz_array(states, N).astype(np.float64)
    sgn = np.array([(-1) ** k for k in range(N)], dtype=np.float64)
    mass_count = 0.5 * np.sum(1.0 + sgn[None, :] * sz, axis=1)  # number of particles+antiparticles
    Lmat = 0.5 * np.cumsum(sgn[None, :] + sz, axis=1)[:, : N - 1]
    return mass_count, Lmat


def hopping_matrix(states, index, N):
    rows, cols = [], []
    for i, s in enumerate(states):
        s = int(s)
        for n in range(N - 1):
            b1 = (s >> n) & 1
            b2 = (s >> (n + 1)) & 1
            if b1 != b2:
                t = s ^ ((1 << n) | (1 << (n + 1)))
                rows.append(i)
                cols.append(index[t])
    data = np.ones(len(rows))
    D = len(states)
    return sp.csr_matrix((data, (rows, cols)), shape=(D, D))


class Schwinger:
    def __init__(self, N):
        assert N % 2 == 0
        self.N = N
        self.states, self.index = basis_zero_charge(N)
        self.D = len(self.states)
        self.mass_count, self.Lmat = diagonal_parts(self.states, N)
        self.Hop = hopping_matrix(self.states, self.index, N)
        vac = 0
        for k in range(1, N, 2):
            vac |= 1 << k
        self.vac = self.index[vac]

    def diag(self, mu, l):
        return mu * self.mass_count + np.sum((l + self.Lmat) ** 2, axis=1)

    def W(self, x, mu, l):
        return x * self.Hop + sp.diags(self.diag(mu, l))

    def state(self, config_bits):
        v = np.zeros(self.D)
        v[self.index[int(config_bits)]] = 1.0
        return v

    def vacuum(self):
        v = np.zeros(self.D)
        v[self.vac] = 1.0
        return v


# ---------------------------------------------------------------- Krylov tools

def spectral_measure(H_dense, psi0, tol=1e-15, degtol=1e-11):
    """Eigen-decompose H and return distinct energies with weights |<E|psi0>|^2,
    merging (numerically) degenerate levels."""
    E, V = eigh(H_dense)
    w = (V.T @ psi0) ** 2
    # merge degenerate levels
    Em, wm = [], []
    i = 0
    while i < len(E):
        j = i + 1
        acc = w[i]
        while j < len(E) and abs(E[j] - E[i]) < degtol:
            acc += w[j]
            j += 1
        Em.append(E[i]); wm.append(acc)
        i = j
    Em = np.array(Em); wm = np.array(wm)
    keep = wm > tol
    wm = wm[keep] / wm[keep].sum()
    return Em[keep], wm


def krylov_from_measure(E, w):
    """Householder tridiagonalisation of diag(E) seeded with sqrt(w).
    Returns a (D), b (D-1) and the Krylov vectors K (columns) in the eigenbasis.
    Exact up to floating point; equivalent to Lanczos with full reorthogonalisation."""
    D = len(E)
    v = np.sqrt(w)
    # Householder reflector P with P e1 = v
    e1 = np.zeros(D); e1[0] = 1.0
    u = v - e1
    nu = np.linalg.norm(u)
    if nu < 1e-15:
        P = np.eye(D)
    else:
        u /= nu
        P = np.eye(D) - 2.0 * np.outer(u, u)
    A = (P * E[None, :]) @ P  # P diag(E) P
    A = 0.5 * (A + A.T)
    T, Q = hessenberg(A, calc_q=True)
    K = P @ Q  # columns: Krylov vectors expressed in the eigenbasis
    a = np.diag(T).copy()
    b = np.abs(np.diag(T, -1)).copy()
    return a, b, K


def spread_complexity(E, w, K, times):
    """C(t) = sum_n n |<K_n|psi(t)>|^2 , psi(t) = sum_i e^{-i E_i t} sqrt(w_i) |E_i>."""
    v = np.sqrt(w)
    n = np.arange(K.shape[1])
    C = np.empty(len(times))
    for it, t in enumerate(times):
        amp = K.T @ (np.exp(-1j * E * t) * v)
        p = np.abs(amp) ** 2
        C[it] = np.dot(n, p)
    return C


def spread_complexity_timeavg(w, K):
    """Infinite-time average for a non-degenerate (merged) spectrum:
    Cbar = sum_n n sum_i |K_{in}|^2 w_i."""
    n = np.arange(K.shape[1])
    pbar = (K ** 2).T @ w
    return float(np.dot(n, pbar)), pbar


def lanczos_sparse(H, psi0, nmax, reorth=True):
    """Direct Lanczos with full (twice) reorthogonalisation on a sparse H, for
    cross-checking the Householder route and for larger N (truncated depth)."""
    D = H.shape[0]
    Q = np.zeros((D, nmax + 1))
    a = np.zeros(nmax + 1); b = np.zeros(nmax + 1)
    q = psi0 / np.linalg.norm(psi0)
    Q[:, 0] = q
    for n in range(nmax + 1):
        r = H @ Q[:, n]
        a[n] = np.dot(Q[:, n], r)
        if n == nmax:
            break
        r = r - a[n] * Q[:, n] - (b[n - 1] * Q[:, n - 1] if n > 0 else 0)
        if reorth:
            for _ in range(2):
                r -= Q[:, : n + 1] @ (Q[:, : n + 1].T @ r)
        b[n] = np.linalg.norm(r)
        if b[n] < 1e-12:
            return a[: n + 1], b[:n], Q[:, : n + 1]
        Q[:, n + 1] = r / b[n]
    return a, b[:nmax], Q


def chain_evolve(a, b, times):
    """Evolve on the (truncated) Krylov chain, psi(0) = K_0; returns p_n(t)."""
    T = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    ev, U = eigh(T)
    c0 = U[0, :]
    P = np.empty((len(times), len(a)))
    for it, t in enumerate(times):
        amp = U @ (np.exp(-1j * ev * t) * c0)
        P[it] = np.abs(amp) ** 2
    return P


class SchwingerQLM(Schwinger):
    """Spin-1/2 quantum-link truncation: keep only Gauss-law configurations with
    L_n in {0,-1} on every link, i.e. E_n = L_n + 1/2 in {+1/2,-1/2}.
    Then (l+L_n)^2 = 1/4 + 2(l-1/2) E_n + (l-1/2)^2 : the theta-term becomes a
    linear 'topological' term with coefficient 2(l-1/2)."""
    def __init__(self, N):
        assert N % 2 == 0
        self.N = N
        st, _ = basis_zero_charge(N)
        mc, Lm = diagonal_parts(st, N)
        keep = np.all((np.abs(Lm) < 1e-9) | (np.abs(Lm + 1) < 1e-9), axis=1)
        self.states = st[keep]
        self.index = {int(s): i for i, s in enumerate(self.states)}
        self.D = len(self.states)
        self.mass_count, self.Lmat = mc[keep], Lm[keep]
        rows, cols = [], []
        for i, s in enumerate(self.states):
            s = int(s)
            for n in range(N - 1):
                if ((s >> n) & 1) != ((s >> (n + 1)) & 1):
                    t = s ^ ((1 << n) | (1 << (n + 1)))
                    if t in self.index:
                        rows.append(i); cols.append(self.index[t])
        self.Hop = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.D, self.D))
        vac = 0
        for k in range(1, N, 2):
            vac |= 1 << k
        self.vac = self.index[vac]
