# h7_ac_bridge.py
# Python 3.11+
# pip install mpmath sympy

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Iterable, Optional

import math
from mpmath import mp

# Try different import paths for Dirichlet characters
try:
    from sympy.ntheory.residue_ntheory import DirichletGroup
    has_sympy_dirichlet = True
except ImportError:
    try:
        from sympy.ntheory import DirichletGroup
        has_sympy_dirichlet = True
    except ImportError:
        has_sympy_dirichlet = False

# High precision for safety in frequency scans (real-domain part relies on float ops)
mp.dps = 50

# =========================
# 0) Utilities & arithmetic
# =========================

def sieve_primes(N: int) -> List[int]:
    """Simple fast sieve: return primes <= N."""
    if N < 2:
        return []
    sieve = bytearray(b"\x01") * (N + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(N ** 0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            sieve[start:N + 1:p] = b"\x00" * ((N - start) // p + 1)
    return [i for i, v in enumerate(sieve) if v]


def von_mangoldt_array(N: int) -> List[mp.mpf]:
    """
    Return array L[0..N] with L[n] = Λ(n) (von Mangoldt).
    Λ(p^k) = log p, else 0. Constructed by marking prime powers.
    """
    L = [mp.mpf("0")] * (N + 1)
    for p in sieve_primes(N):
        lp = mp.log(p)
        k = p
        while k <= N:
            L[k] = lp
            k *= p
    return L


# =========================
# 1) Dirichlet characters - Fallback implementation
# =========================

if not has_sympy_dirichlet:
    # Simplified fallback - only supports principal character properly
    class SimpleDirichletCharacter:
        """Simple implementation - only principal character is correct."""
        def __init__(self, modulus: int, index: int):
            self.modulus = modulus
            self.index = index
            self.is_principal_char = (index == 0)
        def __call__(self, n: int) -> complex:
            if math.gcd(n, self.modulus) != 1:
                return 0.0 + 0.0j
            if self.is_principal_char:
                return 1.0 + 0.0j
            return 1.0 + 0.0j
        def is_primitive(self) -> bool:
            return not self.is_principal_char
        def is_principal(self) -> bool:
            return self.is_principal_char

    class SimpleDirichletGroup:
        """Simple Dirichlet group - only principal character works correctly."""
        def __init__(self, modulus: int):
            self.modulus = modulus
            self.characters = [SimpleDirichletCharacter(modulus, 0)]
        def __len__(self):
            return len(self.characters)
        def __getitem__(self, index: int):
            if index >= len(self.characters):
                return self.characters[0]
            return self.characters[index]

    DirichletGroup = SimpleDirichletGroup
    print("Using fallback Dirichlet character implementation (principal only)")


@dataclass(frozen=True)
class Character:
    modulus: int
    index: int
    primitive: bool
    _char_func: Callable[[int], complex]
    def __call__(self, n: int) -> complex:
        return self._char_func(n)


def is_principal_char(chi) -> bool:
    v = getattr(chi, "is_principal", None)
    if isinstance(v, bool):
        return v
    if callable(v):
        try:
            return bool(v())
        except Exception:
            pass
    # Fallback heuristic: test small units
    q = getattr(chi, "mod", None) or getattr(chi, "modulus", None)
    if q is None:
        return False
    for n in range(1, min(q, 100)):
        if math.gcd(n, q) == 1:
            try:
                val = chi(n)
            except Exception:
                return False
            if abs(val - 1.0) > 1e-10:
                return False
    return True


def is_primitive_char(chi) -> bool:
    v = getattr(chi, "is_primitive", None)
    if callable(v):
        try:
            return v()
        except Exception:
            pass
    if isinstance(v, bool):
        return v
    # Fallback: if not principal, treat as primitive
    return not is_principal_char(chi)


def dirichlet_characters(q: int, primitive_only: bool = True) -> List[Character]:
    """Enumerate Dirichlet characters modulo q (primitive-only by default)."""
    G = DirichletGroup(q)
    chars: List[Character] = []
    for i in range(len(G)):
        chi = G[i]
        prim = is_primitive_char(chi)
        if primitive_only and not prim:
            continue

        def make_char_func(chi_obj):
            def char_func(n: int) -> complex:
                try:
                    val = chi_obj(n)
                except Exception:
                    return 0.0 + 0.0j
                if val == 0:
                    return 0.0 + 0.0j
                try:
                    # sympy numbers
                    if hasattr(val, 'evalf'):
                        return complex(val.evalf())
                except Exception:
                    pass
                return complex(val)
            return char_func

        chars.append(Character(q, i, prim, make_char_func(chi)))
    return chars


# ====================================
# 2) H7 filter F(n) from chosen chars
# ====================================

def build_F_factory(coeffs: Dict[Tuple[int, int], complex]) -> Callable[[int], complex]:
    """
    coeffs: dict keyed by (q, chi_index) -> complex coefficient c_{q,chi}
    Returns F(n) = sum_{(q,chi)} c_{q,chi} * chi(n)
    """
    cache: Dict[int, List[Character]] = {}
    for (q, _) in coeffs.keys():
        if q not in cache:
            G = DirichletGroup(q)
            char_list: List[Character] = []
            for i in range(len(G)):
                chi = G[i]

                def make_char_func(chi_obj):
                    def char_func(n: int) -> complex:
                        try:
                            val = chi_obj(n)
                        except Exception:
                            return 0.0 + 0.0j
                        if val == 0:
                            return 0.0 + 0.0j
                        try:
                            if hasattr(val, 'evalf'):
                                return complex(val.evalf())
                        except Exception:
                            pass
                        return complex(val)
                    return char_func

                char_list.append(Character(q, i, is_primitive_char(chi), make_char_func(chi)))
            cache[q] = char_list

    def F(n: int) -> complex:
        acc = 0.0 + 0.0j
        for (q, idx), c in coeffs.items():
            chi = cache[q][idx]
            acc += c * chi(n)
        return acc

    return F


# ==========================================
# 3) Hardened Euler trace: LHS & Euler RHS
# ==========================================

def T_H7_LHS(s: complex, N: int, F: Callable[[int], complex], Lm: List[mp.mpf]) -> complex:
    """LHS: sum_{n<=N} Λ(n) F(n) / n^s"""
    total = 0.0 + 0.0j
    for n in range(2, N + 1):
        lam = Lm[n]
        if lam != 0:
            total += complex(lam) * F(n) / (n ** s)
    return total


def logderiv_L_combo_RHS(
    s: complex,
    P: int,
    coeffs: Dict[Tuple[int, int], complex],
    max_k: Optional[int] = None,
    tail_threshold: float = 1e-30,
) -> complex:
    """
    RHS: sum_{q,chi} c_{q,chi} * sum_{p<=P} sum_{k>=1} chi(p)^k * log p / p^{k s}
    Truncated Euler product expansion of -L'/L(s, chi).
    """
    total = 0.0 + 0.0j
    primes = sieve_primes(P)

    by_q: Dict[int, List[Character]] = {}
    for (q, _) in coeffs.keys():
        if q not in by_q:
            G = DirichletGroup(q)
            lst: List[Character] = []
            for i in range(len(G)):
                chi = G[i]

                def make_char_func(chi_obj):
                    def char_func(n: int) -> complex:
                        try:
                            val = chi_obj(n)
                        except Exception:
                            return 0.0 + 0.0j
                        if val == 0:
                            return 0.0 + 0.0j
                        try:
                            if hasattr(val, 'evalf'):
                                return complex(val.evalf())
                        except Exception:
                            pass
                        return complex(val)
                    return char_func

                lst.append(Character(q, i, is_primitive_char(chi), make_char_func(chi)))
            by_q[q] = lst

    for (q, idx), c in coeffs.items():
        chi = by_q[q][idx]
        subt = 0.0 + 0.0j
        for p in primes:
            lp = mp.log(p)
            chi_p = chi(p)
            if chi_p == 0:
                continue
            z = chi_p / (p ** s)
            term = z * lp
            k = 1
            while True:
                subt += term
                k += 1
                if max_k and k > max_k:
                    break
                term *= z
                if abs(term) < tail_threshold:
                    break
        total += c * subt

    return total


# =========================
# 4) Gamma scan (no plots)
# =========================

def gamma_scan(
    sigma: float,
    gammas: Iterable[float],
    N: int,
    P: int,
    F: Callable[[int], complex],
    Lm: List[mp.mpf],
    coeffs: Dict[Tuple[int, int], complex],
) -> List[Tuple[float, complex, complex]]:
    """Return list of (gamma, LHS_value, RHS_value) at s = sigma + i*gamma."""
    out: List[Tuple[float, complex, complex]] = []
    for g in gammas:
        s = sigma + 1j * g
        lhs = T_H7_LHS(s, N, F, Lm)
        rhs = logderiv_L_combo_RHS(s, P, coeffs)
        out.append((g, lhs, rhs))
    return out


# ==================================================
# 5) Reverse transmission: real-domain (explicit)
# ==================================================

def window_W(u: float) -> float:
    """Normalized log-Gaussian window: W(u) = (1/sqrt(2π)) * exp(-u^2/2)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * u * u)


def mellin_gear(s: complex, x: float, h: float) -> complex:
    """
    Mellin transform of the window:
    ϕ̂_{x,h}(s) = h * x^s * exp((h*s)^2 / 2)
    """
    # Use mp for complex exponentials safely
    return h * (mp.power(x, s)) * mp.e ** (mp.mpf("0.5") * (h * s) ** 2)


def smoothed_prime_signal(x: float, h: float, Lm: List[mp.mpf]) -> float:
    """
    P_h(x) = sum_{n} Λ(n) * W((log n - log x)/h)
    Pure smoothed von Mangoldt sum - NO n^σ factor!
    """
    if x <= 2:
        return 0.0
    lx = math.log(x)
    total = 0.0
    # Loop over prime powers only (entries with Λ(n) != 0)
    for n in range(2, len(Lm)):
        lam = Lm[n]
        if lam != 0:
            u = (math.log(n) - lx) / h
            total += float(lam) * window_W(u)
    return total


# ---- Zeros up to T (cached) ----

class ZeroCache:
    def __init__(self):
        self.zeros: List[Tuple[float, float]] = []  # list of (beta, gamma), beta≈0.5
        self.max_T: float = 0.0

    def ensure_zeros_up_to(self, T: float) -> None:
        """Fetch zeros until the last gamma exceeds T."""
        if T <= self.max_T:
            return
        k = len(self.zeros) + 1
        while True:
            z = mp.zetazero(k)  # kth nontrivial zero on 1/2 + iγ
            g = float(z.imag)
            self.zeros.append((0.5, g))
            if g > T:
                break
            k += 1
        self.max_T = T

    def get_zeros_up_to(self, T: float) -> List[Tuple[float, float]]:
        self.ensure_zeros_up_to(T)
        return [(b, g) for (b, g) in self.zeros if g <= T]


ZERO_CACHE = ZeroCache()


def zero_sum_contribution(x: float, h: float, zeros, include_trivial=True, trivial_count=50) -> float:
    if x <= 2:
        return 0.0
    lx = math.log(x)

    total_nontriv = 0.0
    for beta, gamma in zeros:  # beta=0.5 from mp.zetazero
        amp = h * (x ** beta) * math.exp(0.5 * (h * h * beta * beta)) * math.exp(-0.5 * (h * gamma) ** 2)
        phase = gamma * lx + (h * h * beta * gamma)   # = γlog x + (h^2/2)γ when beta=1/2
        total_nontriv += 2.0 * amp * math.cos(phase)

    total_triv = 0.0
    if include_trivial:
        for k in range(1, trivial_count + 1):
            total_triv += h * (x ** (-2 * k)) * math.exp(0.5 * (h * (-2 * k)) ** 2)

    return -total_nontriv + total_triv



# ---- Guard-banded x-grid with near-Nyquist sampling ----

def x_grid_guard_banded(N: int, h: float, T: float,
                        x_min: Optional[float] = None,
                        oversample: float = 4.0) -> List[float]:
    """
    Build an x-grid respecting the guard band x <= N * e^{-4h}, and sampling
    roughly oversample points per local period (Δx ≈ 2π x / T / oversample).
    """
    x_max = N * math.exp(-4.0 * h)
    if x_max <= 100.0:
        return []

    if x_min is None:
        # Start roughly two hundredth of x_max, but not below ~1000
        x_min = max(1000.0, x_max / 250.0)
    x_min = max(3.0, x_min)
    x_max = max(x_min * 1.1, x_max)  # ensure > x_min

    xs: List[float] = []
    x = x_min
    while x <= x_max:
        xs.append(float(x))
        # Local wavelength for dominant γ≈T: λ_x ≈ 2π x / T
        dx = max(1.0, (2.0 * math.pi * x) / max(1.0, T) / oversample)
        x += dx
    return xs


# ==================================
# 6) Demos (print-only, no plotting)
# ==================================

def demo_identity_and_scan():
    """
    Demo 1: Forward direction - frequency domain verification against Euler side.
    """
    print("== Demo 1: Forward trace (frequency domain) ==")
    N = 100_000
    P = 100_000
    Lm = von_mangoldt_array(N)

    # Principal character modulo 3
    q = 3
    G = DirichletGroup(q)
    trivial_idx = None
    for i in range(len(G)):
        if is_principal_char(G[i]):
            trivial_idx = i
            break
    if trivial_idx is None:
        raise RuntimeError("Could not locate principal character modulo 3")

    coeffs: Dict[Tuple[int, int], complex] = {(q, trivial_idx): 1.0 + 0.0j}
    F = build_F_factory(coeffs)

    s = 1.4 + 1j * 14.0
    lhs = T_H7_LHS(s, N, F, Lm)
    rhs = logderiv_L_combo_RHS(s, P, coeffs)
    print(f"N={N}, P={P}, s={s}")
    print(f"LHS (Λ*F/n^s)      = {lhs}")
    print(f"RHS (Euler -L'/L)  = {rhs}")
    print(f"|LHS - RHS|        = {float(abs(lhs - rhs)):.3e}")

    gammas = [k * 0.5 for k in range(0, 21)]
    scan = gamma_scan(1.4, gammas, N, P, F, Lm, coeffs)
    print("\nγ-scan (first 10 rows):")
    for g, lval, rval in scan[:10]:
        print(
            f"gamma={g:6.2f}  |LHS|={float(abs(lval)):.6e}  |RHS|={float(abs(rval)):.6e}  diff={float(abs(lval - rval)):.2e}"
        )


def corr_and_rms(v1: List[float], v2: List[float]) -> Tuple[float, float, float, float]:
    """Return (corr, rms_v1, rms_v2, ratio=v2/v1)."""
    m1 = sum(v1) / len(v1)
    m2 = sum(v2) / len(v2)
    v1c = [x - m1 for x in v1]
    v2c = [x - m2 for x in v2]
    num = sum(a * b for a, b in zip(v1c, v2c))
    d1 = math.sqrt(sum(x * x for x in v1c)) or 1.0
    d2 = math.sqrt(sum(x * x for x in v2c)) or 1.0
    corr = num / (d1 * d2)
    rms1 = math.sqrt(sum(x * x for x in v1) / len(v1))
    rms2 = math.sqrt(sum(x * x for x in v2) / len(v2))
    ratio = (rms2 / rms1) if rms1 > 0 else 0.0
    return corr, rms1, rms2, ratio


def demo_reverse_to_reals():
    """
    Demo 2: Reverse bridge (explicit formula) with T = c/h zero budget.
    No plotting; prints rows: (h, c, T, M, |X|, corr, RMS_P_h, RMS_Zero, ratio).
    """
    print("\n== Demo 2: Reverse bridge (explicit formula, T=c/h) ==")

    # Data and knobs
    N = 1_000_000
    print(f"Building von Mangoldt array up to N={N}...")
    Lm = von_mangoldt_array(N)

    h_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    c_values = [3.0, 3.5, 4.0]
    include_trivial = True
    trivial_k = 50
    oversample = 4.0

    # Precompute zeros up to max T to avoid repeated zetazero calls
    T_global = max(c_values) / min(h_values)
    ZERO_CACHE.ensure_zeros_up_to(T_global)

    print(f"Using c in {c_values} with T=c/h. Precomputed zeros up to T={T_global:.2f}.")
    print("\nOptimization sweep (per (h, c)):")
    print("h      c      T        M    |X|    Corr (15 d.p.)         RMS_P_h   RMS_Zero   Ratio")

    best = None  # track best by correlation, then ratio closeness to 1
    best_key = None

    for h in h_values:
        for c in c_values:
            T = c / h
            zeros = ZERO_CACHE.get_zeros_up_to(T)
            M = len(zeros)

            xs = x_grid_guard_banded(N, h, T, x_min=None, oversample=oversample)
            if not xs:
                print(f"{h:.3f}  {c:.1f}  {T:7.2f}    {M:4d}     0    (no x-grid after guard-band)")
                continue

            Ph_vals = []
            main_terms = []
            zero_sums = []

            for x in xs:
                ph = smoothed_prime_signal(x, h, Lm)
                Ph_vals.append(ph)
                main_terms.append(h * x * math.exp(0.5 * h * h))
                zsum = zero_sum_contribution(x, h, zeros, include_trivial=include_trivial, trivial_count=trivial_k)
                zero_sums.append(zsum)

            Ph_osc = [ph - m for ph, m in zip(Ph_vals, main_terms)]
            corr, rms_ph, rms_z, ratio = corr_and_rms(Ph_osc, zero_sums)

            print(f"{h:.3f}  {c:3.1f}  {T:7.2f}  {M:5d}  {len(xs):5d}   {corr: .15f}   {rms_ph:9.2f}  {rms_z:9.2f}  {ratio:6.3f}")

            key = (abs(1.0 - corr), abs(1.0 - ratio))
            if (best is None) or (key < best):
                best = key
                best_key = (h, c, T, M, len(xs), corr, rms_ph, rms_z, ratio, xs, Ph_osc, zero_sums)

    if best_key is None:
        print("\nNo valid runs (x-grid empty under guard-bands).")
        return

    h, c, T, M, Xlen, corr, rms_ph, rms_z, ratio, xs, Ph_osc, zero_sums = best_key
    print(f"\nBest (by corr then ratio): h={h:.3f}, c={c:.1f}, T={T:.2f}, M={M}, |X|={Xlen}")
    print(f"  Correlation:     {corr:.15f}")
    print(f"  Amplitude ratio: {ratio:.15f}")
    print(f"  P_h RMS:         {rms_ph:.15f}")
    print(f"  Zero sum RMS:    {rms_z:.15f}")

    # Show first few pointwise differences
    print("\nExplicit formula check (first 5 points):")
    print("x         P_h(x)    Main_term  P_h-Main   Zero_sum   Difference")
    for i in range(min(5, len(xs))):
        x = xs[i]
        # To reconstruct P_h(x) and main term for display:
        ph = Ph_osc[i] + h * x * math.exp(0.5 * h * h)
        main = h * x * math.exp(0.5 * h * h)
        diff = Ph_osc[i] - zero_sums[i]
        print(f"{x:8.0f}  {ph:9.3f}  {main:9.3f}  {Ph_osc[i]:9.3f}  {zero_sums[i]:9.3f}  {diff:9.3f}")


if __name__ == "__main__":
    demo_identity_and_scan()
    demo_reverse_to_reals()
