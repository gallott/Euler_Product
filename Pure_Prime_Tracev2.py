#!/usr/bin/env python3
"""
Prime-only gamma scan (legacy format)
=====================================

Generates the truncated Euler log-trace:
  T_P(γ; σ) = log | ζ_P(σ + iγ) |
            = -1/2 * Σ_{p≤P} log(1 - 2 p^{-σ} cos(γ log p) + p^{-2σ})

Outputs CSV (legacy columns):
  gamma,trace,gradient,curvature,is_predicted_valley

Defaults:
  P=5000, σ=0.5, γ∈[15,615], NPOINTS=797 (step ≈ 0.754716981132)

Usage examples:
  python build_gamma_scan_trace.py
  python build_gamma_scan_trace.py --P 8000 --sigma 0.5 --gmin 60 --gmax 4010 --npoints 50000
  python build_gamma_scan_trace.py --drop-p2 --outfile gamma_scan_trace2.csv
"""

import argparse
import math
from typing import List
import numpy as np
import pandas as pd


# ----------------- prime sieve -----------------
def sieve_primes(N: int) -> List[int]:
    """Return all primes ≤ N."""
    if N < 2:
        return []
    sieve = bytearray(b"\x01") * (N + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(N ** 0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            sieve[start:N + 1:p] = b"\x00" * (((N - start) // p) + 1)
    return [i for i, v in enumerate(sieve) if v]


def approx_sieve_limit_for_P(P: int) -> int:
    """Crude upper bound for the P-th prime: P (log P + log log P) + padding."""
    if P < 6:
        return 20
    return int(P * (math.log(P) + math.log(math.log(P)))) + 200


# --------------- core computation ---------------
def euler_logtrace(primes: List[int], gammas: np.ndarray, sigma: float = 0.5, drop_p2: bool = False) -> np.ndarray:
    """
    Compute T_P(γ;σ) = log|ζ_P(σ + iγ)| using closed-form per-prime term:
      -1/2 * log(1 - 2 r cos(γ log p) + r^2), with r = p^{-σ}
    """
    trace = np.zeros_like(gammas, dtype=float)
    for p in primes:
        if drop_p2 and p == 2:
            continue
        r = p ** (-sigma)
        lp = math.log(p)
        theta = gammas * lp
        m2 = 1.0 - 2.0 * r * np.cos(theta) + r * r  # |1 - r e^{-iθ}|^2
        # numerical floor to avoid log(0)
        np.maximum(m2, 1e-300, out=m2)
        trace += -0.5 * np.log(m2)
    return trace


def build_gamma_grid_uniform(gmin: float, gmax: float, npoints: int) -> np.ndarray:
    """Uniform gamma grid."""
    if npoints < 2:
        return np.array([gmin], dtype=float)
    return np.linspace(gmin, gmax, npoints, dtype=float)


def numerical_derivatives(values: np.ndarray, xs: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Central differences on a (nominally) uniform grid:
      gradient ≈ dT/dγ, curvature ≈ d²T/dγ²
    """
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    grad = np.gradient(values, dx)
    curv = np.gradient(grad, dx)
    # legacy edge behavior: copy neighbors to edges
    if len(values) >= 3:
        grad[0] = grad[1]
        grad[-1] = grad[-2]
        curv[0] = curv[1]
        curv[-1] = curv[-2]
    return grad, curv


# ---------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Build prime-only gamma scan CSV (legacy format).")
    ap.add_argument("--P", type=int, default=5000, help="Number of primes to include (default 5000)")
    ap.add_argument("--sigma", type=float, default=0.5, help="Sigma value (default 0.5)")
    ap.add_argument("--gmin", type=float, default=15.0, help="Gamma start (default 15.0)")
    ap.add_argument("--gmax", type=float, default=615.0, help="Gamma end (default 615.0)")
    ap.add_argument("--npoints", type=int, default=797, help="Number of gamma samples (default 797)")
    ap.add_argument("--drop-p2", action="store_true", help="Parity debias: drop p=2 term")
    ap.add_argument("--outfile", type=str, default="gamma_scan_trace1.csv",
                    help="Output CSV filename (default gamma_scan_trace1.csv)")
    args = ap.parse_args()

    print(f"== Prime-only gamma scan ==")
    print(f"P={args.P}, sigma={args.sigma}, γ∈[{args.gmin}, {args.gmax}], npoints={args.npoints}, drop_p2={args.drop_p2}")
    print(f"Writing → {args.outfile}")

    # primes
    lim = approx_sieve_limit_for_P(args.P)
    primes = sieve_primes(lim)[:args.P]
    if len(primes) < args.P:
        print(f"Warning: only found {len(primes)} primes up to {lim}. Using all available.")

    # gamma grid
    gammas = build_gamma_grid_uniform(args.gmin, args.gmax, args.npoints)

    # trace + derivatives
    trace = euler_logtrace(primes, gammas, sigma=args.sigma, drop_p2=args.drop_p2)
    grad, curv = numerical_derivatives(trace, gammas)

    # legacy CSV
    df = pd.DataFrame({
        "gamma": gammas,
        "trace": trace,
        "gradient": grad,
        "curvature": curv,
        "is_predicted_valley": np.zeros_like(trace, dtype=int),
    })
    df.to_csv(args.outfile, index=False)
    print(f"✅ Wrote {len(df)} rows to {args.outfile}")


if __name__ == "__main__":
    main()
