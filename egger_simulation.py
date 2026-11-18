"""
Type I error of Egger's regression test under the observed heterogeneity.

Supports Section IV-C of:
  Nguyen, T. H. & Thiombiano, D. "Innovation Capabilities, Foreign Ownership and
  Firm Size in Vietnam amid Geopolitical Uncertainty: A Pre-Specified Design and
  Power Analysis" (INCOSEF 2026).

Design
  - Data: the 54 independent samples of the benchmark (benchmark_samples.csv).
  - Effects are analysed on the Fisher-z scale: z_i = atanh(r_i), v_i = 1/(n_i - 3).
  - Simulated data have NO small-study bias:
        z_i* ~ N(mu, tau^2 + v_i),  mu = atanh(0.113)
    using the observed sample sizes and two heterogeneity estimates
    (Hunter-Schmidt tau^2 = 0.0023; REML tau^2 = 0.0147).
  - Egger's test: OLS of z_i/se_i on 1/se_i; two-sided t-test of the intercept
    with k - 2 df, alpha = 0.05.
  - 5,000 replications per scenario, seed 2026.

Expected output (reported in the paper)
  observed Egger p          ~ 0.034
  rejection rate, HS tau^2  ~ 0.57
  rejection rate, REML tau^2 ~ 0.61

Usage
  python egger_simulation.py [--data benchmark_samples.csv] [--reps 5000] [--seed 2026]
Requirements: numpy, scipy
"""
import argparse
import csv

import numpy as np
from scipy import stats

POOLED_R = 0.113
TAU2 = {"Hunter-Schmidt": 0.0023, "REML": 0.0147}


def load(path):
    r, n = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r.append(float(row["r"]))
            n.append(float(row["n"]))
    return np.array(r), np.array(n)


def egger_p(z, se):
    """Two-sided p-value of the intercept in Egger's regression."""
    y, x = z / se, 1.0 / se
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - 2
    s2 = resid @ resid / df
    var_b0 = s2 * np.linalg.inv(X.T @ X)[0, 0]
    t = beta[0] / np.sqrt(var_b0)
    return 2 * stats.t.sf(abs(t), df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="benchmark_samples.csv")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    r, n = load(a.data)
    z, v = np.arctanh(r), 1.0 / (n - 3)
    se = np.sqrt(v)
    print(f"k = {len(r)} samples, total N = {int(n.sum()):,}")
    print(f"Observed Egger test: p = {egger_p(z, se):.4f}")

    rng = np.random.default_rng(a.seed)
    mu = np.arctanh(POOLED_R)
    for name, tau2 in TAU2.items():
        rej = sum(
            egger_p(rng.normal(mu, np.sqrt(tau2 + v)), se) < a.alpha
            for _ in range(a.reps)
        ) / a.reps
        mcse = np.sqrt(rej * (1 - rej) / a.reps)
        print(f"{name:15s} tau^2 = {tau2:.4f}: rejection rate without bias = "
              f"{rej:.3f} (MC s.e. {mcse:.3f})")


if __name__ == "__main__":
    main()
