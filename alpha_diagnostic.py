"""
alpha_diagnostic.py
===================
Sweep the alpha-fair participation policy WITHOUT running any FL training, so you
can see how the per-client probabilities change with alpha and pick a small,
non-redundant set of alpha values for your plots and simulations.

It mirrors `compute_pi` and the R_u / P_on formulas in
`FL_setting_NeurIPS_batuFlavor.py`, so the policy here is identical to the one
the simulator would use (the simulator's `--temp` argument IS alpha).

Outputs
-------
1. A printed table: budget usage, #p=1, #fractional, throughput, fairness.
2. A printed recommendation: distinct alpha values spanning the
   efficiency<->equal-impact axis (ready to paste as `--temp` values).
3. A figure `alpha_diagnostic.pdf` with four panels.

Usage
-----
    python3 alpha_diagnostic.py --num_users 100 --user_prob_disc 0.45 --bufferLimit 10
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Policy primitives (copied verbatim from FL_setting_NeurIPS_batuFlavor.py)
# --------------------------------------------------------------------------
def compute_R_pon(keepProbAvail, keepProbNotAvail):
    """Renewal-reward R_u and stationary availability P_on (FEDFresh Thm 0.1)."""
    N = len(keepProbAvail)
    r = np.zeros(N)
    pon = np.zeros(N)
    for i in range(N):
        P10 = 1.0 - keepProbAvail[i]      # avail -> unavail
        P01 = 1.0 - keepProbNotAvail[i]   # unavail -> avail
        with np.errstate(all="ignore"):
            term1 = (1.0 - P10)
            term2 = (P10 * P01) / (1.0 - P01)
            term3 = (P10 * P01) / ((1.0 - P01) ** 2) * np.log(P01)
            r[i] = (term1 - term2 - term3) / (1.0 + P10 / P01)
            pon[i] = P01 / (P01 + P10)
    return r, pon


def compute_pi(r, pon, alpha, K):
    """alpha-fair waterfilling policy: max Sum_u U_alpha(p_u * R_u) s.t. budget."""
    N = len(r)
    valid = (pon > 1e-12) & (r > 1e-12)
    if np.dot(pon, np.ones(N)) <= K:
        return np.ones(N)

    if alpha == 0:
        ratio = np.where(valid, r / pon, -np.inf)
        order = np.argsort(-ratio)
        p = np.zeros(N)
        budget = K
        for u in order:
            if not valid[u] or budget <= 0:
                break
            alloc = min(1.0, budget / pon[u])
            p[u] = alloc
            budget -= pon[u] * alloc
        return p

    if np.isinf(alpha):
        def load_mm(c):
            p = np.where(valid, np.minimum(1.0, c / r), 0.0)
            return np.dot(pon, p)
        r_max = r[valid].max() if valid.any() else 1.0
        lo, hi = 0.0, r_max * 1e9
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if load_mm(mid) < K:
                lo = mid
            else:
                hi = mid
        c_star = (lo + hi) / 2.0
        return np.where(valid, np.minimum(1.0, c_star / r), 0.0)

    log_phi = np.where(valid, (1.0 - alpha) * np.log(r) - np.log(pon), -np.inf)

    def load(log_nu):
        lp = (log_phi - log_nu) / alpha
        with np.errstate(over="ignore"):
            p = np.where(valid, np.where(lp >= 0.0, 1.0, np.exp(lp)), 0.0)
        return np.dot(pon, p)

    lpv = log_phi[valid]
    lo, hi = lpv.min() - 100.0, lpv.max() + 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if load(mid) > K:
            lo = mid
        else:
            hi = mid
    star = (lo + hi) / 2.0
    lp_star = np.where(valid, (log_phi - star) / alpha, -np.inf)
    with np.errstate(over="ignore"):
        return np.where(valid, np.where(lp_star >= 0.0, 1.0, np.exp(lp_star)), 0.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def jain(x):
    """Jain's fairness index over ALL clients (starved clients counted as 0).
    1 = perfectly equal contribution; ~1/N = one client takes everything."""
    n = len(x)
    denom = n * float(np.dot(x, x))
    if denom <= 0:
        return 1.0
    return float((x.sum() ** 2) / denom)


def summarise(p, r, pon, K):
    x = p * r                                  # per-client contribution
    nz = x[x > 1e-12]
    return dict(
        budget=float(np.dot(pon, p)),
        n_full=int(np.sum(p > 0.999)),
        n_frac=int(np.sum((p > 1e-9) & (p <= 0.999))),
        n_zero=int(np.sum(p <= 1e-9)),         # starved clients
        throughput=float(x.sum()),             # C = Sum p_u R_u
        jain=jain(x),
        ratio=float(nz.max() / nz.min()) if len(nz) else 1.0,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_users", type=int, default=100)
    ap.add_argument("--user_prob_disc", type=float, default=0.45)
    ap.add_argument("--bufferLimit", type=int, default=10, help="budget K")
    ap.add_argument("--out", type=str, default="alpha_diagnostic.pdf")
    args = ap.parse_args()

    N = args.num_users
    disc = args.user_prob_disc
    K = float(args.bufferLimit)

    # keepProb config — identical to lr_001_..._batuFlavor.py
    kpa = np.concatenate([np.full(N // 2, 0.5 - disc),
                          np.full(N - N // 2, 0.5 + disc)])
    kpna = np.concatenate([np.full(N // 2, 0.5 + disc),
                           np.full(N - N // 2, 0.5 - disc)])
    r, pon = compute_R_pon(kpa, kpna)

    print(f"\nConfig: num_users={N}, user_prob_disc={disc}, K={K:g}")
    if not np.all(np.isfinite(r)):
        print("  !! WARNING: R_u has non-finite values (P01 in {0,1}). "
              "Use 0 < user_prob_disc < 0.5.")
        return
    print(f"  R_u   unique : {np.unique(np.round(r, 6))}")
    print(f"  P_on  unique : {np.unique(np.round(pon, 6))}")
    print(f"  gamma unique : {np.unique(np.round(np.where(pon>0, r/pon, 0), 6))}")
    if np.dot(pon, np.ones(N)) <= K:
        print("  Note: total availability <= K, every client gets p=1 for all "
              "alpha (alpha has no effect). Increase users or lower K.")

    # ---- candidate alpha grid -------------------------------------------------
    candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
                  10.0, np.inf]
    policies = {a: compute_pi(r, pon, a, K) for a in candidates}

    print(f"\n{'alpha':>8} {'budget':>9} {'#p=1':>6} {'#frac':>7} {'#p=0':>6} "
          f"{'C=SumpR':>10} {'Jain':>8} {'max/min':>10}")
    for a in candidates:
        s = summarise(policies[a], r, pon, K)
        astr = "inf" if np.isinf(a) else f"{a:g}"
        print(f"{astr:>8} {s['budget']:>9.4f} {s['n_full']:>6} "
              f"{s['n_frac']:>7} {s['n_zero']:>6} {s['throughput']:>10.4f} "
              f"{s['jain']:>8.4f} {s['ratio']:>10.2f}")

    # ---- de-duplicate: keep alpha values whose policy is visibly distinct -----
    distinct = []
    for a in candidates:
        p = policies[a]
        if all(np.max(np.abs(p - policies[b])) > 1e-4 for b in distinct):
            distinct.append(a)

    # if the last (saturated) regime equals the alpha=inf policy, label it inf
    # so the recommendation uses the meaningful 'equal-impact' anchor
    if not any(np.isinf(a) for a in distinct):
        if np.max(np.abs(policies[distinct[-1]] - policies[np.inf])) <= 1e-4:
            distinct[-1] = np.inf

    # saturation check
    if not np.isinf(distinct[-1]) or len(distinct) < len(candidates):
        sat = None
        for a in candidates:
            if np.max(np.abs(policies[a] - policies[np.inf])) <= 1e-4:
                sat = a
                break
        if sat is not None and not np.isinf(sat) and sat > 0:
            print(f"\n  Saturation: policy is frozen for alpha >= {sat:g} "
                  f"(identical to alpha=inf). Values above are redundant.")
            print("  For a richer sweep, give clients a spread of "
                  "availability rates instead of two fixed groups.")

    # ---- recommendation: spread across the distinct set, prefer named anchors -
    pref = [a for a in (0.0, 1.0, np.inf) if a in distinct]
    others = [a for a in distinct if a not in pref]
    while len(pref) < min(6, len(distinct)) and others:
        # pick the 'other' alpha most spaced from current picks (on lambda axis)
        def lam(a):
            return 1.0 if np.isinf(a) else a / (1.0 + a)
        best = max(others, key=lambda a: min(abs(lam(a) - lam(b)) for b in pref))
        pref.append(best)
        others.remove(best)
    recommended = sorted(pref, key=lambda a: (np.isinf(a), a))

    rec_str = ", ".join("inf" if np.isinf(a) else f"{a:g}" for a in recommended)
    print(f"\n  RECOMMENDED alpha values ({len(recommended)} distinct policies):")
    print(f"    {rec_str}")
    print(f"  Pass each to the simulator as:  --temp <value>   (use 'inf' for inf)")
    print(f"  alpha=0 -> throughput/greedy | alpha=1 -> prop. fair | "
          f"alpha=inf -> equal impact\n")

    # ---- figure ---------------------------------------------------------------
    order = np.argsort(-np.where(pon > 0, r / pon, 0))   # by gamma, descending
    lam_grid = np.linspace(0.0, 0.999, 60)
    alpha_grid = lam_grid / (1.0 - lam_grid)
    C_curve, J_curve = [], []
    for a in alpha_grid:
        s = summarise(compute_pi(r, pon, a, K), r, pon, K)
        C_curve.append(s["throughput"])
        J_curve.append(s["jain"])

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    for a in recommended:
        lab = ("alpha=inf" if np.isinf(a) else f"alpha={a:g}")
        ax[0, 0].plot(policies[a][order], marker=".", ms=3, label=lab)
        ax[0, 1].plot((policies[a] * r)[order], marker=".", ms=3, label=lab)
    ax[0, 0].set_title("Participation probability  p_u  (clients sorted by gamma)")
    ax[0, 0].set_xlabel("client rank"); ax[0, 0].set_ylabel("p_u")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].set_title("Contribution  x_u = p_u * R_u  (equal => flat line)")
    ax[0, 1].set_xlabel("client rank"); ax[0, 1].set_ylabel("p_u * R_u")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    axb = ax[1, 0]
    axb.plot(lam_grid, C_curve, "b-", label="throughput  C")
    axb.set_xlabel("lambda = alpha/(1+alpha)   [0=greedy ... 1=equal impact]")
    axb.set_ylabel("throughput  C", color="b")
    axb.tick_params(axis="y", labelcolor="b")
    axj = axb.twinx()
    axj.plot(lam_grid, J_curve, "r-", label="Jain fairness")
    axj.set_ylabel("Jain fairness", color="r")
    axj.tick_params(axis="y", labelcolor="r")
    axb.set_title("Efficiency and fairness vs alpha")
    axb.grid(alpha=0.3)

    axp = ax[1, 1]
    axp.plot(J_curve, C_curve, "-", color="0.6", zorder=1)
    for a in recommended:
        s = summarise(policies[a], r, pon, K)
        axp.scatter(s["jain"], s["throughput"], s=60, zorder=2)
        axp.annotate("inf" if np.isinf(a) else f"{a:g}",
                     (s["jain"], s["throughput"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    axp.set_title("Fairness-efficiency Pareto front")
    axp.set_xlabel("Jain fairness  (1 = equal contribution)")
    axp.set_ylabel("throughput  C")
    axp.grid(alpha=0.3)

    fig.suptitle(f"alpha-fair policy diagnostic  "
                 f"(N={N}, disc={disc}, K={K:g})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"  Figure written to: {args.out}\n")


if __name__ == "__main__":
    main()
