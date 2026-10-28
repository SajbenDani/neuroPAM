"""
validate_theorem.py
-------------------
Numerical validation of Theorem 3.1: the Wasserstein hybrid map minimises the
expected squared displacement E[||T(z) - z||^2] among all admissible affine maps.

Theorem (informal):
    Among all affine maps T(z) = Az + b satisfying
        (i)  exact covariance match on the complement subspace W_perp:
             W_perp.T @ (A @ Sigma_s @ A.T) @ W_perp == W_perp.T @ Sigma_t @ W_perp
        (ii) biology axis w preserved: A*w = w, b = mu_t - A*mu_s (full mean match),
    the Wasserstein hybrid map achieves the minimum expected squared displacement
        E_{z ~ N(mu_s, Sigma_s)}[||T(z) - z||^2].

The Wasserstein complement block is:
    B_W = Sigma_s_perp^{-1/2} (Sigma_s_perp^{1/2} Sigma_t_perp Sigma_s_perp^{1/2})^{1/2}
          Sigma_s_perp^{-1/2}

Competing maps are formed by substituting B_W @ O for B_W in the complement block,
where O is a random orthogonal matrix in SO(d-1).  The expected squared displacement
is computed analytically as:
    E[||T(z) - z||^2] = ||mu_t - mu_s||^2 + tr((A - I) @ Sigma_s @ (A - I).T)

Usage:
    python scripts/validate_theorem.py --n-trials 1000 --d 64 --n-competitors 100 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from vae_harmonization.harmonization.transport import bures_wasserstein_map  # noqa: E402


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_random_gaussian_pair(d, seed):
    """Generate two random Gaussian distributions in R^d."""
    rng = np.random.default_rng(seed)
    raw_s = rng.standard_normal(d)
    mean_s = raw_s / np.linalg.norm(raw_s) * (0.5 + rng.uniform())
    raw_t = rng.standard_normal(d)
    mean_t = raw_t / np.linalg.norm(raw_t) * (0.5 + rng.uniform())
    A_s = rng.standard_normal((d, d))
    Sigma_s = A_s @ A_s.T / d + 0.1 * np.eye(d)
    A_t = rng.standard_normal((d, d))
    Sigma_t = A_t @ A_t.T / d + 0.1 * np.eye(d)
    return mean_s, mean_t, Sigma_s, Sigma_t


def generate_random_unit_w(d, seed):
    """Sample a uniformly random unit vector in R^d via QR."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(M)
    return Q[:, 0]


def generate_random_orthogonal(d, seed):
    """Random orthogonal matrix. Odd seeds give a small rotation near the identity, so some
    competitors sit close to the optimum rather than far away."""
    rng = np.random.default_rng(seed)
    if seed % 2:
        from scipy.linalg import expm
        K = rng.standard_normal((d, d))
        K = (K - K.T) / np.sqrt(2 * d)
        return expm(10 ** rng.uniform(-3, -1) * K)
    M = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(M)
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]
    return Q


# ---------------------------------------------------------------------------
# Expected squared displacement (analytical)
# ---------------------------------------------------------------------------

def expected_sq_displacement(A, mean_s, mean_t, Sigma_s):
    """
    Compute E_{z ~ N(mu_s, Sigma_s)}[||Az + b - z||^2]
    where b = mean_t - A @ mean_s (full mean matching).

    = ||mean_t - mean_s||^2 + tr((A - I) @ Sigma_s @ (A - I).T)
    """
    mean_diff_sq = float(np.sum((mean_t - mean_s) ** 2))
    diff = A - np.eye(A.shape[0])
    trace_term = float(np.trace(diff @ Sigma_s @ diff.T))
    return mean_diff_sq + trace_term


# ---------------------------------------------------------------------------
# Core map computations
# ---------------------------------------------------------------------------

def compute_wasserstein_map(mean_s, mean_t, Sigma_s, Sigma_t, w):
    """
    Compute the Wasserstein hybrid harmonization map.

    A = w @ w.T + W_perp @ B_W @ W_perp.T
    where B_W is the Wasserstein optimal transport block on the complement.

    Returns
    -------
    A : (d, d) linear part
    W_perp : (d, d-1) complement basis
    Sigma_s_perp : (d-1, d-1) source complement covariance
    Sigma_t_perp : (d-1, d-1) target complement covariance
    esd_hybrid : float, E[||T(z) - z||^2]
    """
    d = len(w)
    w = w / np.linalg.norm(w)

    W_perp = null_space(w.reshape(1, -1))  # (d, d-1)

    reg = 1e-8 * np.eye(d - 1)
    Sigma_s_perp = W_perp.T @ Sigma_s @ W_perp + reg
    Sigma_t_perp = W_perp.T @ Sigma_t @ W_perp + reg

    B_W = bures_wasserstein_map(Sigma_s_perp, Sigma_t_perp)

    w_col = w.reshape(-1, 1)
    A = w_col @ w_col.T + W_perp @ B_W @ W_perp.T

    esd_hybrid = expected_sq_displacement(A, mean_s, mean_t, Sigma_s)

    return A, W_perp, Sigma_s_perp, Sigma_t_perp, B_W, esd_hybrid


def compute_competitor_esd(W_perp, B_W, S_half, S_inv_half, w, mean_s, mean_t, Sigma_s, O):
    """
    Expected squared displacement of a feasible competitor map.

    Every feasible complement block has the form B = C S^{-1/2} with C C^T = Sigma_t_perp
    and S = Sigma_s_perp. The competitor takes C = (B_W S^{1/2}) O for a random orthogonal O,
    so B = B_W S^{1/2} O S^{-1/2} still satisfies B S B^T = Sigma_t_perp.
    """
    B_comp = B_W @ S_half @ O @ S_inv_half

    w_col = w.reshape(-1, 1)
    A_comp = w_col @ w_col.T + W_perp @ B_comp @ W_perp.T

    return expected_sq_displacement(A_comp, mean_s, mean_t, Sigma_s)


# ---------------------------------------------------------------------------
# Constraint verification
# ---------------------------------------------------------------------------

def verify_constraints(A, mean_s, mean_t, Sigma_s, Sigma_t_perp, W_perp, w, tol=1e-5):
    """Verify that the Wasserstein map satisfies constraints (i) and (ii)."""
    d = len(w)
    w_unit = w / np.linalg.norm(w)

    # (i) Complement covariance match
    transformed_cov_perp = W_perp.T @ (A @ Sigma_s @ A.T) @ W_perp
    cov_residual = float(np.max(np.abs(transformed_cov_perp - Sigma_t_perp)))
    cov_match = cov_residual < tol

    # (ii) Biology axis preservation
    axis_residual = float(np.linalg.norm(A @ w_unit - w_unit))
    axis_preserved = axis_residual < 1e-7

    # (iii) Full mean matching: b = mean_t - A @ mean_s => T(mean_s) = mean_t
    b = mean_t - A @ mean_s
    mean_residual = float(np.linalg.norm(A @ mean_s + b - mean_t))
    mean_match = mean_residual < 1e-8

    all_pass = cov_match and axis_preserved and mean_match

    return {
        'cov_match': cov_match,
        'cov_residual': cov_residual,
        'axis_preserved': axis_preserved,
        'axis_residual': axis_residual,
        'mean_match': mean_match,
        'mean_residual': mean_residual,
        'all_pass': all_pass,
    }


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_validation(n_trials=1000, d=64, n_competitors_per_trial=100, seed=0, out_dir=None):
    """
    Validate Theorem 3.1: Wasserstein hybrid map minimises E[||T(z) - z||^2].

    For each trial:
    - Constructs the Wasserstein hybrid map.
    - Generates n_competitors_per_trial competitor maps (each with a random
      orthogonal rotation O applied to the complement block; all feasible).
    - Records gap = min_competitor_ESD - wasserstein_ESD.

    A positive gap in every trial confirms the theorem.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if out_dir is None:
        out_dir = repo_root / 'results' / 'theorem'
    else:
        out_dir = Path(out_dir)

    figures_dir = repo_root / 'paper' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    gaps = []
    esd_hybrids = []
    esd_min_competitors = []
    constraint_passes = []
    competitor_residuals = []

    print(f"Validating Theorem 3.1 (Wasserstein optimal transport):")
    print(f"  Objective: E[||T(z) - z||^2] (expected squared displacement)")
    print(f"  n_trials={n_trials}, d={d}, n_competitors={n_competitors_per_trial}, seed={seed}")
    print("-" * 70)

    for t in range(n_trials):
        mean_s, mean_t, Sigma_s, Sigma_t = generate_random_gaussian_pair(d, seed=seed + t)
        w = generate_random_unit_w(d, seed=seed + n_trials + t)

        A, W_perp, Sigma_s_perp, Sigma_t_perp, B_W, esd_hybrid = compute_wasserstein_map(
            mean_s, mean_t, Sigma_s, Sigma_t, w)

        constraint_result = verify_constraints(
            A, mean_s, mean_t, Sigma_s, Sigma_t_perp, W_perp, w)
        constraint_passes.append(constraint_result['all_pass'])

        ev, V = np.linalg.eigh(Sigma_s_perp)
        S_half, S_inv_half = (V * np.sqrt(ev)) @ V.T, (V / np.sqrt(ev)) @ V.T
        esd_competitors = []
        for k in range(n_competitors_per_trial):
            comp_seed = seed + 2 * n_trials + t * n_competitors_per_trial + k
            O = generate_random_orthogonal(d - 1, seed=comp_seed)
            if k == 0:
                B_c = B_W @ S_half @ O @ S_inv_half
                competitor_residuals.append(float(np.max(np.abs(B_c @ Sigma_s_perp @ B_c.T - Sigma_t_perp))))
            esd_comp = compute_competitor_esd(
                W_perp, B_W, S_half, S_inv_half, w, mean_s, mean_t, Sigma_s, O)
            esd_competitors.append(esd_comp)

        min_comp = float(np.min(esd_competitors))
        gap = min_comp - esd_hybrid

        gaps.append(gap)
        esd_hybrids.append(esd_hybrid)
        esd_min_competitors.append(min_comp)

        if (t + 1) % 100 == 0:
            print(f"  Trial {t+1}/{n_trials}: mean_gap_so_far={np.mean(gaps):.6f}, "
                  f"n_negative_gaps={sum(g < 0 for g in gaps)}")

    gaps = np.array(gaps)
    esd_hybrids = np.array(esd_hybrids)
    esd_min_competitors = np.array(esd_min_competitors)

    stat, p_value = wilcoxon(gaps, alternative='greater')

    n_positive = int(np.sum(gaps > 0))
    n_zero = int(np.sum(gaps == 0))
    n_negative = int(np.sum(gaps < 0))
    all_constraints_pass = all(constraint_passes)
    n_constraints_passed = sum(constraint_passes)

    print("-" * 70)
    print(f"Results over {n_trials} trials (d={d}):")
    print(f"  Gap = min_competitor_ESD - wasserstein_ESD")
    print(f"  Mean gap      : {np.mean(gaps):.6f}")
    print(f"  Std gap       : {np.std(gaps):.6f}")
    print(f"  Min gap       : {np.min(gaps):.6f}")
    print(f"  Max gap       : {np.max(gaps):.6f}")
    print(f"  Positive gaps : {n_positive}/{n_trials}")
    print(f"  Zero gaps     : {n_zero}/{n_trials}")
    print(f"  Negative gaps : {n_negative}/{n_trials}")
    print(f"  Wilcoxon stat : {stat:.4f}")
    print(f"  p-value       : {p_value:.2e}")
    print(f"  Constraints passed: {n_constraints_passed}/{n_trials}")
    print(f"  Competitor constraint residual (max): {np.max(competitor_residuals):.2e}")
    print(f"  Theorem SUPPORTED: {n_negative == 0 and p_value < 0.05}")

    results = {
        'theorem': 'Wasserstein map minimises E[||T(z)-z||^2] (Theorem 3.1)',
        'objective': 'expected_squared_displacement',
        'n_trials': n_trials,
        'd': d,
        'n_competitors_per_trial': n_competitors_per_trial,
        'seed': seed,
        'gap_mean': float(np.mean(gaps)),
        'gap_std': float(np.std(gaps)),
        'gap_min': float(np.min(gaps)),
        'gap_max': float(np.max(gaps)),
        'gap_median': float(np.median(gaps)),
        'n_positive_gaps': n_positive,
        'n_zero_gaps': n_zero,
        'n_negative_gaps': n_negative,
        'wilcoxon_stat': float(stat),
        'p_value': float(p_value),
        'n_constraints_passed': n_constraints_passed,
        'all_constraints_passed': bool(all_constraints_pass),
        'theorem_supported': bool(n_negative == 0 and p_value < 0.05),
        'esd_hybrid_mean': float(np.mean(esd_hybrids)),
        'esd_competitor_min_mean': float(np.mean(esd_min_competitors)),
        'competitor_constraint_residual_max': float(np.max(competitor_residuals)),
        'gaps': gaps.tolist(),
    }

    json_path = out_dir / 'theorem_validation.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gaps, bins=50, color='steelblue', edgecolor='white', alpha=0.85,
            label=f'gap values (n={n_trials})')
    ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5,
               label='gap = 0 (null)')
    ax.axvline(x=float(np.mean(gaps)), color='orange', linestyle='-', linewidth=1.5,
               label=f'mean gap = {np.mean(gaps):.4f}')

    ax.set_xlabel('gap = min_competitor_ESD - wasserstein_ESD', fontsize=12)
    ax.set_ylabel('count', fontsize=12)
    ax.set_title(
        f'Theorem Validation: Expected Squared Displacement Gap\n'
        f'mean gap = {np.mean(gaps):.4f}, p-value = {p_value:.2e} '
        f'(Wilcoxon, alternative=greater)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig_path = figures_dir / 'theorem_validation.pdf'
    fig.tight_layout()
    fig.savefig(fig_path, format='pdf')
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    return results


# ---------------------------------------------------------------------------
# Plug-in covariance estimators
# ---------------------------------------------------------------------------

def run_estimator_check(n_rep=50, d=99, sizes=(150, 100, 60, 45, 35, 25, 20, 15), seed=0, out_dir=None,
                        eig_ratio=14.0, perturbation=0.35):
    """
    The theorem holds for whatever covariances are plugged in; this check asks how good the plug-in
    is.  Sites are sampled from Gaussians with an anisotropic spectrum (eigenvalue ratio ``eig_ratio``
    between the first and last complement direction, as for PCA scores) under two regimes:

    * shared   : every site has the hub's covariance (the ideal map is the identity);
    * distinct : site covariances are random SPD perturbations of the hub's.

    For every estimator the map ``B_s`` is fitted from the samples and scored on the *population*:
    the Bures-Wasserstein distance between ``B_s Sigma_s B_s`` and the true hub covariance (how far
    the harmonized site is from the hub) and the expected squared displacement ``tr((B_s-I) Sigma_s (B_s-I))``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    from vae_harmonization.harmonization import ProtectedAffineHarmonizer
    from vae_harmonization.harmonization.transport import (
        bures_wasserstein_distance2, expected_squared_displacement, sqrtm_psd,
    )

    out_dir = Path(out_dir) if out_dir is not None else repo_root / 'results' / 'theorem'
    figures_dir = repo_root / 'paper' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    estimators = ('pooled', 'lw', 'diagonal', 'empirical')
    ev = np.geomspace(eig_ratio, 1.0, d)
    rows = []
    print(f"Plug-in estimator check: d={d}, sites={sizes}, {n_rep} replicates")
    for regime in ('shared', 'distinct'):
        for r in range(n_rep):
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            Sigma_hub = (Q * ev) @ Q.T
            S_h = sqrtm_psd(Sigma_hub)
            covs = []
            for s in range(len(sizes)):
                if s == 0 or regime == 'shared':
                    covs.append(Sigma_hub)
                else:
                    G = rng.standard_normal((d, d)) / np.sqrt(d)
                    R = np.eye(d) + perturbation * (G + G.T) / np.sqrt(2)
                    R = R @ R.T                                   # random SPD perturbation around I
                    covs.append(S_h @ R @ S_h)
            Z = np.vstack([rng.multivariate_normal(np.zeros(d), C, size=n) for n, C in zip(sizes, covs)])
            ys = np.repeat(np.arange(len(sizes)), sizes)
            for est in estimators:
                h = ProtectedAffineHarmonizer(target_site=0, protected=(), covariance=est, min_cov_samples=10)
                h.fit(Z, ys)
                for s in range(1, len(sizes)):
                    B = h.site_transforms_[s].B
                    rows.append({'regime': regime, 'rep': r, 'estimator': est, 'site_n': int(sizes[s]),
                                 'w2_pre': bures_wasserstein_distance2(covs[s], Sigma_hub),
                                 'w2_post': bures_wasserstein_distance2(B @ covs[s] @ B.T, Sigma_hub),
                                 'displacement': expected_squared_displacement(B, covs[s]),
                                 'shrinkage': h.site_transforms_[s].shrinkage})
            for s in range(1, len(sizes)):                        # oracle: population covariances
                B = bures_wasserstein_map(covs[s], Sigma_hub)
                rows.append({'regime': regime, 'rep': r, 'estimator': 'oracle', 'site_n': int(sizes[s]),
                             'w2_pre': bures_wasserstein_distance2(covs[s], Sigma_hub), 'w2_post': 0.0,
                             'displacement': expected_squared_displacement(B, covs[s]), 'shrinkage': 0.0})
    # aggregate: regime x estimator x site size
    summary = {}
    for regime in ('shared', 'distinct'):
        summary[regime] = {}
        for est in (*estimators, 'oracle'):
            summary[regime][est] = {}
            for n in sizes[1:]:
                sel = [x for x in rows if x['regime'] == regime and x['estimator'] == est and x['site_n'] == n]
                summary[regime][est][str(n)] = {
                    'w2_pre': float(np.mean([x['w2_pre'] for x in sel])),
                    'w2_post': float(np.mean([x['w2_post'] for x in sel])),
                    'w2_post_median': float(np.median([x['w2_post'] for x in sel])),
                    'displacement': float(np.mean([x['displacement'] for x in sel])),
                    'shrinkage': float(np.mean([x['shrinkage'] for x in sel])),
                }
    results = {'settings': {'n_rep': n_rep, 'd': d, 'sizes': list(sizes), 'seed': seed, 'eig_ratio': eig_ratio,
                            'perturbation': perturbation},
               'summary': summary}
    json_path = out_dir / 'estimator_check.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {json_path}")
    for regime in ('shared', 'distinct'):
        print(f"  [{regime}]  mean W2^2 to hub after harmonization (pre: "
              f"{np.mean([summary[regime]['oracle'][str(n)]['w2_pre'] for n in sizes[1:]]):.2f})")
        print(f"    {'n':>5}" + "".join(f"{e:>12}" for e in (*estimators, 'oracle')))
        for n in sizes[1:]:
            print(f"    {n:>5}" + "".join(f"{summary[regime][e][str(n)]['w2_post']:>12.4g}" for e in (*estimators, 'oracle')))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors = {'pooled': 'tab:blue', 'lw': 'tab:orange', 'diagonal': 'tab:green', 'empirical': 'tab:red', 'oracle': 'k'}
    for ax, regime in zip(axes, ('shared', 'distinct')):
        ns = list(sizes[1:])
        for est in (*estimators, 'oracle'):
            ax.plot(ns, [summary[regime][est][str(n)]['w2_post'] for n in ns], marker='o', color=colors[est],
                    label=est, linestyle='--' if est == 'oracle' else '-')
        ax.plot(ns, [summary[regime]['oracle'][str(n)]['w2_pre'] for n in ns], color='grey', linestyle=':',
                label='before harmonization')
        ax.set_yscale('log')
        ax.set_xlabel('site size n')
        ax.set_ylabel('$W_2^2$ to hub (population)')
        ax.set_title(f'{regime} covariances')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig_path = figures_dir / 'estimator_check.pdf'
    fig.savefig(fig_path, format='pdf')
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Numerically validate Theorem 3.1 and check the plug-in covariance estimators.')
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--n-competitors', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--skip-theorem', action='store_true')
    parser.add_argument('--skip-estimator-check', action='store_true')
    parser.add_argument('--n-rep', type=int, default=50, help='replicates of the estimator check')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    if not args.skip_theorem:
        run_validation(
            n_trials=args.n_trials,
            d=args.d,
            n_competitors_per_trial=args.n_competitors,
            seed=args.seed,
            out_dir=out_dir,
        )
    if not args.skip_estimator_check:
        run_estimator_check(n_rep=args.n_rep, seed=args.seed, out_dir=out_dir)


if __name__ == '__main__':
    main()
