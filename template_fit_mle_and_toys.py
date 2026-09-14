import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit

from template_fit_multicanale import (
    ChannelConfig,
    CHANNELS,
    build_counts_and_variance,
    check_template_population,
    fit_signal_strengths,
)


# =====================================================================
# CONFIGURAZIONE
# =====================================================================
N_TOYS: int = 10_000
RANDOM_SEED: int = 12345
TOY_MODE: str = "poisson_effective"  # "poisson_effective" oppure "gaussian"

OUTPUT_DIR: str = "output_template_fit_validation"


# =====================================================================
# CROSS-CHECK MLE (iminuit)
# =====================================================================
def run_mle_cross_check(
    A: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    signal_names: Sequence[str],
    wls_result: Dict[str, np.ndarray],
) -> Dict:
    n_signal = A.shape[1]
    param_names = [f"mu{i}" for i in range(n_signal)]

    def chi2(*mu):
        mu_arr = np.array(mu)
        residual = y - A @ mu_arr
        return float(np.sum((residual / sigma) ** 2))

    m = Minuit(chi2, *np.ones(n_signal), name=param_names)
    m.errordef = Minuit.LEAST_SQUARES  # dice a iminuit che chi2 e' gia' su scala corretta
    m.migrad()
    m.hesse()

    mu_mle = np.array([m.values[name] for name in param_names])
    unc_mle = np.array([m.errors[name] for name in param_names])

    print("\n--- Cross-check MLE (iminuit) vs WLS analitico ---")
    print(f"{'processo':<8} {'mu (WLS)':>12} {'mu (MLE)':>12} {'unc (WLS)':>12} {'unc (MLE)':>12}")
    for i, name in enumerate(signal_names):
        print(f"{name:<8} {wls_result['mu'][i]:>12.4f} {mu_mle[i]:>12.4f} "
              f"{wls_result['unc'][i]:>12.4f} {unc_mle[i]:>12.4f}")
    print(f"chi2 WLS = {wls_result['chi2']:.4f}   fval MLE (Migrad) = {m.fval:.4f}")
    if not m.valid:
        print("  [ATTENZIONE] Migrad non converge in modo valido: ispeziona m.fmin.")

    return {
        "mu": mu_mle,
        "unc": unc_mle,
        "cov": np.array(m.covariance) if m.covariance is not None else None,
        "fval": m.fval,
        "valid": m.valid,
    }

def save_mle_comparison(
    wls_result: Dict, mle_result: Dict, channel: ChannelConfig, output_dir: str
) -> None:
    df = pd.DataFrame({
        "processo": channel.signal_names,
        "mu_wls": wls_result["mu"],
        "unc_wls": wls_result["unc"],
        "mu_mle": mle_result["mu"],
        "unc_mle": mle_result["unc"],
    })
    df["delta_mu"] = df["mu_wls"] - df["mu_mle"]
    df.to_csv(os.path.join(output_dir, "confronto_wls_mle.csv"), index=False)

    max_delta = float(df["delta_mu"].abs().max())
    if max_delta > 1e-3:
        print(
            f"  [ATTENZIONE] WLS e MLE differiscono fino a {max_delta:.2e}: per un "
            "modello lineare con errori gaussiani dovrebbero coincidere quasi "
            "esattamente. Controlla la convergenza di Migrad (m.valid/m.fmin)."
        )


# =====================================================================
# TOY MC
# =====================================================================
def generate_toy_poisson_effective(
    y_nominal: np.ndarray,
    y_obs_ref: np.ndarray,
    y_obs_var_ref: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    w_eff = np.divide(
        y_obs_var_ref, y_obs_ref,
        out=np.zeros_like(y_obs_ref), where=y_obs_ref > 0,
    )
    n_eff = np.divide(
        y_nominal, w_eff,
        out=np.zeros_like(y_nominal), where=w_eff > 0,
    )
    n_drawn = rng.poisson(n_eff)
    return n_drawn * w_eff

def generate_toy_residual_poisson_effective(
    y_nominal: np.ndarray,
    bkg_pred: np.ndarray,
    bkg_var: np.ndarray,
    y_obs_ref: np.ndarray,
    y_obs_var_ref: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    data_component = generate_toy_poisson_effective(y_nominal, y_obs_ref, y_obs_var_ref, rng)
    bkg_component = generate_toy_poisson_effective(bkg_pred, bkg_pred, bkg_var, rng)
    return data_component - bkg_component

def _solve_wls_from_residual(cov: np.ndarray, AtW: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return cov @ (AtW @ residual)

def run_toy_mc(
    A: np.ndarray,
    bkg_pred: np.ndarray,
    bkg_var: np.ndarray,
    sigma2: np.ndarray,
    mu_true: np.ndarray,
    y_obs_ref: np.ndarray,
    y_obs_var_ref: np.ndarray,
    n_toys: int,
    rng: np.random.Generator,
    mode: str,
) -> Dict[str, np.ndarray]:
    n_signal = A.shape[1]
    sigma = np.sqrt(sigma2)
    y_nominal = A @ mu_true + bkg_pred
    residual_nominal = A @ mu_true  

    W = np.diag(1.0 / sigma2)
    AtW = A.T @ W
    cov = np.linalg.inv(AtW @ A)  
    unc = np.sqrt(np.diag(cov))

    all_mu = np.empty((n_toys, n_signal))
    for t in range(n_toys):
        if mode == "gaussian":
            residual_toy = residual_nominal + rng.normal(0.0, sigma)
        elif mode == "poisson_effective":
            residual_toy = generate_toy_residual_poisson_effective(
                y_nominal, bkg_pred, bkg_var, y_obs_ref, y_obs_var_ref, rng
            )
        else:
            raise ValueError(f"TOY_MODE sconosciuto: {mode!r}")
        all_mu[t] = _solve_wls_from_residual(cov, AtW, residual_toy)

    pulls = (all_mu - mu_true[np.newaxis, :]) / unc[np.newaxis, :]
    return {"mu": all_mu, "pulls": pulls, "unc": unc, "y_nominal": y_nominal}

def plot_pull_distributions(
    toy_result: Dict[str, np.ndarray], channel: ChannelConfig, output_dir: str
) -> None:
    pulls = toy_result["pulls"]
    n_signal = pulls.shape[1]
    n_cols = min(3, n_signal)
    n_rows = int(np.ceil(n_signal / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    bins = np.linspace(-5, 5, 41)
    x_gauss = np.linspace(-5, 5, 200)
    gauss_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss ** 2)

    for i, name in enumerate(channel.signal_names):
        ax = axes[i // n_cols][i % n_cols]
        p = pulls[:, i]
        ax.hist(p, bins=bins, density=True, alpha=0.6, color="tab:blue", label="Pull")
        mean, std = float(np.mean(p)), float(np.std(p))
        ax.plot(x_gauss, gauss_pdf, "k--", label="N(0,1)")
        ax.set_title(f"{name}\nmean={mean:.3f}, sigma={std:.3f}")
        ax.set_xlabel("pull")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(n_signal, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        f"Pull distribution — {channel.display_label()} "
        f"(N_TOYS={pulls.shape[0]}, mode={TOY_MODE})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pull_distributions.png"), dpi=200)
    plt.close(fig)

def save_toy_summary(toy_result: Dict[str, np.ndarray], channel: ChannelConfig, output_dir: str) -> None:
    pulls = toy_result["pulls"]
    mu_toys = toy_result["mu"]
    summary = pd.DataFrame({
        "processo": channel.signal_names,
        "pull_media": pulls.mean(axis=0),
        "pull_sigma": pulls.std(axis=0),
        "mu_toy_media": mu_toys.mean(axis=0),
        "mu_toy_std": mu_toys.std(axis=0),
        "unc_analitica": toy_result["unc"],
    })
    summary.to_csv(os.path.join(output_dir, "riassunto_toy_mc.csv"), index=False)
    print(f"\n--- Riassunto Toy MC ({TOY_MODE}, N={N_TOYS}) ---")
    print(summary.round(4).to_string(index=False))


# =====================================================================
# ORCHESTRAZIONE (un canale = stessi parquet di template_fit_multicanale)
# =====================================================================
def load_channel_matrices(channel: ChannelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_template = pd.read_parquet(channel.template_events_parquet)
    df_composizione = pd.read_parquet(channel.composition_events_parquet)
    counts_template, variance_template = build_counts_and_variance(df_template, channel.class_names)
    counts_composizione, variance_composizione = build_counts_and_variance(df_composizione, channel.class_names)
    return counts_template, variance_template, counts_composizione, variance_composizione

def validate_channel(channel: ChannelConfig) -> None:
    print(f"\n{'=' * 70}\n VALIDATION: {channel.display_label()}\n{'=' * 70}")

    counts_template, variance_template, counts_composizione, variance_composizione = load_channel_matrices(channel)
    check_template_population(counts_template, channel)

    n_signal = channel.n_signal
    bkg_idx = n_signal

    A = counts_template[:n_signal, :].T
    bkg_pred = counts_template[bkg_idx, :]
    bkg_var = variance_template[bkg_idx, :]
    y_obs = counts_composizione.sum(axis=0)
    y_obs_var = variance_composizione.sum(axis=0)
    sigma2 = y_obs_var + bkg_var
    sigma2 = np.where(sigma2 <= 0, np.inf, sigma2)
    sigma = np.sqrt(sigma2)
    y = y_obs - bkg_pred

    output_dir = os.path.join(OUTPUT_DIR, channel.name)
    os.makedirs(output_dir, exist_ok=True)

    wls_result = fit_signal_strengths(
        counts_template, variance_template, counts_composizione, variance_composizione, n_signal=n_signal,
    )

    mle_result = run_mle_cross_check(A, y, sigma, channel.signal_names, wls_result)
    save_mle_comparison(wls_result, mle_result, channel, output_dir)

    rng = np.random.default_rng(RANDOM_SEED)
    mu_true = np.ones(n_signal)
    toy_result = run_toy_mc(
        A, bkg_pred, bkg_var, sigma2, mu_true, y_obs, y_obs_var, N_TOYS, rng, TOY_MODE,
    )
    plot_pull_distributions(toy_result, channel, output_dir)
    save_toy_summary(toy_result, channel, output_dir)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for channel in CHANNELS:
        try:
            validate_channel(channel)
        except FileNotFoundError as exc:
            print(
                f"\n[SALTATO] canale '{channel.display_label()}': file non trovato "
                f"({exc}). Aggiorna i path in CHANNELS (template_fit_multicanale.py) "
                "quando avrai prodotto quell'output."
            )

    print(f"\nRisultati di validazione salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()