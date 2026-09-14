import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from template_fit_multicanale import ChannelConfig, CHANNELS, check_template_population
from template_fit_mle_and_toys import load_channel_matrices, run_toy_mc


# =====================================================================
# CONFIGURAZIONE
# =====================================================================
MU_SCAN_VALUES: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_TOYS_PER_POINT: int = 2000
RANDOM_SEED: int = 98765
TOY_MODE: str = "poisson_effective"  

OUTPUT_DIR: str = "output_template_fit_linearity"


# =====================================================================
# SCANSIONE
# =====================================================================
def run_linearity_scan(
    A: np.ndarray,
    bkg_pred: np.ndarray,
    bkg_var: np.ndarray,
    sigma2: np.ndarray,
    y_obs: np.ndarray,
    y_obs_var: np.ndarray,
    n_signal: int,
    scanned_idx: int,
    scan_values: Sequence[float],
    n_toys: int,
    rng: np.random.Generator,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_mu = np.zeros((len(scan_values), n_signal))
    std_mu = np.zeros((len(scan_values), n_signal))
    analytic_unc = None

    for k, mu_scan in enumerate(scan_values):
        mu_true = np.ones(n_signal)
        mu_true[scanned_idx] = mu_scan

        toy_result = run_toy_mc(
            A, bkg_pred, bkg_var, sigma2, mu_true, y_obs, y_obs_var, n_toys, rng, mode,
        )
        mean_mu[k, :] = toy_result["mu"].mean(axis=0)
        std_mu[k, :] = toy_result["mu"].std(axis=0)
        if analytic_unc is None:
            analytic_unc = toy_result["unc"]

    return mean_mu, std_mu, analytic_unc

def plot_linearity(
    scan_values: Sequence[float],
    mean_mu: np.ndarray,
    std_mu: np.ndarray,
    analytic_unc: np.ndarray,
    signal_names: Sequence[str],
    scanned_idx: int,
    channel: ChannelConfig,
    output_dir: str,
) -> None:
    scan_arr = np.array(scan_values)
    sem = std_mu / np.sqrt(N_TOYS_PER_POINT)  

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                             gridspec_kw={"height_ratios": [2, 1]})

    ax_top.plot(scan_arr, scan_arr, "k--", label="y = x (expected)", zorder=1)
    for j, name in enumerate(signal_names):
        if j == scanned_idx:
            ax_top.errorbar(scan_arr, mean_mu[:, j], yerr=sem[:, j], fmt="o-", color="tab:red",
                             label=f"{name} (injected)", linewidth=2, capsize=4, zorder=5)
        else:
            ax_top.errorbar(scan_arr, mean_mu[:, j], yerr=sem[:, j], fmt="s--", alpha=0.6,
                             label=f"{name} (expected flat at 1)", capsize=3)
    ax_top.set_ylabel("mu estimated (mean over toys)")
    ax_top.set_title(
        f"Linearity Test — {channel.display_label()}\n"
        f"scanned signal: {signal_names[scanned_idx]} (mode={TOY_MODE}, N={N_TOYS_PER_POINT}/point)"
    )
    ax_top.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax_top.grid(alpha=0.3)

    ratio = std_mu[:, scanned_idx] / analytic_unc[scanned_idx]
    ax_bottom.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_bottom.plot(scan_arr, ratio, "o-", color="tab:purple")
    ax_bottom.set_xlabel(f"mu true injected for {signal_names[scanned_idx]}")
    ax_bottom.set_ylabel("std empirical / unc. declared")
    ax_bottom.grid(alpha=0.3)

    plt.tight_layout()
    clean_name = signal_names[scanned_idx].replace("+", "_").replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"linearita_{clean_name}.png"), dpi=200)
    plt.close(fig)


# =====================================================================
# ORCHESTRAZIONE
# =====================================================================
def validate_linearity_for_channel(channel: ChannelConfig) -> None:
    print(f"\n{'=' * 70}\n TEST DI LINEARITA': {channel.display_label()}\n{'=' * 70}")

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

    output_dir = os.path.join(OUTPUT_DIR, channel.name)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []

    for scanned_idx, name in enumerate(channel.signal_names):
        print(f"  Scansione su {name}...")
        mean_mu, std_mu, analytic_unc = run_linearity_scan(
            A, bkg_pred, bkg_var, sigma2, y_obs, y_obs_var, n_signal,
            scanned_idx, MU_SCAN_VALUES, N_TOYS_PER_POINT, rng, TOY_MODE,
        )
        plot_linearity(
            MU_SCAN_VALUES, mean_mu, std_mu, analytic_unc,
            channel.signal_names, scanned_idx, channel, output_dir,
        )

        for k, mu_scan in enumerate(MU_SCAN_VALUES):
            summary_rows.append({
                "segnale_scansionato": name,
                "mu_iniettato": mu_scan,
                "mu_stimato_medio": mean_mu[k, scanned_idx],
                "scarto_da_atteso": mean_mu[k, scanned_idx] - mu_scan,
                "rapporto_larghezza_su_unc": std_mu[k, scanned_idx] / analytic_unc[scanned_idx],
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "riassunto_linearita.csv"), index=False)
    print(summary.round(4).to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for channel in CHANNELS:
        try:
            validate_linearity_for_channel(channel)
        except FileNotFoundError as exc:
            print(
                f"\n[SALTATO] canale '{channel.display_label()}': file non trovato "
                f"({exc}). Aggiorna i path in CHANNELS (template_fit_multicanale.py)."
            )
    print(f"\nRisultati salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()