import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BKG_LABEL = "Fondo Totale"

# =====================================================================
# CONFIGURAZIONE (un canale = un dataset/classificatore già valutato
# con cascade_inference_eval.py)
# =====================================================================
@dataclass
class ChannelConfig:
    name: str  #
    signal_names: List[str]
    template_events_parquet: str
    composition_events_parquet: str
    label: Optional[str] = None  
    def display_label(self) -> str:
        return self.label or self.name

    @property
    def class_names(self) -> List[str]:
        return list(self.signal_names) + [BKG_LABEL]

    @property
    def n_signal(self) -> int:
        return len(self.signal_names)

CHANNELS: List[ChannelConfig] = [
    ChannelConfig(
        name="4l2v",
        label="4l2v",
        signal_names=["WW+H", "ZZ+H", "ZZ", "WZ", "WW"],
        template_events_parquet="template_eventi.parquet",
        composition_events_parquet="composizione_eventi.parquet",
    ),
]

OUTPUT_DIR = "output_template_fit"


# =====================================================================
# COSTRUZIONE MATRICI (conteggi pesati + varianze) — generica, non
# dipende dal numero di segnali
# =====================================================================
def build_counts_and_variance(
    df_events: pd.DataFrame, class_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    df = df_events.copy()
    df["weight_sq"] = df["weight"].to_numpy(dtype=np.float64) ** 2

    counts = pd.crosstab(
        df["y_true_name"], df["y_pred_name"], values=df["weight"], aggfunc="sum", dropna=False
    ).reindex(index=class_names, columns=class_names, fill_value=0.0).fillna(0.0).to_numpy()

    variance = pd.crosstab(
        df["y_true_name"], df["y_pred_name"], values=df["weight_sq"], aggfunc="sum", dropna=False
    ).reindex(index=class_names, columns=class_names, fill_value=0.0).fillna(0.0).to_numpy()

    return counts, variance

def check_template_population(counts_template: np.ndarray, channel: ChannelConfig) -> None:
    total = counts_template.sum()
    for i, name in enumerate(channel.signal_names):
        row_total = counts_template[i, :].sum()
        frac = row_total / total if total > 0 else 0.0
        if frac < 1e-6:
            print(
                f"  [ATTENZIONE] canale '{channel.display_label()}': il template di "
                f"'{name}' e' sostanzialmente vuoto ({row_total:.3g} eventi fisici). "
                "Il suo mu non sara' vincolato dal fit: valuta se rimuoverlo da "
                "signal_names per questo canale."
            )


# =====================================================================
# FIT (minimi quadrati pesati, soluzione in forma chiusa) — invariato
# =====================================================================
def fit_signal_strengths(
    counts_template: np.ndarray,
    variance_template: np.ndarray,
    counts_composizione: np.ndarray,
    variance_composizione: np.ndarray,
    n_signal: int,
) -> Dict[str, np.ndarray]:
    bkg_idx = n_signal

    A = counts_template[:n_signal, :].T  
    bkg_pred = counts_template[bkg_idx, :]
    bkg_var = variance_template[bkg_idx, :]

    y_obs = counts_composizione.sum(axis=0)
    y_obs_var = variance_composizione.sum(axis=0)

    y = y_obs - bkg_pred
    sigma2 = y_obs_var + bkg_var
    sigma2 = np.where(sigma2 <= 0, np.inf, sigma2)
    W = np.diag(1.0 / sigma2)

    AtW = A.T @ W
    fisher = AtW @ A
    cond_number = np.linalg.cond(fisher)
    if cond_number > 1e10:
        print(
            f"  [ATTENZIONE] matrice del fit mal condizionata (cond={cond_number:.2e}): "
            "alcuni segnali probabilmente migrano in modo troppo simile tra i canali "
            "predetti per essere distinti con questa binnizzazione."
        )

    cov = np.linalg.inv(fisher)
    mu_hat = cov @ AtW @ y
    mu_unc = np.sqrt(np.diag(cov))

    if np.all(mu_unc > 0):
        corr = cov / np.outer(mu_unc, mu_unc)
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > 0.9:
                    print(
                        f"  [ATTENZIONE] correlazione molto alta ({corr[i, j]:.2f}) tra i "
                        f"segnali di indice {i} e {j}: il fit fatica a distinguerli "
                        "(migrano in modo troppo simile nei canali predetti)."
                    )

    y_fit = A @ mu_hat + bkg_pred
    residual = y - A @ mu_hat
    chi2 = float(residual @ W @ residual)
    dof = len(y) - n_signal

    return {
        "mu": mu_hat,
        "cov": cov,
        "unc": mu_unc,
        "chi2": chi2,
        "dof": dof,
        "y_obs": y_obs,
        "y_fit": y_fit,
        "bkg_pred": bkg_pred,
        "sigma": np.sqrt(y_obs_var + bkg_var),
    }


# =====================================================================
# OUTPUT PER SINGOLO CANALE
# =====================================================================
def plot_fit_closure(
    result: Dict[str, np.ndarray], class_names: Sequence[str], output_dir: str, channel_label: str = ""
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_bins = len(class_names)
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, result["y_fit"], alpha=0.45, color="tab:blue", width=0.6,
           label="Fit: Σ μᵢ·Tᵢ + Bkg (fixed)")
    ax.errorbar(x, result["y_obs"], yerr=result["sigma"], fmt="o", color="black",
                capsize=4, label="Observed Composition")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Expected Events")
    titolo = f"Template Fit Closure — {channel_label}" if channel_label else "Template Fit Closure"
    ax.set_title(f"{titolo}\n(chi2/dof = {result['chi2']:.2f} / {result['dof']})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fit_chiusura.png"), dpi=200)
    plt.close(fig)

def run_fit_for_channel(channel: ChannelConfig) -> Dict:
    print(f"\n{'=' * 70}\n CANALE: {channel.display_label()}\n{'=' * 70}")

    df_template = pd.read_parquet(channel.template_events_parquet)
    df_composizione = pd.read_parquet(channel.composition_events_parquet)

    output_dir = os.path.join(OUTPUT_DIR, channel.name)
    os.makedirs(output_dir, exist_ok=True)

    counts_template, variance_template = build_counts_and_variance(df_template, channel.class_names)
    counts_composizione, variance_composizione = build_counts_and_variance(df_composizione, channel.class_names)

    check_template_population(counts_template, channel)

    result = fit_signal_strengths(
        counts_template, variance_template,
        counts_composizione, variance_composizione,
        n_signal=channel.n_signal,
    )

    print(f"\n--- Signal strength mu ({channel.display_label()}) ---")
    for name, mu, unc in zip(channel.signal_names, result["mu"], result["unc"]):
        print(f"  mu({name}) = {mu:.3f} +/- {unc:.3f}")
    print(f"chi2/dof = {result['chi2']:.2f} / {result['dof']}")

    unc = result["unc"]
    corr = result["cov"] / np.outer(unc, unc)
    corr_df = pd.DataFrame(corr, index=channel.signal_names, columns=channel.signal_names)
    print("Matrice di correlazione:")
    print(corr_df.round(3))

    pd.DataFrame({
        "processo": channel.signal_names,
        "mu": result["mu"],
        "incertezza": result["unc"],
    }).to_csv(os.path.join(output_dir, "signal_strengths.csv"), index=False)
    corr_df.to_csv(os.path.join(output_dir, "matrice_correlazione.csv"))

    plot_fit_closure(result, channel.class_names, output_dir, channel_label=channel.display_label())

    return {"channel": channel, "result": result}


# =====================================================================
# CONFRONTO TRA CANALI
# =====================================================================
def plot_cross_channel_compatibility(all_results: List[Dict], output_dir: str) -> None:
    channel_labels = [entry["channel"].display_label() for entry in all_results]
    channel_index = {label: i for i, label in enumerate(channel_labels)}

    per_process: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for entry in all_results:
        channel = entry["channel"]
        result = entry["result"]
        for name, mu, unc in zip(channel.signal_names, result["mu"], result["unc"]):
            per_process.setdefault(name, {})[channel.display_label()] = (mu, unc)

    shared = {name: vals for name, vals in per_process.items() if len(vals) > 1}
    if not shared:
        print("[INFO] Nessun processo condiviso tra più canali: salto il confronto incrociato.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    n_proc = len(shared)
    width = 0.7 / max(n_proc, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_proc))

    for k, (name, vals) in enumerate(shared.items()):
        xs, mus, uncs = [], [], []
        for label, (mu, unc) in vals.items():
            xs.append(channel_index[label] + (k - (n_proc - 1) / 2) * width)
            mus.append(mu)
            uncs.append(unc)
        ax.errorbar(xs, mus, yerr=uncs, fmt="o", color=colors[k], label=name, capsize=4)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(channel_labels)))
    ax.set_xticklabels(channel_labels)
    ax.set_ylabel("mu (signal strength)")
    ax.set_title("Compatibilità tra canali per i processi condivisi\n(linea tratteggiata: mu = 1, atteso SM)")
    ax.legend(title="Processo")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confronto_tra_canali.png"), dpi=200)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    for channel in CHANNELS:
        try:
            all_results.append(run_fit_for_channel(channel))
        except FileNotFoundError as exc:
            print(
                f"\n[SALTATO] canale '{channel.display_label()}': file non trovato ({exc}). "
                "Aggiorna i path in CHANNELS quando avrai prodotto quell'output."
            )

    if len(all_results) > 1:
        plot_cross_channel_compatibility(all_results, OUTPUT_DIR)
        print(f"\nConfronto tra canali salvato in: {os.path.join(OUTPUT_DIR, 'confronto_tra_canali.png')}")

    print(f"\nTutti i risultati salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()