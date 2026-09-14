from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scan_sb_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight_physical: np.ndarray,
    n_thresholds: int = 300,
    current_threshold: Optional[float] = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    w = np.asarray(sample_weight_physical).flatten()

    is_sig = y_true == 1
    is_bkg = ~is_sig
    w_sig_tot = w[is_sig].sum()
    w_bkg_tot = w[is_bkg].sum()
    if w_sig_tot <= 0 or w_bkg_tot <= 0:
        raise ValueError("Il test set non contiene sia segnale che fondo con peso fisico > 0.")

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    for thr in thresholds:
        pred_sig = y_prob >= thr
        tp_w = w[is_sig & pred_sig].sum()
        fp_w = w[is_bkg & pred_sig].sum()
        sig_eff = tp_w / w_sig_tot
        bkg_rejection = 1.0 - (fp_w / w_bkg_tot)
        purity = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else np.nan
        rows.append({
            "threshold": thr,
            "signal_efficiency": sig_eff,
            "bkg_rejection": bkg_rejection,
            "purity": purity,
            "n_selected_signal_weighted": tp_w,
            "n_selected_bkg_weighted": fp_w,
        })

    df = pd.DataFrame(rows)

    if current_threshold is not None:
        idx_current = (df["threshold"] - current_threshold).abs().idxmin()
        current_bkg_rejection = df.loc[idx_current, "bkg_rejection"]
        df["feasible_min_current_bkg_rejection"] = df["bkg_rejection"] >= current_bkg_rejection

        print(f"Soglia attuale ({current_threshold:.3f}): "
              f"efficienza segnale={df.loc[idx_current, 'signal_efficiency']:.3f}, "
              f"rigetto fondo={current_bkg_rejection:.3f}, "
              f"purezza={df.loc[idx_current, 'purity']:.3f}")

        feasible = df[df["feasible_min_current_bkg_rejection"]]
        if not feasible.empty:
            best_idx = feasible["signal_efficiency"].idxmax()
            best = feasible.loc[best_idx]
            print(f"\nMiglior soglia con rigetto fondo >= a quello attuale: "
                  f"soglia={best['threshold']:.3f}, "
                  f"efficienza segnale={best['signal_efficiency']:.3f}, "
                  f"rigetto fondo={best['bkg_rejection']:.3f}, "
                  f"purezza={best['purity']:.3f}")
            gain = best['signal_efficiency'] - df.loc[idx_current, 'signal_efficiency']
            if gain > 1e-4:
                print(f"-> guadagno di efficienza a parità (o miglioramento) di rigetto fondo: +{gain:.3f}")
            else:
                print("-> nessun guadagno disponibile: la soglia attuale è già Pareto-ottimale "
                      "sulla curva corrente. Per fare meglio serve migliorare il training, "
                      "non la soglia.")

    return df

def plot_sb_threshold_scan(
    df: pd.DataFrame,
    output_path: str,
    current_threshold: Optional[float] = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(df["threshold"], df["signal_efficiency"], label="Efficienza segnale", color="tab:blue")
    ax1.plot(df["threshold"], df["bkg_rejection"], label="Rigetto fondo", color="tab:orange")
    ax1.plot(df["threshold"], df["purity"], label="Purezza", color="tab:green", linestyle="--")
    ax1.set_xlabel("Soglia P(S/B)")
    ax1.set_ylabel("Valore")
    ax1.set_ylim(0.0, 1.02)

    if current_threshold is not None:
        ax1.axvline(current_threshold, color="gray", linestyle=":", label=f"Soglia attuale ({current_threshold:.2f})")

    ax1.legend(loc="lower left")
    ax1.set_title("Scan soglia sb_head (pesi fisici)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
