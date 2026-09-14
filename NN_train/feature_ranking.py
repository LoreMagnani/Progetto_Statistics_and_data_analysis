from typing import Dict, Sequence, Callable
import matplotlib.pyplot as plt

import os
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rank_features_all_heads(
    X_test: np.ndarray,
    feature_names: Sequence[str],
    y_test: Dict[str, np.ndarray],
    w_test_physical: Dict[str, np.ndarray],
    class_names: Sequence[str],
) -> pd.DataFrame:
    feature_names = list(feature_names)
    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            f"X_test ha {X_test.shape[1]} colonne ma feature_names ne elenca "
            f"{len(feature_names)} - probabilmente feature_names non e' "
            f"quello effettivamente usato per costruire X_test."
        )

    def _univariate_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> list:
        aucs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            if finite.sum() < 2 or len(np.unique(y[finite])) < 2:
                aucs.append(np.nan)
                continue
            try:
                aucs.append(roc_auc_score(y[finite], col[finite], sample_weight=w[finite]))
            except ValueError:
                aucs.append(np.nan)  # es. colonna costante
        return aucs

    results: Dict[str, list] = {"feature": feature_names}

    # sb_head: segnale (tutti i canali insieme) vs fondo
    y_sb = y_test["sb_head"].flatten()
    w_sb = w_test_physical["sb_head"].flatten()
    results["sb_head_auc"] = _univariate_auc(X_test, y_sb, w_sb)

    # process_head, one-vs-rest per canale, solo veri eventi di segnale
    is_true_signal = y_sb == 1
    y_proc_int = np.argmax(y_test["process_head"], axis=1)[is_true_signal]
    w_proc = w_test_physical["process_head"].flatten()[is_true_signal]
    X_signal = X_test[is_true_signal]

    for c, name in enumerate(class_names):
        y_c = (y_proc_int == c).astype(int)
        results[f"{name}_auc"] = _univariate_auc(X_signal, y_c, w_proc)

    df = pd.DataFrame(results)
    auc_cols = [c for c in df.columns if c.endswith("_auc")]
    df["max_discriminating_power"] = df[auc_cols].apply(
        lambda row: np.nanmax(np.abs(row.to_numpy(dtype=float) - 0.5)), axis=1
    )
    df = df.sort_values("max_discriminating_power", ascending=False).reset_index(drop=True)
    return df

def rank_features_shap(
    predict_fn: Callable,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    output_dir: str,
    target_name: str = "model",
    bg_samples: int = 150,
    eval_samples: int = 500
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    
    # Sottocampionamento per calcolo (SHAP su reti neurali è intensivo)
    bg_indices = rng.choice(X_test.shape[0], min(bg_samples, X_test.shape[0]), replace=False)
    X_bg = X_test[bg_indices]
    
    eval_indices = rng.choice(X_test.shape[0], min(eval_samples, X_test.shape[0]), replace=False)
    X_eval = X_test[eval_indices]
    
    # Riassunto del background tramite k-means per scalabilità
    X_bg_summary = shap.kmeans(X_bg, min(15, bg_samples))
    explainer = shap.KernelExplainer(predict_fn, X_bg_summary)
    
    # Calcolo SHAP values
    shap_values = explainer.shap_values(X_eval, silent=True)
    
    # Gestione delle dimensionalità per output binari o multiclasse
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        shap_values_to_plot = shap_values
        if len(shap_values_to_plot.shape) == 3:
            shap_values_to_plot = shap_values_to_plot[:, :, 0]
        mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
        
    df_shap = pd.DataFrame({
        "feature": feature_names,
        f"mean_abs_shap_{target_name}": mean_abs_shap
    }).sort_values(f"mean_abs_shap_{target_name}", ascending=False).reset_index(drop=True)
    
    # Plot globale a densità di punti (Summary Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_eval, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot - {target_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_{target_name}.png"), dpi=200)
    plt.close()
    
    return df_shap
