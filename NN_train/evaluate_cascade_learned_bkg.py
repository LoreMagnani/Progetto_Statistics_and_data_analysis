import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from multiclassifier_progressivo import plot_conf_matrix, scale_sample_weight_to_luminosity


def _cascaded_prediction_learned_bkg(
    y_prob_sb: np.ndarray,
    y_prob_multi_6class: np.ndarray,
    cascade_threshold: float,
    learned_bkg_idx: int,
    external_bkg_idx: int,
) -> np.ndarray:
    pred_6 = np.argmax(y_prob_multi_6class, axis=1)
    passes_sb = y_prob_sb.flatten() >= cascade_threshold
    passes_multi = pred_6 != learned_bkg_idx
    return np.where(passes_sb & passes_multi, pred_6, external_bkg_idx)

def _weighted_efficiency_purity_per_channel(
    y_true_cascaded: np.ndarray,
    y_pred_cascaded: np.ndarray,
    class_names: Sequence[str],
    sample_weight: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(class_names):
        true_mask = y_true_cascaded == idx
        pred_mask = y_pred_cascaded == idx
        true_w = sample_weight[true_mask].sum()
        pred_w = sample_weight[pred_mask].sum()
        correct_w = sample_weight[true_mask & pred_mask].sum()
        efficiency = correct_w / true_w if true_w > 0 else np.nan
        purity = correct_w / pred_w if pred_w > 0 else np.nan
        rows.append({"canale": name, "efficienza": efficiency, "purezza": purity})
    return pd.DataFrame(rows)

def evaluate_cascade_with_learned_bkg_class(
    model_sb,
    model_multi_6class,
    X_test_sb: np.ndarray,
    X_test_multi: np.ndarray,
    y_test: Dict[str, np.ndarray],
    w_test_raw: Dict[str, np.ndarray],
    class_names_6: Sequence[str],
    total_events: int,
    output_dir: str,
    batch_size: int,
    cascade_threshold: float = 0.85,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    learned_bkg_idx = len(class_names_6) - 1 
    external_bkg_idx = learned_bkg_idx  
    class_names_cascade = list(class_names_6)  

    y_prob_sb = model_sb.predict(X_test_sb, batch_size=batch_size, verbose=0)
    y_prob_multi = model_multi_6class.predict(X_test_multi, batch_size=batch_size, verbose=0)

    y_true_sb = y_test["sb_head"].flatten()
    y_true_process_6 = np.argmax(y_test["process_head"], axis=1) 

    y_pred_cascaded = _cascaded_prediction_learned_bkg(
        y_prob_sb=y_prob_sb, y_prob_multi_6class=y_prob_multi,
        cascade_threshold=cascade_threshold,
        learned_bkg_idx=learned_bkg_idx, external_bkg_idx=external_bkg_idx,
    )

    w_test_physical = scale_sample_weight_to_luminosity(w_test_raw["process_head"], total_events)
    w_test_raw_proc = w_test_raw["process_head"].flatten()

    plot_conf_matrix(
        y_true=y_true_process_6, y_pred=y_pred_cascaded,
        class_names=class_names_cascade, output_dir=output_dir,
        filename="confusion_matrix_cascaded_learned_bkg.png",
        sample_weight=w_test_physical, sample_weight_raw=w_test_raw_proc,
        total_events=total_events,
        threshold_info=f"(P(S/B) >= {cascade_threshold}, classe 'Fondo' imparata da process_head)",
    )

    summary_df = _weighted_efficiency_purity_per_channel(
        y_true_process_6, y_pred_cascaded, class_names_cascade, w_test_physical,
    )
    summary_path = os.path.join(output_dir, "learned_bkg_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n6x6 con classe 'Fondo' imparata salvata in: {output_dir}")
    print(f"Riepilogo efficienza/purezza per canale: {summary_path}\n")
    print(summary_df.round(3).to_string(index=False))

    return summary_df