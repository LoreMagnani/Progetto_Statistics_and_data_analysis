import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from multiclassifier_progressivo import (
    plot_conf_matrix,
    scale_sample_weight_to_luminosity,
    PROCESS_CLASS_NAMES,
)


def _cascaded_prediction(
    y_prob_sb: np.ndarray,
    y_prob_multi: np.ndarray,
    cascade_threshold: float,
    confidence_threshold: float,
    bkg_class_idx: int,
) -> np.ndarray:
    pred_process_int = np.argmax(y_prob_multi, axis=1)
    max_multi_prob = y_prob_multi.max(axis=1)

    passes_sb = y_prob_sb.flatten() >= cascade_threshold
    passes_confidence = max_multi_prob >= confidence_threshold

    return np.where(passes_sb & passes_confidence, pred_process_int, bkg_class_idx)

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

def evaluate_cascade_with_confidence_filter(
    y_prob_sb: np.ndarray,
    y_prob_multi: np.ndarray,
    y_test: Dict[str, np.ndarray],
    w_test_raw: Dict[str, np.ndarray],
    total_events: int,
    output_dir: str,
    cascade_threshold: float = 0.85,
    confidence_thresholds: Sequence[float] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    y_true_sb = y_test["sb_head"].flatten()
    y_true_process_int = np.argmax(y_test["process_head"], axis=1)
    class_names = list(PROCESS_CLASS_NAMES)
    bkg_class_idx = len(class_names)
    all_names = class_names + ["Fondo Totale"]

    y_true_cascaded = np.where(y_true_sb == 1, y_true_process_int, bkg_class_idx)

    w_test_physical_sb = scale_sample_weight_to_luminosity(w_test_raw["sb_head"], total_events)
    w_test_raw_sb = w_test_raw["sb_head"].flatten()

    summary_rows = []
    for conf_thr in confidence_thresholds:
        y_pred_cascaded = _cascaded_prediction(
            y_prob_sb=y_prob_sb, y_prob_multi=y_prob_multi,
            cascade_threshold=cascade_threshold, confidence_threshold=conf_thr,
            bkg_class_idx=bkg_class_idx,
        )

        safe_thr = str(conf_thr).replace(".", "p")
        plot_conf_matrix(
            y_true=y_true_cascaded, y_pred=y_pred_cascaded,
            class_names=all_names, output_dir=output_dir,
            filename=f"confusion_matrix_cascaded_6x6_conf{safe_thr}.png",
            sample_weight=w_test_physical_sb, sample_weight_raw=w_test_raw_sb,
            total_events=total_events,
            threshold_info=f"(P(S/B) >= {cascade_threshold}, confidenza multiclasse >= {conf_thr})",
        )

        per_channel = _weighted_efficiency_purity_per_channel(
            y_true_cascaded, y_pred_cascaded, class_names, w_test_physical_sb,
        )
        per_channel["confidence_threshold"] = conf_thr
        summary_rows.append(per_channel)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df = summary_df.pivot(index="canale", columns="confidence_threshold", values=["efficienza", "purezza"])
    summary_df = summary_df.reindex(class_names)
    summary_path = os.path.join(output_dir, "confidence_filter_summary.csv")
    summary_df.to_csv(summary_path)

    print(f"\n6x6 generate per {len(confidence_thresholds)} soglie di confidenza in: {output_dir}")
    print(f"Tabella riassuntiva efficienza/purezza per canale: {summary_path}\n")
    print(summary_df.round(3).to_string())

    return summary_df
