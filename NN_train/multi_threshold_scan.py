import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from sb_threshold_scan import scan_sb_threshold, plot_sb_threshold_scan


def scan_multiclass_thresholds(
    y_true_int: np.ndarray,
    y_prob: np.ndarray,
    sample_weight_physical: np.ndarray,
    class_names: Sequence[str],
    n_thresholds: int = 300,
) -> Dict[str, pd.DataFrame]:
    y_true_int = np.asarray(y_true_int).flatten()
    class_names = list(class_names)
    n_classes = len(class_names)
    if y_prob.shape[1] != n_classes:
        raise ValueError(
            f"y_prob ha {y_prob.shape[1]} colonne ma class_names ne elenca {n_classes}."
        )

    results: Dict[str, pd.DataFrame] = {}
    print("Scan one-vs-rest per canale (pesi fisici):\n")
    for c, name in enumerate(class_names):
        y_true_c = (y_true_int == c).astype(int)
        y_prob_c = y_prob[:, c]
        print(f"--- {name} ---")
        df = scan_sb_threshold(
            y_true_c, y_prob_c, sample_weight_physical,
            n_thresholds=n_thresholds, current_threshold=None,
        )
        results[name] = df
        print()

    return results

def plot_multiclass_threshold_scan(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, df in results.items():
        safe_name = name.replace("+", "_plus_").replace(" ", "_")
        plot_sb_threshold_scan(
            df,
            output_path=os.path.join(output_dir, f"threshold_scan_{safe_name}.png"),
        )
