import dataclasses
from typing import List, Optional, Tuple

import numpy as np
from tensorflow import keras

from multiclassifier_progressivo import (
    Config,
    Preprocessor,
    PROCESS_MAP,
    SB_MAP,
    PROCESS_CLASS_NAMES,
    run_feature_engineering,
    fit_preprocessor,
    normalize_weights,
    bootstrap_resample,
    plot_history,
)

from shared_data import FEATURE_DICT, SharedSplit, PreparedData


def _is_signal_and_process_idx(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    is_signal = np.array([SB_MAP.get(int(v), 0) for v in y_raw]) == 1
    process_idx = np.array([PROCESS_MAP.get(int(v), -1) for v in y_raw])
    return is_signal, process_idx

def _one_hot(labels_1d: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(labels_1d), n_classes), dtype=np.float32)
    oh[np.arange(len(labels_1d)), labels_1d] = 1.0
    return oh

def _select_hard_background(
    is_signal: np.ndarray,
    sb_scores: np.ndarray,
    sample_size: Optional[int],
    process_idx: np.ndarray,
    n_signal_classes: int,
    easy_bkg_fraction: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, int]:
    signal_rows = np.where(is_signal)[0]
    bkg_rows = np.where(~is_signal)[0]

    if sample_size is None:
        counts = np.array([np.sum(process_idx[signal_rows] == c) for c in range(n_signal_classes)])
        sample_size = int(counts.max()) if counts.size else 0

    if len(bkg_rows) > sample_size:
        order = np.argsort(-sb_scores[bkg_rows])  # score decrescente: il fondo piu' "segnale-simile" per primo

        n_easy = int(round(sample_size * easy_bkg_fraction))
        n_hard = sample_size - n_easy

        hard_rows = bkg_rows[order[:n_hard]]
        remaining_rows = bkg_rows[order[n_hard:]]  # copre tutto il resto dello spettro, incluso il fondo piu' ovvio

        if n_easy > 0 and len(remaining_rows) > 0:
            rng = np.random.default_rng(random_state)
            n_easy_actual = min(n_easy, len(remaining_rows))
            easy_rows = rng.choice(remaining_rows, size=n_easy_actual, replace=False)
        else:
            easy_rows = np.array([], dtype=int)

        selected_bkg_rows = np.sort(np.concatenate([hard_rows, easy_rows]))
    else:
        selected_bkg_rows = bkg_rows

    keep_rows = np.sort(np.concatenate([signal_rows, selected_bkg_rows]))
    return keep_rows, len(selected_bkg_rows)

def prepare_multiclass_with_bkg_class(
    shared: SharedSplit,
    multi_run_config: Config,
    model_sb: keras.Model,
    sb_preproc: Preprocessor,
    sb_feature_set: str,
    sb_batch_size: int,
    bkg_class_name: str = "Fondo",
    bkg_sample_size: Optional[int] = None,
) -> Tuple[PreparedData, List[str], List[str]]:
    n_signal_classes = len(PROCESS_CLASS_NAMES)
    bkg_class_idx = n_signal_classes
    class_names_6 = list(PROCESS_CLASS_NAMES) + [bkg_class_name]

    raw_branches = FEATURE_DICT[multi_run_config.feature_set]
    cols_needed = list(dict.fromkeys(
        list(raw_branches) + [shared.label_branch, shared.weight_branch]
    ))
    df_net = shared.df_raw.loc[:, cols_needed]
    df_net = run_feature_engineering(df_net)
    feature_columns = [c for c in df_net.columns if c not in (shared.label_branch, shared.weight_branch)]
    X_full = df_net.loc[:, feature_columns].to_numpy(dtype=np.float32)

    X_train_raw = X_full[shared.idx_train]
    X_val_raw = X_full[shared.idx_val]
    X_test_raw = X_full[shared.idx_test]
    y_train, y_val, y_test = shared.y_train, shared.y_val, shared.y_test
    w_train, w_val, w_test = shared.w_train, shared.w_val, shared.w_test

    sb_raw_branches = FEATURE_DICT[sb_feature_set]
    sb_cols_needed = list(dict.fromkeys(
        list(sb_raw_branches) + [shared.label_branch, shared.weight_branch]
    ))
    df_sb = shared.df_raw.loc[:, sb_cols_needed]
    df_sb = run_feature_engineering(df_sb)
    sb_feature_columns = [c for c in df_sb.columns if c not in (shared.label_branch, shared.weight_branch)]
    X_sb_full = df_sb.loc[:, sb_feature_columns].to_numpy(dtype=np.float32)

    X_sb_train_s = sb_preproc.transform_x(X_sb_full[shared.idx_train])
    X_sb_val_s = sb_preproc.transform_x(X_sb_full[shared.idx_val])

    sb_scores_train = model_sb.predict(X_sb_train_s, batch_size=sb_batch_size, verbose=0).flatten()
    sb_scores_val = model_sb.predict(X_sb_val_s, batch_size=sb_batch_size, verbose=0).flatten()

    is_signal_train, process_idx_train = _is_signal_and_process_idx(y_train)
    is_signal_val, process_idx_val = _is_signal_and_process_idx(y_val)
    is_signal_test, process_idx_test = _is_signal_and_process_idx(y_test)

    keep_train, n_bkg_train = _select_hard_background(
        is_signal_train, sb_scores_train, bkg_sample_size, process_idx_train, n_signal_classes,
        easy_bkg_fraction=multi_run_config.easy_bkg_fraction,
        random_state=multi_run_config.random_state,
    )
    keep_val, n_bkg_val = _select_hard_background(
        is_signal_val, sb_scores_val, bkg_sample_size, process_idx_val, n_signal_classes,
        easy_bkg_fraction=multi_run_config.easy_bkg_fraction,
        random_state=multi_run_config.random_state,
    )
    print(f"[bkg_class] fondo 'difficile' incluso: {n_bkg_train} eventi train, {n_bkg_val} eventi val")

    labels_train_full = np.where(is_signal_train, process_idx_train, bkg_class_idx)
    labels_val_full = np.where(is_signal_val, process_idx_val, bkg_class_idx)
    labels_test_full = np.where(is_signal_test, process_idx_test, bkg_class_idx)

    X_train_kept = X_train_raw[keep_train]
    X_val_kept = X_val_raw[keep_val]
    w_train_kept = w_train[keep_train]
    w_val_kept = w_val[keep_val]
    labels_train_kept = labels_train_full[keep_train]
    labels_val_kept = labels_val_full[keep_val]



    # -------------------------------------------------------------------
    # 4. Bootstrap e Scaler fittato sul set FINALE
    # -------------------------------------------------------------------
    if multi_run_config.bootstrap:
        X_train_kept, labels_train_kept, w_train_kept = bootstrap_resample(
            X_train_kept, 
            labels_train_kept, 
            w_train_kept, 
            random_state=multi_run_config.random_state
        )

    preproc = fit_preprocessor(X_train_kept, labels_train_kept)
    
    X_train_s = preproc.transform_x(X_train_kept)
    X_val_s = preproc.transform_x(X_val_kept)
    X_test_s = preproc.transform_x(X_test_raw)

    # -------------------------------------------------------------------
    # 5. One-hot a 6 classi + pesi e ricostruzione dell'allineamento sb_head
    # -------------------------------------------------------------------
    y_train_process = _one_hot(labels_train_kept, n_signal_classes + 1)
    y_val_process = _one_hot(labels_val_kept, n_signal_classes + 1)
    y_test_process = _one_hot(labels_test_full, n_signal_classes + 1)
    is_signal_train_kept = (labels_train_kept != bkg_class_idx)

    w_train_process = normalize_weights(
        w_train_kept, y=labels_train_kept,
        mode=multi_run_config.norm_weight, alpha=multi_run_config.norm_weight_sb_alpha,
    ).astype(np.float32)
    
    w_val_process = normalize_weights(
        w_val_kept, y=labels_val_kept,
        mode=multi_run_config.norm_weight, alpha=multi_run_config.norm_weight_sb_alpha,
    ).astype(np.float32)

    w_test_process_raw = w_test.astype(np.float32)

    y_train_dict = {
        "sb_head": is_signal_train_kept.astype(np.float32).reshape(-1, 1),
        "process_head": y_train_process,
    }
    y_val_dict = {
        "sb_head": is_signal_val[keep_val].astype(np.float32).reshape(-1, 1),
        "process_head": y_val_process,
    }
    y_test_dict = {
        "sb_head": is_signal_test.astype(np.float32).reshape(-1, 1),
        "process_head": y_test_process,
    }

    w_train_dict = {"sb_head": np.ones(len(X_train_kept), dtype=np.float32), "process_head": w_train_process}
    w_val_dict = {"sb_head": np.ones(len(keep_val), dtype=np.float32), "process_head": w_val_process}
    w_test_dict = {"sb_head": np.ones(len(y_test), dtype=np.float32), "process_head": w_test_process_raw}

    w_train_raw_dict = {"sb_head": w_train_kept.astype(np.float32), "process_head": w_train_kept.astype(np.float32)}
    w_val_raw_dict = {"sb_head": w_val_kept.astype(np.float32), "process_head": w_val_kept.astype(np.float32)}
    w_test_raw_dict = {"sb_head": w_test.astype(np.float32), "process_head": w_test.astype(np.float32)}

    data: PreparedData = (
        X_train_s, X_val_s, X_test_s,
        y_train_dict, y_val_dict, y_test_dict,
        w_train_dict, w_val_dict, w_test_dict,
        w_train_raw_dict, w_val_raw_dict, w_test_raw_dict,
        preproc,
    )
    return data, feature_columns, class_names_6