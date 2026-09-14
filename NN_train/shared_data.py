import dataclasses
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from multiclassifier_progressivo import (
    Config,
    load_root_to_dataframe,
    load_root_to_dataframe_per2coppie,
    run_feature_engineering,
    make_train_val_test_split,
    bootstrap_resample,
    fit_preprocessor,
    normalize_weights,
)


FEATURE_DICT = {
    "all": [
        "m", "dphi", "deta", "dr", "pt", "px", "py", "pz", "px_i", "px_j", "py_i", "py_j", "pz_i", "pz_j",
        "neutrinos_pt", "neutrinos_phi", "neutrinos_eta", "eta", "phi",
        "pt_i", "pt_j", "phi_i", "phi_j", "eta_i", "eta_j", "pdg_i", "pdg_j",
    ],
    "raw": [
        "pt_i", "pt_j", "phi_i", "phi_j", "eta_i", "eta_j", "pdg_i", "pdg_j", "px_i", "px_j", "py_i", "py_j", "pz_i", "pz_j",
    ],
    "digerite": [
        "m", "dphi", "deta", "dr", "pt", "px", "py", "pz", "pdg_i", "pdg_j",
        "neutrinos_pt", "neutrinos_phi", "neutrinos_eta", "eta", "phi",
    ],
    "4l2n": [
        "m 4l", "m recoil 4l", "dphi sys nu",
        "m_min", "m_midlow", "m_midhigh", "m_max",
        "pt_min", "pt_midlow", "pt_midhigh", "pt_max",
        "dphi_min", "dphi_midlow", "dphi_midhigh", "dphi_max",
        "deta_min", "deta_midlow", "deta_midhigh", "deta_max",
        "dr_min", "dr_midlow", "dr_midhigh", "dr_max",
        "eta_min", "eta_midlow", "eta_midhigh", "eta_max",
        "phi_min", "phi_midlow", "phi_midhigh", "phi_max",
        "mT min nu", "mT midlow nu", "mT midhigh nu", "mT max nu",
        "is_os_sf_min", "is_os_sf_midlow", "is_os_sf_midhigh", "is_os_sf_max",
        "phi_lep1", "phi_lep2", "phi_lep3", "phi_lep4",
        "eta_lep1", "eta_lep2", "eta_lep3", "eta_lep4",
        "pt_lep1", "pt_lep2", "pt_lep3", "pt_lep4",
        "pdg_lep1", "pdg_lep2", "pdg_lep3", "pdg_lep4",
        "neutrinos_pt", "neutrinos_eta", "neutrinos_phi", "neutrinos_m",
    ],
}


_SPLIT_RELEVANT_FIELDS = (
    "input_files", "tree_name", "label_branch", "weight_branch",
    "chunk_size", "max_entries", "test_size", "val_size", "random_state",
)


def _coerce_override_value(key: str, value):
    if value is None:
        return None

    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in {"none", "null"}:
            return None
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"

        if key in {"hidden_layers", "hidden_layers_sb", "hidden_layers_multi"}:
            if raw == "":
                return []
            cleaned = raw.strip("[]")
            if cleaned == "":
                return []
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            return [int(p) for p in parts]

        if raw == "":
            return raw

        try:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    return value

def _build_run_config(base_config: Config, overrides: dict) -> Config:
    normalized = {key: _coerce_override_value(key, value) for key, value in overrides.items()}
    return dataclasses.replace(base_config, **normalized)

@dataclasses.dataclass
class SharedSplit:
    df_raw: pd.DataFrame         
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    y_train: np.ndarray          
    y_val: np.ndarray
    y_test: np.ndarray
    w_train: np.ndarray        
    w_val: np.ndarray
    w_test: np.ndarray
    total_events: int
    label_branch: str
    weight_branch: str

def load_and_split_shared(
    base_config: Config,
    sb_overrides: dict,
    multi_overrides: dict,
) -> SharedSplit:
    sb_config = _build_run_config(base_config, sb_overrides)
    multi_config = _build_run_config(base_config, multi_overrides)

    mismatched = [
        f for f in _SPLIT_RELEVANT_FIELDS
        if getattr(sb_config, f) != getattr(multi_config, f)
    ]
    if mismatched:
        raise ValueError(
            "SB_OVERRIDES e MULTI_OVERRIDES hanno valori diversi per campi che "
            "determinano lo split condiviso (deve essere lo stesso per entrambe "
            f"le reti): {mismatched}. Allinea questi valori tra i due dizionari "
            "di override prima di procedere."
        )

    sb_branches = FEATURE_DICT[sb_config.feature_set]
    multi_branches = FEATURE_DICT[multi_config.feature_set]
    union_branches = list(dict.fromkeys(list(sb_branches) + list(multi_branches)))

    if len(sb_config.input_files) == 2:
        df_raw = load_root_to_dataframe_per2coppie(
            file_coppia1=sb_config.input_files[0],
            file_coppia2=sb_config.input_files[1],
            tree_name=sb_config.tree_name,
            feature_branches=union_branches,
            label_branch=sb_config.label_branch,
            weight_branch=sb_config.weight_branch,
            chunk_size=sb_config.chunk_size,
            max_entries=sb_config.max_entries,
        )
    else:
        df_raw = load_root_to_dataframe(
            root_files=sb_config.input_files,
            tree_name=sb_config.tree_name,
            feature_branches=union_branches,
            label_branch=sb_config.label_branch,
            weight_branch=sb_config.weight_branch,
            chunk_size=sb_config.chunk_size,
            max_entries=sb_config.max_entries,
        )

    if df_raw.empty:
        raise ValueError("Il DataFrame è vuoto dopo il caricamento dei file ROOT.")

    total_events = len(df_raw)
    y_full = df_raw[sb_config.label_branch].to_numpy()
    w_full = df_raw[sb_config.weight_branch].to_numpy(dtype=np.float64)
    idx_full = np.arange(total_events).reshape(-1, 1)

    idx_train, idx_val, idx_test, y_train, y_val, y_test, w_train, w_val, w_test = make_train_val_test_split(
        X=idx_full,
        y=y_full,
        w=w_full,
        test_size=sb_config.test_size,
        val_size=sb_config.val_size,
        random_state=sb_config.random_state,
        stratify=True,
    )

    return SharedSplit(
        df_raw=df_raw,
        idx_train=idx_train.flatten(),
        idx_val=idx_val.flatten(),
        idx_test=idx_test.flatten(),
        y_train=y_train, y_val=y_val, y_test=y_test,
        w_train=w_train, w_val=w_val, w_test=w_test,
        total_events=total_events,
        label_branch=sb_config.label_branch,
        weight_branch=sb_config.weight_branch,
    )

def _get_w_dict(w_raw: np.ndarray, y_dict: Dict[str, np.ndarray], config: Config) -> Dict[str, np.ndarray]:
    sb_labels = y_dict["sb_head"].flatten()
    is_signal = sb_labels == 1
    process_labels = np.argmax(y_dict["process_head"], axis=1)

    if config.reweight_for_sb_head:
        w_sb_pre = w_raw.copy()
        if np.any(is_signal):
            w_sb_pre[is_signal] = normalize_weights(
                w_raw[is_signal], y=process_labels[is_signal], mode="per_class_sublinear",
                alpha=config.norm_pre_weight_sb_alpha,
            )
        w_sb = normalize_weights(w_sb_pre, y=sb_labels, mode=config.norm_weights_sb, alpha=config.norm_weight_sb_alpha)
    else:
        w_sb = normalize_weights(w_raw, y=sb_labels, mode=config.norm_weights_sb, alpha=config.norm_weight_sb_alpha)

    w_proc = w_raw.copy()
    w_proc[~is_signal] = 0.0
    w_proc = normalize_weights(w_proc, y=process_labels, mode=config.norm_weight, alpha=config.norm_weight_sb_alpha)

    return {"sb_head": w_sb.astype(np.float32), "process_head": w_proc.astype(np.float32)}

def _get_w_raw_dict(w_raw: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    w_sb = w_raw.astype(np.float32)
    w_proc = w_raw.copy()
    bg_mask = y_dict["sb_head"].flatten() == 0
    w_proc[bg_mask] = 0.0
    return {"sb_head": w_sb, "process_head": w_proc.astype(np.float32)}

PreparedData = Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
    object,
]

def prepare_for_network(
    shared: SharedSplit,
    run_config: Config,
) -> Tuple[PreparedData, List[str]]:
    raw_branches = FEATURE_DICT[run_config.feature_set]

    available_cols = list(shared.df_raw.columns)
    try:
        n_inputs = len(run_config.input_files)
    except Exception:
        n_inputs = 1

    resolved = []
    for name in raw_branches:
        if n_inputs == 2:
            s1 = f"{name}_coppia1"
            s2 = f"{name}_coppia2"
            if s1 in available_cols and s2 in available_cols:
                resolved.extend([s1, s2])
                continue
            if name in available_cols:
                resolved.append(name)
                continue
            if s1 in available_cols:
                resolved.append(s1)
                continue
            if s2 in available_cols:
                resolved.append(s2)
                continue
            resolved.append(name)
        else:
            if name in available_cols:
                resolved.append(name)
            else:
                resolved.append(name)

    cols_needed = list(dict.fromkeys(resolved + [shared.label_branch, shared.weight_branch]))

    missing = [c for c in cols_needed if c not in available_cols]
    if missing:
        preview = available_cols[:50]
        raise KeyError(
            f"Missing columns required for feature_set '{run_config.feature_set}': {missing}. "
            f"Available columns (first 50): {preview}{'...' if len(available_cols) > 50 else ''}. "
            "Check FEATURE_DICT, the input ROOT->DataFrame loader, or the feature engineering step for renamed/removed columns."
        )

    df_net = shared.df_raw.loc[:, cols_needed]
    df_net = run_feature_engineering(df_net)

    feature_columns = [
        c for c in df_net.columns
        if c not in (shared.label_branch, shared.weight_branch)
    ]
    X_full = df_net.loc[:, feature_columns].to_numpy(dtype=np.float32)

    X_train = X_full[shared.idx_train]
    X_val = X_full[shared.idx_val]
    X_test = X_full[shared.idx_test]
    y_train, y_val, y_test = shared.y_train, shared.y_val, shared.y_test
    w_train, w_val, w_test = shared.w_train, shared.w_val, shared.w_test

    if run_config.bootstrap:
        X_train, y_train, w_train = bootstrap_resample(
            X_train, y_train, w_train, random_state=run_config.random_state
        )

    preproc = fit_preprocessor(X_train, y_train)

    X_train_s = preproc.transform_x(X_train)
    X_val_s = preproc.transform_x(X_val)
    X_test_s = preproc.transform_x(X_test)

    y_train_dict = preproc.transform_y(y_train)
    y_val_dict = preproc.transform_y(y_val)
    y_test_dict = preproc.transform_y(y_test)

    w_train_dict = _get_w_dict(w_train, y_train_dict, run_config)
    w_val_dict = _get_w_dict(w_val, y_val_dict, run_config)
    w_test_dict = _get_w_dict(w_test, y_test_dict, run_config)

    w_train_raw_dict = _get_w_raw_dict(w_train, y_train_dict)
    w_val_raw_dict = _get_w_raw_dict(w_val, y_val_dict)
    w_test_raw_dict = _get_w_raw_dict(w_test, y_test_dict)

    data: PreparedData = (
        X_train_s, X_val_s, X_test_s,
        y_train_dict, y_val_dict, y_test_dict,
        w_train_dict, w_val_dict, w_test_dict,
        w_train_raw_dict, w_val_raw_dict, w_test_raw_dict,
        preproc,
    )

    print("reweight_for_sb_head: ", run_config.reweight_for_sb_head)
    print("config: ", run_config.norm_weights_sb)
    print("Rapporto peso max/min: ", (w_train_dict["sb_head"].max() / w_train_dict["sb_head"].min()))
    print("Pesi raw per binario: ", w_train_raw_dict["sb_head"])
    print("Pesi rinormalizzati per binario: ", w_train_dict["sb_head"])
    sig_w = w_train_dict["sb_head"][y_train_dict["sb_head"].flatten() == 1]
    bkg_w = w_train_dict["sb_head"][y_train_dict["sb_head"].flatten() == 0]
    print("Rapporto max/min DENTRO il segnale (governato da norm_pre_weight_sb_alpha):", sig_w.max() / sig_w.min())
    print("Rapporto segnale-tot / fondo-tot (governato da norm_weight_sb_alpha):", sig_w.sum() / bkg_w.sum())


    return data, feature_columns


