from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import math
import dataclasses
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, Add
from tensorflow.keras.models import Sequential


PROCESS_CLASS_NAMES = ["WW+H", "ZZ+H", "ZZ", "WZ", "WW"]
LUMINOSITY = 10800000

# -----------------------------
# Configuration
# -----------------------------
def save_config_to_txt(config: Any, output_path: str) -> None:
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    if dataclasses.is_dataclass(config):
        config_dict = dataclasses.asdict(config)
    elif hasattr(config, "__dict__"):
        config_dict = vars(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError(f"Impossibile estrarre i campi dall'oggetto di tipo {type(config)}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("      CONFIGURAZIONE DELL'ESPERIMENTO    \n")
        f.write("========================================\n\n")
        
        for key in sorted(config_dict.keys()):
            val = config_dict[key]
            f.write(f"{key:<25}: {val}\n")

@dataclass
class Config:
    input_files: Sequence[str]  
    feature_set: str = "digerite"
    output_dir: str = "test_outputs"
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    hidden_layers_sb: List[int] = field(default_factory=lambda: [128, 64, 32])
    hidden_layers_multi: List[int] = field(default_factory=lambda: [128, 64, 32])
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    use_dropout: bool = False
    dropout_rate: float = 0.15
    weight_decay: float = 1e-4
    optimizer_type: str = "AdamW"
    scheduler_type: str = "ReduceLROnPlateau"
    learning_rate: float = 1e-3
    one_cycle_max_lr: float = 1e-3
    sgd_momentum: float = 0.9
    step_lr_period: int = 20
    step_lr_gamma: float = 0.1
    use_focal_loss_sb: bool = False
    use_focal_loss_multi: bool = False
    focal_gamma_sb: float = 2.0
    focal_alpha_sb: float = 0.25
    focal_gamma_multi: float = 2.0
    focal_alpha_multi: float = 0.25
    batch_size: int = 256
    bootstrap: bool = False
    norm_weight: str = "per_class"
    norm_weights_sb: str = "per_class"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    epochs: int = 100
    tree_name: str = "Events"
    label_branch: str = "kind_events"
    weight_branch: str = "Weight_weight"
    chunk_size: int = 250000
    max_entries: Optional[int] = None
    use_warmup: bool = False
    warmup_steps: int = 1000
    grad_accum_steps: int = 1
    global_clipnorm: float = 4.0
    sb_pos_weight_multiplier: float = 1.0  
    loss_weight_sb: float = 3.0
    loss_weight_multi: float = 1.0
    reweight_for_sb_head: bool = True
    use_class_weight_sb: bool = True
    train_stage: str = "both"
    pretrain_epochs: int = 50
    freeze_shared_after_pretrain: bool = True
    freeze_sb_after_pretrain: bool = True
    pretrained_weights_path: str = "pretrained_sb_weights.weights.h5"
    fine_tune_lr: float = 1e-5
    two_stage_auto: bool = False
    use_residual_in_multi: bool = False
    norm_weight_sb_alpha: float = 0.5
    norm_pre_weight_sb_alpha: float = 1
    easy_bkg_fraction: float = 0.25  

def str_to_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "y", "t"}

def parse_args_to_config() -> Config:
    parser = argparse.ArgumentParser(description="Parser per lo Scan delle configurazioni NN")

    parser.add_argument("--output_dir", type=str, default="test_outputs")

    # Mai toccati
    parser.add_argument("--tree_name", type=str, default="Events")
    parser.add_argument("--label_branch", type=str, default="kind_events")
    parser.add_argument("--weight_branch", type=str, default="Weight_weight")
    parser.add_argument("--chunk_size", type=int, default=250000)
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)

    # Bootstrap
    parser.add_argument("--bootstrap", type=str, default="False")

    # Più importanti da modificare
    parser.add_argument("--feature_set", type=str, choices=["all", "raw", "digerite", "4l2n"], default="digerite")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--hidden_layers_sb", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--hidden_layers_multi", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--scheduler_type", type=str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau", "CosineAnnealing", "OneCycle", "StepLR"])
    parser.add_argument("--norm_weight", type=str, default="per_class", choices=["sum_to_one", "mean_one", "per_class", "per_class_sublinear", "none"])
    parser.add_argument("--sb_pos_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--norm_weight_sb_alpha", type=float, default=0.5)
    parser.add_argument("--norm_pre_weight_sb_alpha", type=float, default=1)
    parser.add_argument("--norm_weights_sb", type=str, default="per_class", choices=["sum_to_one", "mean_one", "per_class", "per_class_sublinear", "none"])
    parser.add_argument("--loss_weight_sb", type=float, default=3.0)
    parser.add_argument("--loss_weight_multi", type=float, default=1.0)

    # Di base fissi se non voglio essere creativo
    parser.add_argument("--use_batch_norm", type=str, default="False")
    parser.add_argument("--use_layer_norm", type=str, default="True")
    parser.add_argument("--use_dropout", type=str, default="False")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # Parametri per scheduler
    parser.add_argument("--one_cycle_max_lr", type=float, default=1e-3)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--step_lr_period", type=int, default=20)
    parser.add_argument("--step_lr_gamma", type=float, default=0.1)

    # Focal
    parser.add_argument("--use_focal_loss_sb", type=str, default="False")
    parser.add_argument("--focal_gamma_sb", type=float, default=2.0)
    parser.add_argument("--focal_alpha_sb", type=float, default=0.25)
    parser.add_argument("--use_focal_loss_multi", type=str, default="False")
    parser.add_argument("--focal_gamma_multi", type=float, default=2.0)
    parser.add_argument("--focal_alpha_multi", type=float, default=0.25)

    # Boost vari (no danni se accesi, forse aiutano)
    parser.add_argument("--use_warmup", type=str, default="False") # Obbliga a usare cosineaAnneling
    parser.add_argument("--warmup_steps", type=int, default=1000) 
    parser.add_argument("--grad_accum_steps", type=int, default=1) # > 1 per attivarlo
    parser.add_argument("--global_clipnorm", type=float, default=4.0) # > 1 per attivarlo
    parser.add_argument("--reweight_for_sb_head", type=str, default="True")
    parser.add_argument("--use_class_weight_sb", type=str, default="False") # Annulla quello che fa il parser sopra

    # Aggiunge residui
    parser.add_argument("--use_residual_in_multi", type=str, default="False")

    # Per 2 stage
    parser.add_argument("--train_stage", type=str, default="both", choices=["both","sb_only","multi_only","two_stage"])
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--freeze_shared_after_pretrain", type=str, default="True")
    parser.add_argument("--freeze_sb_after_pretrain", type=str, default="True")
    parser.add_argument("--pretrained_weights_path", type=str, default="pretrained_sb_weights.weights.h5")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--two_stage_auto", type=str, default="False")

    args = parser.parse_args()

    hidden_layers_tmp = args.hidden_layers
    if hidden_layers_tmp == [-1]:
        hidden_layers_tmp = []
    hidden_layers_sb_tmp = args.hidden_layers_sb
    if hidden_layers_sb_tmp == [-1]:
        hidden_layers_sb_tmp = []
    hidden_layers_multi_tmp = args.hidden_layers_multi
    if hidden_layers_multi_tmp == [-1]:
        hidden_layers_multi_tmp = []

    return Config(
        feature_set=args.feature_set,
        output_dir=args.output_dir,
        hidden_layers=hidden_layers_tmp,
        hidden_layers_sb=hidden_layers_sb_tmp,
        hidden_layers_multi=hidden_layers_multi_tmp,
        use_batch_norm=str_to_bool(args.use_batch_norm),
        use_layer_norm=str_to_bool(args.use_layer_norm),
        use_dropout=str_to_bool(args.use_dropout),
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        scheduler_type=args.scheduler_type,
        learning_rate=args.learning_rate,
        one_cycle_max_lr=args.one_cycle_max_lr,
        sgd_momentum=args.sgd_momentum,
        step_lr_period=args.step_lr_period,
        step_lr_gamma=args.step_lr_gamma,
        use_focal_loss_sb=str_to_bool(args.use_focal_loss_sb),
        focal_gamma_sb=args.focal_gamma_sb,
        focal_alpha_sb=args.focal_alpha_sb,
        use_focal_loss_multi=str_to_bool(args.use_focal_loss_multi),
        focal_gamma_multi=args.focal_gamma_multi,
        focal_alpha_multi=args.focal_alpha_multi,
        batch_size=args.batch_size,
        bootstrap=str_to_bool(args.bootstrap),
        norm_weight=args.norm_weight,
        norm_weights_sb=args.norm_weights_sb,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        epochs=args.epochs,
        #input_files=["pair4lep.root", "pair4lep_other.root"],
        input_files=["4chlep2nu_300.root"],
        tree_name=args.tree_name,
        label_branch=args.label_branch,
        weight_branch=args.weight_branch,
        chunk_size=args.chunk_size,
        max_entries=args.max_entries,
        use_warmup=str_to_bool(args.use_warmup),
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        global_clipnorm=args.global_clipnorm,
        sb_pos_weight_multiplier=args.sb_pos_weight_multiplier,
        loss_weight_sb=args.loss_weight_sb,
        loss_weight_multi=args.loss_weight_multi,
        reweight_for_sb_head=str_to_bool(args.reweight_for_sb_head),
        use_class_weight_sb=str_to_bool(args.use_class_weight_sb),
        train_stage=args.train_stage,
        pretrain_epochs=args.pretrain_epochs,
        freeze_shared_after_pretrain=str_to_bool(args.freeze_shared_after_pretrain),
        freeze_sb_after_pretrain=str_to_bool(args.freeze_sb_after_pretrain),
        pretrained_weights_path=args.pretrained_weights_path,
        fine_tune_lr=args.fine_tune_lr,
        two_stage_auto=str_to_bool(args.two_stage_auto),
        use_residual_in_multi=str_to_bool(args.use_residual_in_multi),
        norm_weight_sb_alpha=args.norm_weight_sb_alpha,
        norm_pre_weight_sb_alpha=args.norm_pre_weight_sb_alpha,
    )

PROCESS_MAP = {
    -5: 0,  # WW with H
    -4: 1,  # ZZ with H
    -3: 2,  # ZZ
    -2: 3,  # WZ
    -1: 4,  # WW
     1: -1, # Fondo (da ignorare in process_head)
     2: -1, # Fondo (da ignorare in process_head)
}

SB_MAP = {
    -5: 1, -4: 1, -3: 1, -2: 1, -1: 1,
     1: 0,  2: 0,
}

def encode_labels(y: np.ndarray) -> np.ndarray:
    return np.array([PROCESS_MAP[v] for v in y], dtype=np.int64)

class CustomLabelEncoder:
    def __init__(self, process_map: dict):
        self.process_map = process_map

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        return np.array([self.process_map.get(int(v), -1) for v in y], dtype=np.int64)

class CustomMultiTaskEncoder:
    def __init__(self, sb_map: dict, process_map: dict):
        self.sb_map = sb_map
        self.process_map = process_map
        self.num_process_classes = int(max(v for v in process_map.values() if v >= 0) + 1)

    def transform(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y)
        sb = np.array([self.sb_map.get(int(v), 0) for v in y], dtype=np.float32).reshape(-1, 1)
        process = np.zeros((len(y), self.num_process_classes), dtype=np.float32)

        for idx, raw_value in enumerate(y):
            mapped = self.process_map.get(int(raw_value), -1)
            if mapped >= 0:
                process[idx, mapped] = 1.0

        return sb, process


# -----------------------------
# ROOT extraction
# -----------------------------
def load_root_to_dataframe(
    root_files: Sequence[str],
    tree_name: str,
    feature_branches: Sequence[str],
    label_branch: str,
    weight_branch: str,
    chunk_size: int = 250000,
    max_entries: Optional[int] = None
) -> pd.DataFrame:    
    branches = list(feature_branches) + [label_branch, weight_branch]
    frames: List[pd.DataFrame] = []

    print("\nInizio caricamento a blocchi (Chunk Size: 250k) per singolo file...")
    for path in root_files:
        print(f"Apertura del file: {path}")
        
        chunk_iterator = uproot.iterate(
            f"{path}:{tree_name}",
            expressions=branches,
            step_size=chunk_size,
            entry_stop=max_entries,
            library="pd"
        )
        
        for i, chunk_df in enumerate(chunk_iterator):
            frames.append(chunk_df)
            print(f" -> Elaborato blocco {i+1}: {len(chunk_df)} eventi caricati in RAM.")

    # Concatena tutti i blocchi estratti lungo le righe (verticalmente)
    df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Caricamento completato! Dimensioni finali dataset: {df.shape}\n")
    
    return df

def load_root_to_dataframe_per2coppie(
    file_coppia1: str,
    file_coppia2: str,
    tree_name: str,
    feature_branches: Sequence[str],
    label_branch: str,
    weight_branch: str,
    chunk_size: int = 250000,
    max_entries: int = None  
) -> pd.DataFrame:
    neutrino_branch = ["neutrinos_pt", "neutrinos_phi", "neutrinos_eta"]
    branches_f1 = list(feature_branches) + [label_branch, weight_branch]
    branches_f2 = [b for b in feature_branches if b not in neutrino_branch]    
    iter_f1 = uproot.iterate(
        f"{file_coppia1}:{tree_name}", 
        expressions=branches_f1, 
        step_size=chunk_size, 
        entry_stop=max_entries, 
        library="pd"
    )
    iter_f2 = uproot.iterate(
        f"{file_coppia2}:{tree_name}", 
        expressions=branches_f2, 
        step_size=chunk_size, 
        entry_stop=max_entries, 
        library="pd"
    )

    frames = []
    print("\nInizio caricamento a blocchi (Chunk Size: 250k) per le due coppie...")
    
    for i, (df1, df2) in enumerate(zip(iter_f1, iter_f2)):
        df1 = df1.rename(columns={col: f"{col}_coppia1" for col in branches_f2})
        df2 = df2.rename(columns={col: f"{col}_coppia2" for col in branches_f2})
        
        df_merged = pd.concat([df1, df2], axis=1)
        frames.append(df_merged)
        
        print(f" -> Elaborato blocco {i+1}: {len(df_merged)} eventi caricati in RAM.")
        
    final_df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Caricamento completato! Dimensioni finali dataset: {final_df.shape}\n")
    
    return final_df


# -----------------------------
# Feature engineering helpers
# -----------------------------
def wrap_phi(phi: pd.Series | np.ndarray) -> np.ndarray:
    phi = np.asarray(phi)
    return (phi + np.pi) % (2 * np.pi) - np.pi

def add_trig_phi_features(df: pd.DataFrame, phi_columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in phi_columns:
        phi = wrap_phi(df[col].values)
        df[f"{col}_sin"] = np.sin(phi)
        df[f"{col}_cos"] = np.cos(phi)
    return df

def process_pdg_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pdg_cols = [c for c in df.columns if c.startswith("pdg")]

    for col in pdg_cols:
        suffix = col.replace("pdg", "")  # es. "_lep1"
        abs_pdg = df[col].abs()

        df[f"charge{suffix}"] = -np.sign(df[col]).astype(np.float32)
        df[f"is_e{suffix}"] = (abs_pdg == 11).astype(np.float32)
        df[f"is_mu{suffix}"] = (abs_pdg == 13).astype(np.float32)
        df[f"is_tau{suffix}"] = (abs_pdg == 15).astype(np.float32)

    if pdg_cols:
        df = df.drop(columns=pdg_cols)

    return df

def apply_sqrt_to_separations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        is_deta_dr = col_lower.startswith("deta") or col_lower.startswith("dr")

        if is_deta_dr:
            df[col] = np.sqrt(np.abs(df[col]))

    return df

def scale_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        c = col.lower()

        is_pt = c.startswith("pt") or c == "neutrinos_pt"
        is_mass = (
            c.startswith("m_")
            or c == "m"
            or c == "m 4l"
            or c.startswith("m recoil")
            or c.startswith("mt ")  
            or c == "neutrinos_m"
        )

        if is_pt:
            df[col] = df[col] / 120.0
        elif is_mass:
            df[col] = df[col] / 240.0

    return df

def transform_dphi_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        is_dphi = col_lower.startswith("dphi")

        if is_dphi:
            df[col] = np.cos(df[col])

    return df

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    phi_cols = [c for c in df.columns if c.startswith("phi")]
    if phi_cols:
        df = add_trig_phi_features(df, phi_cols)
        df = df.drop(columns=phi_cols)

    df = process_pdg_features(df)
    df = apply_sqrt_to_separations(df)
    df = scale_kinematic_features(df)
    df = transform_dphi_features(df)

    return df


# -----------------------------
# Plotting utilities
# -----------------------------
class BatchLRTracker(Callback):
    def __init__(self):
        super().__init__()
        self.batch_lrs: List[float] = []

    def on_train_batch_end(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.batch_lrs.append(float(lr))

def plot_history(history: keras.callbacks.History, output_dir: str) -> None:
    if history is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    hist = history.history

    def _plot_metric(metric_name: str, filename: str, title: str):
        if metric_name not in hist:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(hist[metric_name], label=metric_name)
        if f"val_{metric_name}" in hist:
            plt.plot(hist[f"val_{metric_name}"], label=f"val_{metric_name}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()

    _plot_metric("loss", "loss_curve_total.png", "Total Loss")
    _plot_metric("sb_head_loss", "loss_curve_bin.png", "Binary Loss (S/B)")
    _plot_metric("sb_head_accuracy", "accuracy_curve_bin.png", "Binary Accuracy")
    _plot_metric("process_head_loss", "loss_curve_multi.png", "Multiclass Loss")
    _plot_metric("process_head_accuracy", "accuracy_curve_multi.png", "Multiclass Accuracy")

def plot_conf_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: str,
    filename: str,
    sample_weight: Optional[np.ndarray] = None,
    sample_weight_raw: Optional[np.ndarray] = None,
    total_events: Optional[int] = None,
    threshold_info: str = ""
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)

    # Matrici normalizzate
    cm_true = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, normalize="true")
    cm_pred = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, normalize="pred")

    # Conteggi Eventi Grezzi (Raw)
    raw_true_counts = np.array([np.sum(y_true == i) for i in range(num_classes)], dtype=int)
    raw_pred_counts = np.array([np.sum(y_pred == i) for i in range(num_classes)], dtype=int)

    # Conteggi Fisici (Scalati alla luminosità)
    physical_true_counts = np.zeros(num_classes, dtype=float)
    physical_pred_counts = np.zeros(num_classes, dtype=float)

    if sample_weight_raw is not None and total_events is not None:
        phys_weights = scale_sample_weight_to_luminosity(sample_weight_raw, total_events)
        physical_true_counts = np.array([np.sum(phys_weights[y_true == i]) for i in range(num_classes)], dtype=float)
        physical_pred_counts = np.array([np.sum(phys_weights[y_pred == i]) for i in range(num_classes)], dtype=float)
    elif sample_weight is not None:
        physical_true_counts = np.array([np.sum(sample_weight[y_true == i]) for i in range(num_classes)], dtype=float)
        physical_pred_counts = np.array([np.sum(sample_weight[y_pred == i]) for i in range(num_classes)], dtype=float)

    # Etichette dinamiche per Righe (True) e Colonne (Pred) affidate alla funzione base
    row_labels = [
        f"{class_names[i]}\nraw={raw_true_counts[i]}\nphys={physical_true_counts[i]:.1f}"
        for i in range(num_classes)
    ]
    col_labels = [
        f"{class_names[i]}\nraw={raw_pred_counts[i]}\nphys={physical_pred_counts[i]:.1f}"
        for i in range(num_classes)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Titoli dinamici
    title_true = "Normalizzata per Classe Vera\n(Efficienza / Righe = 1)"
    title_pred = "Normalizzata per Classe Predetta\n(Purità / Colonne = 1)"
    if threshold_info:
        title_true += f"\n{threshold_info}"
        title_pred += f"\n{threshold_info}"

    # Matrice normalizzata per Classe Vera (Righe)
    disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=class_names)
    disp_true.plot(ax=ax1, cmap="Blues", values_format=".3f", colorbar=True, xticks_rotation=45)
    ax1.set_title(title_true)
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))
    ax1.set_yticklabels(row_labels)
    ax1.set_xticklabels(col_labels, rotation=45, ha="right")

    # Matrice normalizzata per Classe Predetta (Colonne)
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=class_names)
    disp_pred.plot(ax=ax2, cmap="Blues", values_format=".3f", colorbar=True, xticks_rotation=45)
    ax2.set_title(title_pred)
    ax2.set_xticks(np.arange(num_classes))
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_yticklabels(row_labels)
    ax2.set_xticklabels(col_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)

def plot_cascaded_confusion_matrix(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_probs_dict: Dict[str, np.ndarray],
    w_dict: Dict[str, np.ndarray],
    w_raw_dict: Dict[str, np.ndarray],
    signal_class_names: Sequence[str],
    output_dir: str,
    total_events: int,
    threshold: float = 0.5,
    filename: str = "confusion_matrix_cascaded_6x6.png"
) -> None:
    # 1. Estrazione vettori
    y_true_sb = y_true_dict["sb_head"].flatten()
    y_prob_sb = y_pred_probs_dict["sb_head"].flatten()
    w_events = w_dict["sb_head"].flatten()
    w_events_raw = w_raw_dict["sb_head"].flatten()

    y_true_multi = np.argmax(y_true_dict["process_head"], axis=1)
    y_pred_multi = np.argmax(y_pred_probs_dict["process_head"], axis=1)

    num_signals = len(signal_class_names)
    bkg_class_idx = num_signals  # Indice 5 per il Fondo Totale

    # 2. Costruzione Verità (0..4 per i 5 segnali, 5 per il Fondo)
    y_true_6 = np.where(y_true_sb == 1, y_true_multi, bkg_class_idx)

    # 3. Logica a Cascata
    y_pred_6 = np.where(y_prob_sb >= threshold, y_pred_multi, bkg_class_idx)

    # 4. Nomi delle classi semplici (I raw/phys vengono formattati da plot_conf_matrix)
    cascaded_class_names = list(signal_class_names) + ["Fondo Totale"]

    # 5. Generazione grafico
    plot_conf_matrix(
        y_true=y_true_6,
        y_pred=y_pred_6,
        class_names=cascaded_class_names,
        output_dir=output_dir,
        filename=filename,
        sample_weight=w_events,
        sample_weight_raw=w_events_raw,
        total_events=total_events,
        threshold_info=f"(Cascata con P(S/B) >= {threshold})"
    )

def plot_multitask_confusion_matrices(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_probs_dict: Dict[str, np.ndarray],
    w_dict: Dict[str, np.ndarray],
    w_raw_dict: Dict[str, np.ndarray],
    signal_class_names: Sequence[str],
    output_dir: str,
    total_events: int,
    cascade_threshold: float = 0.5
) -> None:
    # -------------------------------------------------------------------------
    # A. MATRICE 2x2: TESTA BINARIA (S/B) - ORA SINCRONIZZATA CON cascade_threshold
    # -------------------------------------------------------------------------
    y_true_sb = y_true_dict["sb_head"].flatten()
    
    y_pred_sb = (y_pred_probs_dict["sb_head"].flatten() >= cascade_threshold).astype(int)
    
    w_sb = w_dict["sb_head"].flatten()
    w_sb_raw = w_raw_dict["sb_head"].flatten()

    sb_base_names = ["Fondo Totale", "Segnale VBS"]

    plot_conf_matrix(
        y_true=y_true_sb,
        y_pred=y_pred_sb,
        class_names=sb_base_names,
        output_dir=output_dir,
        filename="confusion_matrix_sb_head.png",
        sample_weight=w_sb,
        sample_weight_raw=w_sb_raw,
        total_events=total_events,
        threshold_info=f"(Taglio P(S/B) >= {cascade_threshold})"
    )

    # -------------------------------------------------------------------------
    # B. MATRICE 5x5: TESTA MULTICLASSE (Sui soli eventi di Vero Segnale)
    # -------------------------------------------------------------------------
    is_true_signal = y_true_sb == 1
    if np.any(is_true_signal):
        y_true_process = np.argmax(y_true_dict["process_head"][is_true_signal], axis=1)
        y_pred_process = np.argmax(y_pred_probs_dict["process_head"][is_true_signal], axis=1)
        w_multi = w_dict["process_head"][is_true_signal].flatten()
        w_multi_raw = w_raw_dict["process_head"][is_true_signal].flatten()

        plot_conf_matrix(
            y_true=y_true_process,
            y_pred=y_pred_process,
            class_names=signal_class_names,
            output_dir=output_dir,
            filename="confusion_matrix_process_head.png",
            sample_weight=w_multi,
            sample_weight_raw=w_multi_raw,
            total_events=total_events,
            threshold_info="(Nessun filtro S/B applicato)"
        )
    else:
        print("[INFO] Nessun evento segnale presente nel test set per la matrice multiclass.")

    # -------------------------------------------------------------------------
    # C. MATRICE 6x6: LOGICA A CASCATA (Filtro Binario + Multiclasse)
    # -------------------------------------------------------------------------
    plot_cascaded_confusion_matrix(
        y_true_dict=y_true_dict,
        y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_dict,
        w_raw_dict=w_raw_dict,
        signal_class_names=signal_class_names,
        output_dir=output_dir,
        total_events=total_events,
        threshold=cascade_threshold,
        filename="confusion_matrix_cascaded_6x6.png"
    )

def plot_multitask_roc_curves(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_probs_dict: Dict[str, np.ndarray],
    w_dict: Dict[str, np.ndarray],
    signal_class_names: Sequence[str],
    output_dir: str
) -> None:
    plot_roc_binary(
        y_true=y_true_dict["sb_head"].flatten(),
        y_prob=y_pred_probs_dict["sb_head"].flatten(),
        output_dir=output_dir,
        sample_weight=w_dict["sb_head"].flatten(),
    )

    is_true_signal = y_true_dict["sb_head"].flatten() == 1
    if np.any(is_true_signal):
        y_true_process = y_true_dict["process_head"][is_true_signal]
        y_pred_process = y_pred_probs_dict["process_head"][is_true_signal]
        w_multi = w_dict["process_head"][is_true_signal].flatten()
        plot_roc_multiclass(
            y_true_onehot=y_true_process,
            y_prob=y_pred_process,
            class_names=signal_class_names,
            output_dir=output_dir,
            sample_weight=w_multi,
        )
    else:
        print("[INFO] Nessun evento segnale presente nel test set per la ROC multiclass.")

def scale_sample_weight_to_luminosity(
    sample_weight: np.ndarray,
    total_events: int,
    luminosity: float = LUMINOSITY,
) -> np.ndarray:
    if total_events <= 0:
        raise ValueError("total_events must be positive")
    return sample_weight.astype(np.float64) / float(total_events) * float(luminosity)

def plot_roc_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (Segnale vs Fondo)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_binary.png"), dpi=200)
    plt.close()
    return roc_auc

def plot_roc_multiclass(
    y_true_onehot: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_dir: str,
    sample_weight: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_classes = y_true_onehot.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        y_true_i = y_true_onehot[:, i]
        if np.unique(y_true_i).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_i, y_prob[:, i], sample_weight=sample_weight)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves Multiclasse (One-vs-Rest sui Segnali)")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_multiclass.png"), dpi=200)
    plt.close()

def plot_detailed_lr(lr_tracker: BatchLRTracker, output_dir: str):
    if not lr_tracker.batch_lrs:
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(lr_tracker.batch_lrs, color="#1f77b4", linewidth=2, label="LR per Batch")
    plt.title("Evoluzione del Learning Rate")
    plt.xlabel("Iterazioni (Train Steps / Batch)")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "detailed_lr_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Grafico dettagliato del Learning Rate salvato in: {save_path}")

def analyze_binary_signal_region(
    y_true: np.ndarray,
    y_prob_sig: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float = 0.85,
    report_path: str = None
):
    def log(text=""):
        print(text)
        if report_path:
            with open(report_path, "a") as f:
                f.write(text + "\n")

    log(f"\n{'='*50}")
    log(f" ANALISI SIGNAL REGION BINARIA (Taglio >= {threshold*100}%)")
    log(f"{'='*50}")

    pass_cut_mask = y_prob_sig >= threshold
    total_yield = np.sum(sample_weight)
    passed_yield = np.sum(sample_weight[pass_cut_mask])

    if passed_yield <= 0:
        log("Nessun evento supera il taglio. Nessuna analisi aggiuntiva possibile.\n")
        return

    is_true_signal = y_true == 1
    true_positives_yield = np.sum(sample_weight[is_true_signal & pass_cut_mask])
    total_true_signals_yield = np.sum(sample_weight[is_true_signal])

    purity = (true_positives_yield / passed_yield) * 100
    efficiency = (true_positives_yield / total_true_signals_yield) * 100 if total_true_signals_yield > 0 else 0

    log(f"Yield totale analizzato (S+B iniziale): {total_yield:.2f}")
    log(f"Yield sopravvissuto al taglio (S+B finale): {passed_yield:.2f}")
    log(f"Purity (S_pass / (S_pass + B_pass)): {purity:.1f}%")
    log(f"Efficiency (S_pass / S_tot): {efficiency:.1f}%\n")

def analyze_signal_region(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    sample_weight: np.ndarray,
    class_names: list,
    target_class_idx: int = 0,
    threshold: float = 0.85,
    report_path: str = None
):
    def log(text=""):
        print(text)
        if report_path:
            with open(report_path, "a") as f:
                f.write(text + "\n")

    target_name = class_names[target_class_idx]

    log(f"\n{'='*50}")
    log(f" ANALISI CANALE: {target_name} (Taglio Prob >= {threshold*100}%)")
    log(f"{'='*50}")

    target_probs = y_probs[:, target_class_idx]
    pass_cut_mask = target_probs >= threshold

    total_yield = np.sum(sample_weight)
    passed_yield = np.sum(sample_weight[pass_cut_mask])

    if passed_yield <= 0:
        log("Nessun evento supera il taglio. Nessuna analisi aggiuntiva possibile.\n")
        return

    log(f"Yield totale disponibile: {total_yield:.2f}")
    log(f"Yield sopravvissuto (Eventi previsti): {passed_yield:.2f}")
    log("\nComposizione del campione sopravvissuto (Expected Yields):")

    unique_classes = np.unique(y_true[pass_cut_mask])
    for cls in unique_classes:
        mask = pass_cut_mask & (y_true == cls)
        yield_cls = np.sum(sample_weight[mask])
        class_name = class_names[int(cls)] if 0 <= int(cls) < len(class_names) else f"Classe {int(cls)}"
        log(f"  {class_name}: {yield_cls:.2f}")

    is_true_target = y_true == target_class_idx
    true_positives_yield = np.sum(sample_weight[is_true_target & pass_cut_mask])
    total_true_target_yield = np.sum(sample_weight[is_true_target])

    purity = (true_positives_yield / passed_yield) * 100
    efficiency = (true_positives_yield / total_true_target_yield) * 100 if total_true_target_yield > 0 else 0

    log(f"\n--- RIASSUNTO METRICHE AL TAGLIO {threshold} ---")
    log(f"Purity del canale {target_name}: {purity:.1f}%")
    log(f"Efficiency del canale {target_name}: {efficiency:.1f}%\n")

def plot_prediction_composition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: str,
    filename_prefix: str = "prediction_composition",
    sample_weight: Optional[np.ndarray] = None,
    sample_weight_raw: Optional[np.ndarray] = None,
    total_events: Optional[int] = None,
    title_suffix: str = "",
    bkg_label: str = "Fondo Totale"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)

    # -------------------------------------------------------------------------
    # 0. MATRICI DI CONFUSIONE E NORMALIZZAZIONI
    # -------------------------------------------------------------------------
    weight_for_cm = sample_weight if sample_weight is not None else None
    phys_weights = None
    if sample_weight_raw is not None and total_events is not None:
        phys_weights = scale_sample_weight_to_luminosity(sample_weight_raw, total_events)

    cm_raw = confusion_matrix(y_true, y_pred)
    cm_proc = confusion_matrix(y_true, y_pred, sample_weight=weight_for_cm)
    phys_cm = confusion_matrix(y_true, y_pred, sample_weight=phys_weights) if phys_weights is not None else None

    # Base fisica per i grafici
    cm_abs = phys_cm if phys_cm is not None else cm_proc
    has_weights = phys_cm is not None or weight_for_cm is not None

    # NORMALIZZAZIONE PER RIGA (True Label): sum_j(P(Pred_j | True_i)) = 1.0
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm_row = np.divide(cm_abs, row_sums, out=np.zeros_like(cm_abs, dtype=float), where=row_sums != 0)

    def _raw_phys_str(raw_value, phys_value=None) -> str:
        raw_str = f"raw={int(round(raw_value))}"
        if phys_value is None:
            return raw_str
        return f"{raw_str}, phys={phys_value:.1f}"

    # Indici esclusivi per le sole classi di segnale
    signal_indices = [i for i, name in enumerate(class_names) if name != bkg_label]

    colors_true = plt.cm.tab10(np.linspace(0, 1, num_classes))
    x_positions = np.arange(num_classes)

    if phys_cm is not None:
        y_label_abs = "Eventi Fisici (Somma dei Pesi scalati alla luminosità)"
    elif weight_for_cm is not None:
        y_label_abs = "Eventi Fisici (Somma dei Pesi utilizzati per CM)"
    else:
        y_label_abs = "Numero di Eventi Grezzi"

    # -------------------------------------------------------------------------
    # 1. PLOT SINGOLI ASSOLUTI (Per singola VERA classe di segnale)
    # -------------------------------------------------------------------------
    # Mostra come gli eventi appartenenti a una Vera Classe i si distribuiscono nelle Predizioni j
    for idx_sig, true_class_idx in enumerate(signal_indices):
        true_name = class_names[true_class_idx]
        
        # Estrazione della RIGA i (distribuzione dell'evento vero sulle predizioni)
        heights = cm_abs[true_class_idx, :]
        heights_raw = cm_raw[true_class_idx, :]
        row_integral = np.sum(heights)
        row_integral_raw = np.sum(heights_raw)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x_positions, 
            heights, 
            color=colors_true[true_class_idx], 
            alpha=0.85, 
            edgecolor='black', 
            linewidth=1.0, 
            width=0.5
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Classe Predetta (Predicted Label)", fontsize=12)
        ax.set_ylabel(y_label_abs, fontsize=12)
        ax.set_title(f"Smistamento Predizioni - Vera Classe: {true_name}\n{title_suffix}", fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for j, h in enumerate(heights):
            if h > 0:
                fmt = f"{h:.1f}" if has_weights else f"{int(h)}"
                ax.text(j, h, fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

        integral_str = _raw_phys_str(row_integral_raw, row_integral if phys_cm is not None else None)
        integral_label = f"Totale Eventi Veri ({true_name})\n{integral_str}"
        ax.text(
            0.98, 0.96, integral_label, 
            transform=ax.transAxes, ha='right', va='top', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round')
        )

        plt.tight_layout()
        clean_name = true_name.replace("+", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_absolute_true_{clean_name}.png"), dpi=200)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. STACKED BAR CHART ASSOLUTO (Smistamento Segnale Impilato per Predizione)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    bottoms_abs = np.zeros(num_classes)

    # Impila SOLO i contributi delle VERE classi di segnale (escludendo il Fondo Reale)
    for i in signal_indices:
        heights_true_i = cm_abs[i, :] # Dove finisce la vera classe i
        ax.bar(
            x_positions,
            heights_true_i,
            bottom=bottoms_abs,
            label=f"Vero: {class_names[i]}",
            color=colors_true[i],
            edgecolor='black',
            linewidth=0.5,
            width=0.55
        )
        bottoms_abs += heights_true_i

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Classe Predetta (Predicted Label)", fontsize=12)
    ax.set_ylabel(y_label_abs, fontsize=12)
    ax.set_title(f"Smistamento Assoluto Eventi di Segnale nelle Classi Predette\n{title_suffix}", fontsize=14)
    ax.legend(title="Vera Classe (True Label)", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_stacked_absolute.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. STACKED BAR CHART NORMALIZZATO (100% Efficienza / Migrazione Relativa)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Asse X sulle sole VERE classi di segnale per mostrare il loro smistamento relativo
    x_positions_sig = np.arange(len(signal_indices))
    signal_class_names = [class_names[i] for i in signal_indices]
    bottoms_norm = np.zeros(len(signal_indices))

    # Impila per ogni Vera Classe di segnale la frazione di eventi finita nelle varie classi PREDETTE
    for j in range(num_classes):
        fraction_pred_j = cm_norm_row[signal_indices, j] # Frazione di ogni vera classe finita in pred_j
        ax.bar(
            x_positions_sig,
            fraction_pred_j,
            bottom=bottoms_norm,
            label=f"Predetto: {class_names[j]}",
            color=plt.cm.tab10(j / num_classes),
            edgecolor='black',
            linewidth=0.5,
            width=0.55
        )
        bottoms_norm += fraction_pred_j

    ax.set_xticks(x_positions_sig)
    ax.set_xticklabels(signal_class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Vera Classe di Segnale (True Label)", fontsize=12)
    ax.set_ylabel("Frazione di Eventi Smistati (Somma Barra = 1.0)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Distribuzione Relativa delle Predizioni per Vera Classe (Efficienza)\n{title_suffix}", fontsize=14)
    ax.legend(title="Classe Predetta", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_stacked_normalized.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. PLOT OVERLAYED NORMALIZZATO (Confronto Step Chart tra Vere Classi)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    bin_edges = np.arange(num_classes + 1) - 0.5

    for idx_sig, true_class_idx in enumerate(signal_indices):
        true_name = class_names[true_class_idx]
        heights_norm = cm_norm_row[true_class_idx, :]
        original_integral = np.sum(cm_abs[true_class_idx, :])
        yield_str = f"{original_integral:.1f}" if has_weights else f"{int(original_integral)}"

        ax.stairs(
            heights_norm, 
            bin_edges, 
            label=f"Vero: {true_name} (Int: {yield_str})", 
            color=colors_true[true_class_idx], 
            linewidth=2.2
        )
        ax.fill_between(
            np.arange(num_classes), 
            heights_norm, 
            step="mid", 
            color=colors_true[true_class_idx], 
            alpha=0.08
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Classe Predetta (Predicted Label)", fontsize=12)
    ax.set_ylabel("Frazione Relativa (Area Singola Linea = 1.0)", fontsize=12)
    ax.set_ylim(0, max(1.05, cm_norm_row.max() * 1.1))
    ax.set_title(f"Distribuzioni Vere Overlayed Normalizzate a 1 (Efficienza/Migrazione)\n{title_suffix}", fontsize=14)
    ax.legend(title="Vera Classe (True Label)", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_overlayed_normalized.png"), dpi=200)
    plt.close(fig)


# -----------------------------
# Weight handling
# -----------------------------
def normalize_weights(
    weights: np.ndarray,
    y: np.ndarray = None,
    mode: str = "mean_one",
    alpha: float = 0.5,
    class_multipliers: dict = None
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if mode == "sum_to_one":
        total = np.sum(w)
        return w / total if total > 0 else w

    if mode == "mean_one":
        mean = np.mean(w)
        return w / mean if mean > 0 else w

    if mode == "per_class":
        if y is None:
            raise ValueError("y is required for per_class weight normalization")
        y = np.asarray(y)
        result = np.zeros_like(w)
        for cls in np.unique(y):
            mask = y == cls
            class_sum = np.sum(w[mask])
            if class_sum > 0:
                result[mask] = w[mask] / class_sum
        if class_multipliers:
            for cls, factor in class_multipliers.items():
                result[y == cls] *= factor
        return result

    if mode == "per_class_sublinear":
        if y is None:
            raise ValueError("y is required for per_class_sublinear weight normalization")
        y = np.asarray(y)
        result = np.zeros_like(w)
        for cls in np.unique(y):
            mask = y == cls
            class_count = np.sum(mask)
            class_sum = np.sum(w[mask])
            target = class_count**alpha
            if class_sum > 0:
                result[mask] = w[mask] * (target / class_sum)
        if class_multipliers:
            for cls, factor in class_multipliers.items():
                result[y == cls] *= factor
        return result

    if mode == "none":
        return w

    raise ValueError(f"Unknown normalization mode: {mode}")

def compute_class_weights_from_sample_weights(
    y: np.ndarray, sample_weight: np.ndarray
) -> Dict[int, float]:
    classes = np.unique(y)
    result = {}
    for c in classes:
        mask = y == c
        count = np.sum(mask)
        if count > 0:
            result[int(c)] = np.sum(sample_weight[mask]) / count
        else:
            result[int(c)] = 0.0
    return result


# -----------------------------
# Optional bootstrap
# -----------------------------
def bootstrap_resample(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    target_samples = n_samples if n_samples is not None else np.max(counts)

    X_resampled = []
    y_resampled = []
    w_resampled = []

    for cls, count in zip(unique_classes, counts):
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        w_cls = w[mask]

        if len(X_cls) == 0:
            continue

        indices = rng.choice(len(X_cls), size=target_samples, replace=True)
        scale = len(X_cls) / target_samples if target_samples > 0 else 1.0

        X_resampled.append(X_cls[indices])
        y_resampled.append(y_cls[indices])
        w_resampled.append(w_cls[indices] * scale)

    if len(X_resampled) == 0:
        return X, y, w

    X_resampled = np.concatenate(X_resampled, axis=0)
    y_resampled = np.concatenate(y_resampled, axis=0)
    w_resampled = np.concatenate(w_resampled, axis=0)

    permutation = rng.permutation(len(X_resampled))
    X_resampled = X_resampled[permutation]
    y_resampled = y_resampled[permutation]
    w_resampled = w_resampled[permutation]

    print(f"[Bootstrap] Dataset bilanciato: ogni classe ora ha {target_samples} eventi.")
    return X_resampled, y_resampled, w_resampled


# -----------------------------
# Split and preprocess
# -----------------------------
def make_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> Tuple[np.ndarray, ...]:
    strat = y if stratify else None

    X_train_val, X_test, y_train_val, y_test, w_train_val, w_test = train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    val_frac_rel = val_size / (1.0 - test_size)
    strat2 = y_train_val if stratify else None

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_train_val,
        y_train_val,
        w_train_val,
        test_size=val_frac_rel,
        random_state=random_state,
        stratify=strat2,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test

@dataclass
class Preprocessor:
    scaler: StandardScaler
    label_encoder: CustomMultiTaskEncoder

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def transform_y(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        sb, process = self.label_encoder.transform(y)
        return {"sb_head": sb, "process_head": process}

def fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Preprocessor:
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Salva lo scaler già fitted per usarlo successivamente
    joblib.dump(scaler, "standard_scaler.pkl")
    label_encoder = CustomMultiTaskEncoder(sb_map=SB_MAP, process_map=PROCESS_MAP)
    preprocessor = Preprocessor(scaler=scaler, label_encoder=label_encoder)
    joblib.dump(preprocessor, "preprocessor.pkl")
    return preprocessor


# -----------------------------
# Model and training utilities
# -----------------------------
def analyze_cascaded_signal_region(
    y_true_multi_int: np.ndarray,
    y_prob_bin: np.ndarray,
    y_prob_multi: np.ndarray,
    class_names: list,
    threshold: float = 0.85,
    report_path: str = None
):
    def log(text=""):
        print(text)
        if report_path:
            with open(report_path, "a") as f:
                f.write(text + "\n")

    log(f"\n{'='*50}")
    log(f" ANALISI CASCATA (Taglio S/B >= {threshold*100}%)")
    log(f"{'='*50}")

    pass_cut_mask = y_prob_bin >= threshold
    total_events = len(y_true_multi_int)
    passed_events = int(np.sum(pass_cut_mask))

    if passed_events == 0:
        log("Nessun evento supera il taglio S/B.\n")
        return

    log(f"Eventi totali analizzati: {total_events}")
    log(f"Eventi in Signal Region (sopravvissuti): {passed_events}")

    y_true_passed = y_true_multi_int[pass_cut_mask]
    y_pred_passed = np.argmax(y_prob_multi[pass_cut_mask], axis=1)

    accuracy = np.mean(y_pred_passed == y_true_passed) if passed_events > 0 else 0.0
    log(f"Accuracy della testa multiclasse nella signal region: {accuracy:.3f}")

    log("\nComposizione reale degli eventi in signal region:")
    for cls in np.unique(y_true_passed):
        cls_name = class_names[int(cls)] if 0 <= int(cls) < len(class_names) else f"Classe {int(cls)}"
        log(f"  {cls_name}: {(y_true_passed == cls).sum()} eventi")

    log("\nComposizione predetta degli eventi in signal region:")
    for cls in np.unique(y_pred_passed):
        cls_name = class_names[int(cls)] if 0 <= int(cls) < len(class_names) else f"Classe {int(cls)}"
        log(f"  {cls_name}: {(y_pred_passed == cls).sum()} eventi")

    log("")

def residual_block(x, units, name_prefix, use_layer_norm=False):
    shortcut = x

    y = Dense(units, name=f"{name_prefix}_dense1")(x)
    if use_layer_norm:
        y = keras.layers.LayerNormalization(name=f"{name_prefix}_ln1")(y)
    y = Activation("relu", name=f"{name_prefix}_act1")(y)

    y = Dense(units, name=f"{name_prefix}_dense2")(y)
    if use_layer_norm:
        y = keras.layers.LayerNormalization(name=f"{name_prefix}_ln2")(y)
    # no activation yet, will be applied after add

    # project shortcut if dimension mismatch
    if int(shortcut.shape[-1]) != int(units):
        shortcut = Dense(units, name=f"{name_prefix}_proj")(shortcut)

    out = Add(name=f"{name_prefix}_add")([shortcut, y])
    out = Activation("relu", name=f"{name_prefix}_out_act")(out)
    return out

def make_metrics():
    return {
        "sb_head": [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
        "process_head": [
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc", multi_label=True, curve="ROC"),
            keras.metrics.Precision(name="precision"),   # macro-like batch precision
            keras.metrics.Recall(name="recall"),         # macro-like batch recall
        ],
    }

def compiler_utils(config: Config, decay_steps: Optional[int] = None,):
    if config.use_warmup:
        # Use a warmup + cosine decay schedule. decay_steps can be passed
        # (computed in train_model) or falls back to a conservative default.
        if decay_steps is None:
            decay_steps = max(10000, int(config.warmup_steps * 10))
        lr_to_use = WarmUpCosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=decay_steps,
            warmup_steps=config.warmup_steps,
            min_learning_rate=1e-7,
        )
    else:
        lr_to_use = config.learning_rate

    # ==========================================
    # 2. IMPLEMENTAZIONE GRADIENT ACCUMULATION
    # ==========================================
    # Assembliamo un dizionario di argomenti base per l'ottimizzatore
    opt_kwargs = {
        "learning_rate": lr_to_use,
        "global_clipnorm": config.global_clipnorm, # es. 1.0
    } 

    if config.grad_accum_steps > 1:
        # Supportato nativamente nelle versioni recenti di Keras
        opt_kwargs["gradient_accumulation_steps"] = config.grad_accum_steps

    if config.optimizer_type == "AdamW":
        opt = keras.optimizers.AdamW(weight_decay=config.weight_decay, epsilon=1e-8, **opt_kwargs)
    elif config.optimizer_type == "Adam":
        opt = keras.optimizers.Adam(**opt_kwargs)
    elif config.optimizer_type == "SGD":
        opt = keras.optimizers.SGD(momentum=config.sgd_momentum, **opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")

    losses = {
        "sb_head": keras.losses.BinaryFocalCrossentropy(gamma=config.focal_gamma_sb, alpha=config.focal_alpha_sb) if config.use_focal_loss_sb else keras.losses.BinaryCrossentropy(),
        "process_head": keras.losses.CategoricalFocalCrossentropy(gamma=config.focal_gamma_multi, alpha=config.focal_alpha_multi) if config.use_focal_loss_multi else keras.losses.CategoricalCrossentropy(),
    }

    return opt, losses

def _align_outputs_for_model(model: keras.Model, y_dict, sw_dict):
    # Ottieni nomi delle uscite in modo robusto
    if hasattr(model, "output_names") and model.output_names:
        raw_names = model.output_names
    else:
        # fallback: usa model.outputs (tensors) e prendi la parte prima di '/'
        raw_names = [getattr(o, "name", str(o)) for o in getattr(model, "outputs", [])]

    # Normalizza a lista di stringhe semplici (senza suffissi di scope)
    output_names = []
    for n in raw_names:
        if isinstance(n, str):
            # se è tipo "sb_head:0" o "sb_head/Relu:0", prendi la parte utile
            n_clean = n.split(":")[0]
            n_clean = n_clean.split("/")[0]
            output_names.append(n_clean)
        else:
            # oggetto con .name
            name = getattr(n, "name", str(n))
            name = name.split(":")[0].split("/")[0]
            output_names.append(name)

    # Se il modello ha una sola uscita, ritorna array singolo (non dict)
    if len(output_names) == 1:
        name = output_names[0]
        # estrai y
        if isinstance(y_dict, dict):
            if name in y_dict:
                y_for_fit = y_dict[name]
            else:
                # fallback: prendi primo valore del dict
                y_for_fit = next(iter(y_dict.values()))
        else:
            y_for_fit = y_dict
        # estrai sample weight
        if sw_dict is None:
            sw_for_fit = None
        elif isinstance(sw_dict, dict):
            sw_for_fit = sw_dict.get(name, next(iter(sw_dict.values())))
        else:
            sw_for_fit = sw_dict
        return y_for_fit, sw_for_fit

    # Multi-output: costruisci dict coerenti con output_names
    y_for = {}
    sw_for = {}
    for name in output_names:
        if isinstance(y_dict, dict):
            y_for[name] = y_dict.get(name)
        else:
            y_for[name] = None
        if sw_dict is None:
            sw_for[name] = None
        elif isinstance(sw_dict, dict):
            sw_for[name] = sw_dict.get(name)
        else:
            sw_for[name] = None

    return y_for, sw_for

def _sw_to_vector(w):
    w = np.asarray(w)
    if w.ndim == 2:
        return np.sum(w, axis=1)
    return w.flatten()

def _check_loaded_layers(model, prefixes: Sequence[str]):
    found = False
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in prefixes):
            if layer.get_weights():
                found = True
                break
    if not found:
        print("[WARN] Nessun peso caricato per i layer con i prefissi richiesti. Controlla i nomi dei layer e il file dei pesi.")

def build_scheduler(config: Config):
    if config.scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=1e-7,
            verbose=1,
        )

    elif config.scheduler_type == "CosineAnnealing":
        def cosine_schedule(epoch):
            base_lr = config.learning_rate
            min_lr = 1e-7
            total = max(config.epochs, 1)
            cos_decay = 0.5 * (1 + np.cos(np.pi * epoch / total))
            return min_lr + (base_lr - min_lr) * cos_decay

        return LearningRateScheduler(cosine_schedule, verbose=1)

    elif config.scheduler_type == "OneCycle":
        def onecycle_schedule(epoch):
            base_lr = config.learning_rate
            max_lr = config.one_cycle_max_lr
            if config.epochs <= 1:
                return max_lr
            half = max(1, config.epochs // 2)
            if epoch < half:
                return base_lr + (max_lr - base_lr) * (epoch / half)
            return max_lr * (1 - (epoch - half) / (config.epochs - half))

        return LearningRateScheduler(onecycle_schedule, verbose=1)

    elif config.scheduler_type == "StepLR":
        def step_schedule(epoch):
            return config.learning_rate * (config.step_lr_gamma ** (epoch // config.step_lr_period))
        return LearningRateScheduler(step_schedule, verbose=1)

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps=0, min_learning_rate=0.0, name=None):
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = float(max(1, decay_steps))
        self.warmup_steps = float(max(0, warmup_steps))
        self.min_learning_rate = float(min_learning_rate)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        initial = tf.cast(self.initial_learning_rate, tf.float32)
        min_lr = tf.cast(self.min_learning_rate, tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = min_lr + (initial - min_lr) * (step / tf.maximum(warmup_steps, 1.0))
        else:
            warmup_lr = initial

        decay_steps = tf.cast(self.decay_steps, tf.float32)
        progress = (step - warmup_steps) / tf.maximum(decay_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        cosine_lr = self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * cosine_decay

        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "min_learning_rate": self.min_learning_rate,
            "name": self.name,
        }

def build_callbacks(config: Config, lr_tracker: BatchLRTracker = None):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if not getattr(config, "use_warmup", False):
        scheduler_cb = build_scheduler(config)
        if scheduler_cb is not None:
            callbacks.append(scheduler_cb)

    if lr_tracker is not None:
        callbacks.append(lr_tracker)
    return callbacks

def extend_callbacks(config: Config, model: keras.Model):
    has_sb_head = "sb_head" in model.output_names
    is_single_output = len(model.output_names) == 1
    if has_sb_head and not is_single_output:
        monitor_metric = "val_sb_head_pr_auc"
    elif is_single_output:
        monitor_metric = "val_pr_auc"  
    else:
        monitor_metric = "val_loss"
    mode = "max" if "pr_auc" in monitor_metric else "auto"

    best_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.output_dir, "best_model.keras"),
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=False,   
        mode=mode,
        verbose=1,
    )
    best_weight_cb = keras.callbacks.ModelCheckpoint(
        filepath=getattr(config, "best_weights_path", os.path.join(config.output_dir, "best_weights.weights.h5")),
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=True,   
        mode=mode,
        verbose=1,
    )
    return best_cb, best_weight_cb

class ConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(
        self, 
        X_val, 
        y_val, 
        w_val_raw, 
        out_dir, 
        total_events=None, 
        every_n_epochs=5, 
        class_names=None
    ):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.w_val_raw = w_val_raw
        self.out_dir = out_dir
        self.total_events = total_events
        self.every_n = every_n_epochs
        self.class_names = class_names
        os.makedirs(out_dir, exist_ok=True)

        w_proc_raw = self.w_val_raw["process_head"]
        
        if w_proc_raw.ndim == 2 and w_proc_raw.shape[1] > 1:
            w_raw_flat = np.sum(w_proc_raw, axis=1)
        else:
            w_raw_flat = np.asarray(w_proc_raw).flatten()

        if self.total_events is not None:
            self.weights_physical = scale_sample_weight_to_luminosity(w_raw_flat, self.total_events)
        else:
            self.weights_physical = w_raw_flat

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return

        # Inference
        preds = self.model.predict(self.X_val, verbose=0)
        
        # Ground truth e predizioni per la testa multiclass
        y_true = np.argmax(self.y_val["process_head"], axis=1)
        y_pred = np.argmax(preds["process_head"], axis=1)

        # Matrice di confusione calcolata sui PESI FISICI REALI
        cm = confusion_matrix(y_true, y_pred, sample_weight=self.weights_physical)
        
        # Salvataggio CSV dei conteggi fisici
        csv_path = os.path.join(self.out_dir, f"cm_physical_epoch_{epoch+1}.csv")
        np.savetxt(csv_path, cm, delimiter=",", fmt="%0.6f")

        # Normalizzazione per riga (Efficienza)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm_row = cm.astype(float) / row_sums

        # Normalizzazione per colonna (Purezza)
        col_sums = cm.sum(axis=0, keepdims=True) + 1e-12
        cm_col = cm.astype(float) / col_sums

        # Plot grafico
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        im0 = axes[0].imshow(cm_row, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title(f"Efficienza Fisica per classe (epoch {epoch+1})")
        
        im1 = axes[1].imshow(cm_col, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title(f"Purezza Fisica per classe (epoch {epoch+1})")

        n_classes = cm.shape[0]
        labels = self.class_names if self.class_names is not None else [str(i) for i in range(n_classes)]
        
        for ax in axes:
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        png_path = os.path.join(self.out_dir, f"cm_physical_epoch_{epoch+1}.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

def build_model(
    input_dim: int,
    num_classes: int,
    config: Config,
) -> keras.Model:
    
    regularizer = keras.regularizers.l2(config.weight_decay) if config.weight_decay and config.weight_decay > 0 else None

    inputs = keras.Input(shape=(input_dim,), name="features")

    # ==========================================
    # 1. SHARED BACKBONE (opzionale)
    # ==========================================
    if len(config.hidden_layers) > 0:
        x = inputs
        for i, units in enumerate(config.hidden_layers):
            x = Dense(
                units,
                kernel_regularizer=regularizer,
                name=f"dense_shared_{i}",
            )(x)
            if config.use_batch_norm:
                x = keras.layers.BatchNormalization(name=f"bn_shared_{i}")(x)
            elif config.use_layer_norm:
                x = keras.layers.LayerNormalization(name=f"ln_shared_{i}")(x)
            x = Activation("relu", name=f"act_shared_{i}")(x)
            if config.use_dropout:
                x = Dropout(config.dropout_rate, name=f"dropout_shared_{i}")(x)

        shared_features = x
    else:
        shared_features = inputs

    # ==========================================
    # 2. BRANCH SPECIALIZZATO: Testa Binaria (S/B)
    # ==========================================
    x_bin = shared_features  # Inizializzazione della catena
    if len(config.hidden_layers_sb) > 0:
        for i_bin, units_bin in enumerate(config.hidden_layers_sb):
            x_bin = Dense(
                units_bin,
                kernel_regularizer=regularizer,
                name=f"dense_bin_{i_bin}",
            )(x_bin)
            
            if config.use_batch_norm:
                x_bin = keras.layers.BatchNormalization(name=f"bn_bin_{i_bin}")(x_bin)
            elif config.use_layer_norm:
                x_bin = keras.layers.LayerNormalization(name=f"ln_bin_{i_bin}")(x_bin)
            x_bin = Activation("relu", name=f"act_bin_{i_bin}")(x_bin)
            if config.use_dropout:
                x_bin = Dropout(config.dropout_rate, name=f"dropout_bin_{i_bin}")(x_bin)
        
    out_bin = keras.layers.Dense(1, activation="sigmoid", name="sb_head")(x_bin)

    # ==========================================
    # 3. BRANCH SPECIALIZZATO: Testa Multiclasse (Segnali)
    # ==========================================
    x_multi = shared_features  
    if len(config.hidden_layers_multi) > 0:
        for i_multi, units_multi in enumerate(config.hidden_layers_multi):
            x_multi = Dense(
                units_multi,
                kernel_regularizer=regularizer,
                name=f"dense_multi_{i_multi}",
            )(x_multi)  
            
            if config.use_batch_norm:
                x_multi = keras.layers.BatchNormalization(name=f"bn_multi_{i_multi}")(x_multi)
            elif config.use_layer_norm:
                x_multi = keras.layers.LayerNormalization(name=f"ln_multi_{i_multi}")(x_multi)
            x_multi = Activation("relu", name=f"act_multi_{i_multi}")(x_multi)
            if getattr(config, "use_residual_in_multi", False) and (i_multi%2 == 0):
                x_multi = residual_block(x_multi, units_multi, name_prefix=f"res_multi_{i_multi}", use_layer_norm=config.use_layer_norm)
            if config.use_dropout:
                x_multi = Dropout(config.dropout_rate, name=f"dropout_multi_{i_multi}")(x_multi)

    out_multi = keras.layers.Dense(num_classes, activation="softmax", name="process_head")(x_multi)

    # ==========================================
    # 4. ASSEMBLAGGIO E COMPILAZIONE
    # ==========================================
    if config.train_stage == "sb_only":
        model = keras.Model(
            inputs=inputs,
            outputs=out_bin,
            name="vbs_cascaded_mlp",
        )
    elif config.train_stage == "multi_only":
        model = keras.Model(
            inputs=inputs,
            outputs=out_multi,
            name="vbs_cascaded_mlp",
        )
    else:
        model = keras.Model(
            inputs=inputs,
            outputs={"sb_head": out_bin, "process_head": out_multi},
            name="vbs_cascaded_mlp",
        )

    model.summary()
    return model


# -----------------------------
# End-to-end training and evaluation
# -----------------------------
def prepare_data(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    weight_column: str,
    config: Config,
) -> Tuple[np.ndarray, ...]:
    X = df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y = df.loc[:, label_column].to_numpy()
    w = df.loc[:, weight_column].to_numpy(dtype=np.float64)

    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = make_train_val_test_split(
        X=X,
        y=y,
        w=w,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        stratify=True,
    )

    if config.bootstrap:
        X_train, y_train, w_train = bootstrap_resample(X_train, y_train, w_train, random_state=config.random_state)

    preproc = fit_preprocessor(X_train, y_train)

    X_train_s = preproc.transform_x(X_train)
    X_val_s = preproc.transform_x(X_val)
    X_test_s = preproc.transform_x(X_test)

    y_train_dict = preproc.transform_y(y_train)
    y_val_dict = preproc.transform_y(y_val)
    y_test_dict = preproc.transform_y(y_test)

    def get_w_dict(w_raw, y_dict):
        sb_labels = y_dict["sb_head"].flatten()
        is_signal = sb_labels == 1
        process_labels = np.argmax(y_dict["process_head"], axis=1) 


        if config.reweight_for_sb_head:
            w_sb_pre = w_raw.copy()
            if np.any(is_signal):
                w_sb_pre[is_signal] = normalize_weights(
                    w_raw[is_signal], y=process_labels[is_signal], mode="per_class_sublinear", alpha=config.norm_pre_weight_sb_alpha
                )
            w_sb = normalize_weights(w_sb_pre, y=sb_labels, mode=config.norm_weights_sb, alpha=config.norm_weight_sb_alpha)
        else:
            w_sb = normalize_weights(w_raw, y=sb_labels, mode=config.norm_weights_sb, alpha=config.norm_weight_sb_alpha)

        w_proc = w_raw.copy()
        w_proc[~is_signal] = 0.0
        w_proc = normalize_weights(w_proc, y=process_labels, mode=config.norm_weight, alpha=config.norm_weight_sb_alpha)

        return {"sb_head": w_sb.astype(np.float32), "process_head": w_proc.astype(np.float32)}

    def get_w_raw_dict(w_raw, y_dict):
        w_sb = w_raw.astype(np.float32)
        w_proc = w_raw.copy()
        bg_mask = y_dict["sb_head"].flatten() == 0
        w_proc[bg_mask] = 0.0
        return {"sb_head": w_sb, "process_head": w_proc.astype(np.float32)}

    w_train_dict = get_w_dict(w_train, y_train_dict)
    w_val_dict = get_w_dict(w_val, y_val_dict)
    w_test_dict = get_w_dict(w_test, y_test_dict)

    w_train_raw_dict = get_w_raw_dict(w_train, y_train_dict)
    w_val_raw_dict = get_w_raw_dict(w_val, y_val_dict)
    w_test_raw_dict = get_w_raw_dict(w_test, y_test_dict)

    print("reweight_for_sb_head: ", config.reweight_for_sb_head)
    print("config: ", config.norm_weights_sb)
    print("Rapporto peso max/min: ", (w_train_dict["sb_head"].max() / w_train_dict["sb_head"].min()))
    print("Pesi raw per binario: ", w_train_raw_dict["sb_head"])
    print("Pesi rinormalizzati per binario: ", w_train_dict["sb_head"])
    sig_w = w_train_dict["sb_head"][y_train_dict["sb_head"].flatten() == 1]
    bkg_w = w_train_dict["sb_head"][y_train_dict["sb_head"].flatten() == 0]
    print("Rapporto max/min DENTRO il segnale (governato da norm_pre_weight_sb_alpha):", sig_w.max() / sig_w.min())
    print("Rapporto segnale-tot / fondo-tot (governato da norm_weight_sb_alpha):", sig_w.sum() / bkg_w.sum())

    return (
        X_train_s, X_val_s, X_test_s,
        y_train_dict, y_val_dict, y_test_dict,
        w_train_dict, w_val_dict, w_test_dict,
        w_train_raw_dict, w_val_raw_dict, w_test_raw_dict,
        preproc,
    )

def train_model(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    w_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    w_val: Dict[str, np.ndarray],
    w_val_raw: Dict[str, np.ndarray],
    total_events: int,
    config: Config,
    extra_callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    os.makedirs(config.output_dir, exist_ok=True)

    num_classes = y_train["process_head"].shape[1]
    if config.use_warmup:
        steps_per_epoch = max(1, math.ceil(X_train.shape[0] / max(1, config.batch_size)))
        total_steps = steps_per_epoch * max(1, config.epochs)
        decay_steps = max(total_steps, int(config.warmup_steps * 10), 1000)
        if config.scheduler_type != "CosineAnnealing":
            print(f"[WARN] use_warmup=True: bypassing parser scheduler '{config.scheduler_type}' and using warmup+cosine schedule.")
    else:
        decay_steps = None

    if config.train_stage != "two_stage":
        model = build_model(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            config=config,
        )

        lr_tracker = BatchLRTracker()
        callbacks = build_callbacks(config, lr_tracker=lr_tracker)
        cm_dir = os.path.join(config.output_dir, "confusion_matrices_physical")
        class_names = getattr(config, "class_names", None)  # opzionale
        
        cm_cb = ConfusionMatrixCallback(
            X_val=X_val, 
            y_val=y_val, 
            w_val_raw=w_val_raw,             
            out_dir=cm_dir, 
            total_events=total_events,        
            every_n_epochs=5, 
            class_names=class_names,
        )
        callbacks.append(cm_cb)

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

    def compute_class_weights_binary(y, pos_multiplier=1.0):
        total = len(y)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w_pos = (total / (2.0 * n_pos)) * pos_multiplier
        w_neg = total / (2.0 * n_neg)
        return {0: w_neg, 1: w_pos}

    def freeze_layers_by_prefix(model: keras.Model, prefixes: Sequence[str]):
            for layer in model.layers:
                if any(layer.name.startswith(p) for p in prefixes):
                    layer.trainable = False

    sample_weight_process = w_train["process_head"]
    val_sample_weight_process = w_val["process_head"]

    if getattr(config, "use_class_weight_sb", False):
        cw = compute_class_weights_binary(
            y_train["sb_head"].flatten(),
            pos_multiplier=getattr(config, "sb_pos_weight_multiplier", 1.0),
        )
        y_sb = y_train["sb_head"].flatten()
        sample_weight_sb = np.array([cw[int(label)] for label in y_sb], dtype=float)
        val_sample_weight_sb = w_val_raw["sb_head"]
    else:
        sample_weight_sb = w_train["sb_head"].astype(np.float64)
        val_sample_weight_sb = w_val_raw["sb_head"]
    sample_weight_fit = {
        "sb_head": sample_weight_sb,
        "process_head": sample_weight_process,
    }
    validation_sample_weight = {
        "sb_head": val_sample_weight_sb,
        "process_head": val_sample_weight_process,
    }

    # ---------- Pretrain SB function ----------
    opt, losses = compiler_utils(config=config, decay_steps=decay_steps)

    def pretrain_sb(model_local):
        output_names = model_local.output_names
        if len(output_names) == 1:
            loss_for_compile = losses["sb_head"]
            loss_weights_for_compile = None
        else:
            loss_for_compile = {"sb_head": losses["sb_head"], "process_head": losses["process_head"]}
            loss_weights_for_compile = {"sb_head": 1.0, "process_head": 0.0}

        # compile
        if loss_weights_for_compile is None:
            model_local.compile(optimizer=opt, loss=loss_for_compile, metrics=make_metrics()["sb_head"], weighted_metrics=make_metrics()["sb_head"])
        else:
            model_local.compile(optimizer=opt, loss=loss_for_compile, loss_weights=loss_weights_for_compile, metrics=make_metrics()["sb_head"], weighted_metrics=make_metrics()["sb_head"])

        # build callbacks now that model.metrics_names exists
        local_callbacks = build_callbacks(config, lr_tracker=lr_tracker)

        # prepara y e sample_weight coerenti con la struttura di output
        y_fit, sw_fit = _align_outputs_for_model(model_local, y_train, sample_weight_fit)
        val_y, val_sw = _align_outputs_for_model(model_local, y_val, validation_sample_weight)

        # fit (usa pretrain_batch_size se presente)
        batch_pre = getattr(config, "pretrain_batch_size", None) or config.batch_size
        epochs_pre = getattr(config, "pretrain_epochs", max(1, config.epochs // 4))

        history = model_local.fit(
            X_train,
            y_fit,
            sample_weight=sw_fit,
            validation_data=(X_val, val_y, val_sw),
            batch_size=batch_pre,
            epochs=epochs_pre,
            callbacks=local_callbacks,
            verbose=2,
        )

        # salva i pesi pretrain (utile per two_stage)
        pretrained_path = getattr(config, "pretrained_weights_path", os.path.join(config.output_dir, "pretrained_sb_weights.weights.h5"))
        os.makedirs(os.path.dirname(pretrained_path) or ".", exist_ok=True)
        model_local.save_weights(pretrained_path)
        print(f"[INFO] Pesi pretrain SB salvati in {pretrained_path}")

        return history, pretrained_path

    # ---------- Main flow ----------
    train_stage = getattr(config, "train_stage", "both")

    # ---------- Branch sb_only ----------
    if train_stage == "sb_only":
        print("[FLOW] Eseguo solo train binario (sb_only).")

        output_names = model.output_names
        if len(output_names) == 1:
            model.compile(
                optimizer=opt,
                loss=losses["sb_head"],
                metrics=make_metrics()["sb_head"], weighted_metrics=make_metrics()["sb_head"],
            )
        else:
            model.compile(
                optimizer=opt,
                loss={"sb_head": losses["sb_head"], "process_head": losses["process_head"]},
                loss_weights={"sb_head": config.loss_weight_sb, "process_head": 0.0},
                metrics=make_metrics(), weighted_metrics=make_metrics(),
            )

        best_cb, best_weight_cb = extend_callbacks(config=config, model=model)

        local_callbacks = build_callbacks(config, lr_tracker=lr_tracker)

        local_callbacks.append(best_cb)
        local_callbacks.append(best_weight_cb)

        if extra_callbacks:
            local_callbacks.extend(extra_callbacks)

        y_fit, sw_fit = _align_outputs_for_model(model, y_train, sample_weight_fit)
        val_y, val_sw = _align_outputs_for_model(model, y_val, validation_sample_weight)

        # fit
        history = model.fit(
            X_train,
            y_fit,
            sample_weight=sw_fit,
            validation_data=(X_val, val_y, val_sw),
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=local_callbacks,
            verbose=2,
        )

        try:
            plot_detailed_lr(lr_tracker, config.output_dir)
        except Exception:
            pass

        os.makedirs(config.output_dir, exist_ok=True)
        model.save(os.path.join(config.output_dir, "final_model_sb.keras"))
        model.save_weights(os.path.join(config.output_dir, "final_model_sb.weights.h5"))

        return model, history

    elif train_stage == "multi_only":
        print("[FLOW] Eseguo solo train multiclass (multi_only).")

        output_names = model.output_names
        if len(output_names) == 1:
            # single-output: probabilmente 'process_head'
            model.compile(
                optimizer=opt,
                loss=losses["process_head"],
                metrics=make_metrics()["process_head"], weighted_metrics=make_metrics()["process_head"],
            )
        else:
            # multi-output: assicuriamoci di dare peso solo a process_head
            model.compile(
                optimizer=opt,
                loss={"sb_head": losses["sb_head"], "process_head": losses["process_head"]},
                loss_weights={"sb_head": 0.0, "process_head": config.loss_weight_multi},
                metrics=make_metrics(), weighted_metrics=make_metrics(),
            )

        y_fit, sw_fit = _align_outputs_for_model(model, y_train, sample_weight_fit)
        val_y, val_sw = _align_outputs_for_model(model, y_val, validation_sample_weight)

        best_cb, best_weight_cb = extend_callbacks(config=config, model=model)

        local_callbacks = build_callbacks(config, lr_tracker=lr_tracker)

        local_callbacks.append(best_cb)
        local_callbacks.append(best_weight_cb)

        if extra_callbacks:
            local_callbacks.extend(extra_callbacks)

        history = model.fit(
            X_train,
            y_fit,
            sample_weight=sw_fit,
            validation_data=(X_val, val_y, val_sw),
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=local_callbacks,
            verbose=2,
        )

        # opzionale: plot LR
        try:
            plot_detailed_lr(lr_tracker, config.output_dir)
        except Exception:
            pass

        # salva modello e pesi
        os.makedirs(config.output_dir, exist_ok=True)
        model.save(os.path.join(config.output_dir, "final_model_multi.keras"))
        model.save_weights(os.path.join(config.output_dir, "final_model_multi.weights.h5"))

        return model, history

    # ---------- Branch two_stage (pretrain SB -> train MULTI) ----------
    # Per avere senso dovrei implementare possibilità di modificare pesi e loss in pretrain
    # Oppure modifica per fare le teste separate e vengono pretrainati 2 volte con 2 weight/loss diversi solo i pesi della backbone
    elif train_stage == "two_stage":
        print("[FLOW] Eseguo two_stage: pretrain SB -> train MULTI")

        # infer input_dim and num_classes from training arrays
        input_dim = X_train.shape[1]
        try:
            num_classes = y_train["process_head"].shape[1]
        except Exception:
            try:
                num_classes = y_val["process_head"].shape[1]
            except Exception:
                raise RuntimeError("Impossibile inferire num_classes: assicurati che y_train/y_val contengano 'process_head' one-hot arrays")

        lr_tracker = BatchLRTracker()
        
        # -------------------------
        # 1) Pretrain SB
        # -------------------------
        print("[STAGE A] Costruisco modello per pretrain SB")
        model_sb = build_model(input_dim=input_dim, num_classes=num_classes, config=config)

        print("[STAGE A] Avvio pretraining SB")
        history_pre, pretrained_path = pretrain_sb(model_sb)

        # salva final model e pesi SB
        os.makedirs(config.output_dir, exist_ok=True)
        try:
            model_sb.save(os.path.join(config.output_dir, "final_model_sb.keras"))
            model_sb.save_weights(os.path.join(config.output_dir, "final_model_sb.weights.h5"))
            print(f"[INFO] SB final model saved to {config.output_dir}")
        except Exception as e:
            print(f"[WARN] Salvataggio modello SB fallito: {e}")

        # -------------------------
        # 2) Train MULTI from pretrained SB weights
        # -------------------------
        print("[STAGE B] Costruisco modello per training MULTI")
        model_multi = build_model(input_dim=input_dim, num_classes=num_classes, config=config)

        # Carica pesi pretrain (by_name, skip_mismatch) se esistono
        if os.path.exists(pretrained_path):
            try:
                model_multi.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
                print(f"[INFO] Caricati pesi pretrain da {pretrained_path} (by_name=True, skip_mismatch=True)")
            except Exception as e:
                print(f"[WARN] Errore caricamento pesi pretrain: {e}. Proseguo comunque.")
        else:
            print(f"[WARN] Nessun file pretrain trovato in {pretrained_path}. Procedo comunque.")

        # controllo rapido sui layer caricati (opzionale, utile per debug)
        try:
            _check_loaded_layers(model_multi, prefixes=["dense_shared_", "dense_bin_"])
        except Exception:
            pass

        # congela i layer secondo config
        prefixes = []
        if getattr(config, "freeze_shared_after_pretrain", True):
            prefixes += ["dense_shared_", "bn_shared_", "ln_shared_", "act_shared_", "dropout_shared_"]
        if getattr(config, "freeze_sb_after_pretrain", True):
            prefixes += ["dense_bin_", "bn_bin_", "ln_bin_", "act_bin_", "dropout_bin_", "sb_head"]

        freeze_layers_by_prefix(model_multi, prefixes)

        # ricompilazione: vogliamo allenare solo process_head (sb_head peso 0 se presente)
        output_names = model_multi.output_names
        if len(output_names) == 1:
            loss_for_compile = losses["process_head"]
            loss_weights_for_compile = None
        else:
            loss_for_compile = {"sb_head": losses["sb_head"], "process_head": losses["process_head"]}
            loss_weights_for_compile = {"sb_head": 0.0, "process_head": 1.0}

        # usa fine_tune_lr se presente
        opt_multi = opt

        if loss_weights_for_compile is None:
            model_multi.compile(optimizer=opt_multi, loss=loss_for_compile, metrics=make_metrics(), weighted_metrics=make_metrics(),)
        else:
            model_multi.compile(optimizer=opt_multi, loss=loss_for_compile, loss_weights=loss_weights_for_compile, metrics=make_metrics(), weighted_metrics=make_metrics(),)

        # build callbacks for multi (now that model_multi is compiled)
        callbacks_multi = build_callbacks(config, lr_tracker=lr_tracker)

        # prepara y e sample_weight coerenti con la struttura di output
        # se process sample_weight è matrice, convertila a vettore prima di passare
        if "process_head" in sample_weight_fit:
            sample_weight_fit["process_head"] = _sw_to_vector(sample_weight_fit["process_head"])
        if "process_head" in validation_sample_weight:
            validation_sample_weight["process_head"] = _sw_to_vector(validation_sample_weight["process_head"])

        y_fit_multi, sw_fit_multi = _align_outputs_for_model(model_multi, y_train, sample_weight_fit)
        val_y_multi, val_sw_multi = _align_outputs_for_model(model_multi, y_val, validation_sample_weight)

        print("[STAGE B] Avvio training MULTI")
        history_multi = model_multi.fit(
            X_train,
            y_fit_multi,
            sample_weight=sw_fit_multi,
            validation_data=(X_val, val_y_multi, val_sw_multi),
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=callbacks_multi,
            verbose=2,
        )

        # salva final multi model e pesi
        try:
            model_multi.save(os.path.join(config.output_dir, "final_model_multi.keras"))
            model_multi.save_weights(os.path.join(config.output_dir, "final_model_multi.weights.h5"))
            print(f"[INFO] MULTI final model saved to {config.output_dir}")
        except Exception as e:
            print(f"[WARN] Salvataggio modello MULTI fallito: {e}")

        # ritorna modello multi e le history
        return model_multi, {"pretrain_sb": history_pre, "train_multi": history_multi}

    else:
        # default: comportamento originale (allenamento multitask simultaneo)
        print("[FLOW] Eseguo training multitask simultaneo (default 'both').")
        # compile with original loss weights and optimizer
        model.compile(
            optimizer=opt,
            loss=losses,
            loss_weights={"sb_head": config.loss_weight_sb, "process_head": config.loss_weight_multi},
            metrics=make_metrics(), weighted_metrics=make_metrics(),
        )

        y_fit, sw_fit = _align_outputs_for_model(model, y_train, sample_weight_fit)
        val_y, val_sw = _align_outputs_for_model(model, y_val, validation_sample_weight)

        history = model.fit(
            X_train,
            y_fit,
            sample_weight=sw_fit,
            validation_data=(X_val, val_y, val_sw),
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

    plot_detailed_lr(lr_tracker, config.output_dir)
    model.save(os.path.join(config.output_dir, "final_model.keras"))
    model.save_weights(os.path.join(config.output_dir, "final_model.weights.h5"))
    return model, history

def evaluate_cascaded_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
    w_test: Dict[str, np.ndarray],
    w_test_raw: Dict[str, np.ndarray],
    history: Optional[keras.callbacks.History],
    config: Config,
    total_events: int,
    cascade_threshold: float = 0.85,
) -> Dict[str, float]:
    # 1. Grafico dell'History di addestramento
    if history is not None:
        plot_history(history, config.output_dir)

    # 2. Forward pass sul dataset di test
    predictions = model.predict(X_test, batch_size=config.batch_size, verbose=0)
    if isinstance(predictions, dict):
        y_pred_probs_dict = predictions
    else:
        y_pred_probs_dict = dict(zip(model.output_names, predictions))

    # 3. Disegno delle Confusion Matrix (include la 2x2, la 5x5 e la nuova 6x6 Cascaded)
    plot_multitask_confusion_matrices(
        y_true_dict=y_test,
        y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test,
        w_raw_dict=w_test_raw,
        signal_class_names=PROCESS_CLASS_NAMES,
        output_dir=config.output_dir,
        total_events=total_events,
        cascade_threshold=cascade_threshold,
    )

    # 4. Disegno delle curve ROC per entrambe le teste
    plot_multitask_roc_curves(
        y_true_dict=y_test,
        y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test,
        signal_class_names=PROCESS_CLASS_NAMES,
        output_dir=config.output_dir,
    )

    # 5. Disegno del plot a cascata: predizione finale con 5 segnali + Fondo
    y_true_sb = y_test["sb_head"].flatten()
    y_true_process_int = np.argmax(y_test["process_head"], axis=1)
    y_pred_process_int = np.argmax(y_pred_probs_dict["process_head"], axis=1)

    bkg_class_idx = len(PROCESS_CLASS_NAMES)
    y_true_cascaded = np.where(y_true_sb == 1, y_true_process_int, bkg_class_idx)
    y_pred_cascaded = np.where(
        y_pred_probs_dict["sb_head"].flatten() >= cascade_threshold,
        y_pred_process_int,
        bkg_class_idx,
    )
 
    plot_prediction_composition(
        y_true=y_true_cascaded,
        y_pred=y_pred_cascaded,
        class_names=list(PROCESS_CLASS_NAMES) + ["Fondo Totale"],
        output_dir=config.output_dir,
        filename_prefix="prediction_composition",
        sample_weight=w_test["sb_head"].flatten(),
        sample_weight_raw=w_test_raw["sb_head"].flatten(),
        total_events=total_events,
        title_suffix=f"(Cascata con P(S/B) >= {cascade_threshold})"
    )

    report_path = os.path.join(config.output_dir, "analysis_report.txt")

    # 5. Analisi della Signal Region per la testa binaria (S/B)
    analyze_binary_signal_region(
        y_true=y_test["sb_head"].flatten(),
        y_prob_sig=y_pred_probs_dict["sb_head"].flatten(),
        sample_weight=w_test["sb_head"].flatten(),
        report_path=report_path,
    )

    # 6. Analisi della Signal Region A CASCATA (Attivazione del codice precedentemente non usato)
    y_test_process_int = np.argmax(y_test["process_head"], axis=1)
    analyze_cascaded_signal_region(
        y_true_multi_int=y_test_process_int,
        y_prob_bin=y_pred_probs_dict["sb_head"].flatten(),
        y_prob_multi=y_pred_probs_dict["process_head"],
        class_names=PROCESS_CLASS_NAMES,
        threshold=cascade_threshold,
        report_path=report_path,
    )

    # 7. Analisi individuale per singola classe di segnale
    for idx in range(len(PROCESS_CLASS_NAMES)):
        analyze_signal_region(
            y_true=y_test_process_int,
            y_probs=y_pred_probs_dict["process_head"],
            sample_weight=w_test["process_head"].flatten(),
            class_names=PROCESS_CLASS_NAMES,
            target_class_idx=idx,
            threshold=cascade_threshold,
            report_path=report_path,
        )

    w_test_physical = {
        head: scale_sample_weight_to_luminosity(w_test_raw[head], total_events)
        for head in w_test_raw
    }

    physical_dir = os.path.join(config.output_dir, "physical")

    plot_multitask_confusion_matrices(
        y_true_dict=y_test,
        y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test_physical,
        w_raw_dict=w_test_raw,
        signal_class_names=PROCESS_CLASS_NAMES,
        output_dir=physical_dir,
        total_events=total_events,
        cascade_threshold=cascade_threshold,
    )

    plot_multitask_roc_curves(
        y_true_dict=y_test,
        y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test_physical,
        signal_class_names=PROCESS_CLASS_NAMES,
        output_dir=physical_dir,
    )

    plot_prediction_composition(
        y_true=y_true_cascaded,
        y_pred=y_pred_cascaded,
        class_names=list(PROCESS_CLASS_NAMES) + ["Fondo Totale"],
        output_dir=physical_dir,
        filename_prefix="prediction_composition",
        sample_weight=w_test_physical["sb_head"].flatten(),
        sample_weight_raw=w_test_raw["sb_head"].flatten(),
        total_events=total_events,
        title_suffix=f"(Cascata con P(S/B) >= {cascade_threshold})"
    )

    analyze_binary_signal_region(
        y_true=y_test["sb_head"].flatten(),
        y_prob_sig=y_pred_probs_dict["sb_head"].flatten(),
        sample_weight=w_test_physical["sb_head"].flatten(),
        report_path=os.path.join(physical_dir, "analysis_report.txt"),
    )

    for idx in range(len(PROCESS_CLASS_NAMES)):
        analyze_signal_region(
            y_true=y_test_process_int,
            y_probs=y_pred_probs_dict["process_head"],
            sample_weight=w_test_physical["process_head"].flatten(),
            class_names=PROCESS_CLASS_NAMES,
            target_class_idx=idx,
            threshold=cascade_threshold,
            report_path=os.path.join(physical_dir, "analysis_report.txt"),
        )

    # 8. Calcolo e restituzione delle metriche globali pesate tramite Keras
    evaluation = model.evaluate(X_test, y_test, sample_weight=w_test, verbose=0)
    metric_names = model.metrics_names
    return dict(zip(metric_names, evaluation))

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
    w_test: Dict[str, np.ndarray],
    w_test_raw: Dict[str, np.ndarray],
    history: Optional[keras.callbacks.History],
    config: Config,
    total_events: int,
    cascade_threshold: float = 0.85,
) -> Dict[str, float]:
    output_names = model.output_names
    if len(output_names) != 1:
        raise ValueError(
            f"evaluate_model si aspetta un modello a singola uscita "
            f"(sb_only o multi_only); trovate {len(output_names)} uscite: "
            f"{output_names}. Per un modello a due teste usa "
            f"evaluate_cascaded_model."
        )
    head = output_names[0]
    if head not in ("sb_head", "process_head"):
        raise ValueError(f"Uscita del modello non riconosciuta: '{head}'")

    # 1. Grafico dell'History di addestramento
    if history is not None:
        plot_history(history, config.output_dir)

    # 2. Forward pass sul dataset di test
    predictions = model.predict(X_test, batch_size=config.batch_size, verbose=0)
    y_pred_probs = predictions[head] if isinstance(predictions, dict) else predictions

    report_path = os.path.join(config.output_dir, "analysis_report.txt")
    physical_dir = os.path.join(config.output_dir, "physical")
    os.makedirs(physical_dir, exist_ok=True)
    physical_report_path = os.path.join(physical_dir, "analysis_report.txt")

    w_test_physical = {
        h: scale_sample_weight_to_luminosity(w_test_raw[h], total_events)
        for h in w_test_raw
    }

    if head == "sb_head":
        y_true_sb = y_test["sb_head"].flatten()
        y_prob_sb = y_pred_probs.flatten()
        y_pred_sb = (y_prob_sb >= cascade_threshold).astype(int)
        class_names_sb = ["Fondo Totale", "Segnale VBS"]

        # 3. Confusion matrix 2x2 - training-diagnostic + physical
        plot_conf_matrix(
            y_true=y_true_sb, y_pred=y_pred_sb,
            class_names=class_names_sb,
            output_dir=config.output_dir, filename="confusion_matrix_sb_head.png",
            sample_weight=w_test["sb_head"].flatten(),
            sample_weight_raw=w_test_raw["sb_head"].flatten(),
            total_events=total_events,
            threshold_info=f"(Taglio P(S/B) >= {cascade_threshold})",
        )
        plot_conf_matrix(
            y_true=y_true_sb, y_pred=y_pred_sb,
            class_names=class_names_sb,
            output_dir=physical_dir, filename="confusion_matrix_sb_head.png",
            sample_weight=w_test_physical["sb_head"].flatten(),
            sample_weight_raw=w_test_raw["sb_head"].flatten(),
            total_events=total_events,
            threshold_info=f"(Taglio P(S/B) >= {cascade_threshold})",
        )

        # 4. ROC binaria - training-diagnostic + physical
        plot_roc_binary(
            y_true=y_true_sb, y_prob=y_prob_sb, output_dir=config.output_dir,
            sample_weight=w_test["sb_head"].flatten(),
        )
        plot_roc_binary(
            y_true=y_true_sb, y_prob=y_prob_sb, output_dir=physical_dir,
            sample_weight=w_test_physical["sb_head"].flatten(),
        )
        analyze_binary_signal_region(
            y_true=y_true_sb, y_prob_sig=y_prob_sb,
            sample_weight=w_test["sb_head"].flatten(),
            threshold=cascade_threshold, report_path=report_path,
        )
        analyze_binary_signal_region(
            y_true=y_true_sb, y_prob_sig=y_prob_sb,
            sample_weight=w_test_physical["sb_head"].flatten(),
            threshold=cascade_threshold, report_path=physical_report_path,
        )

    else:  # head == "process_head"
        is_true_signal = y_test["sb_head"].flatten() == 1
        if not np.any(is_true_signal):
            raise ValueError(
                "Nessun evento di vero segnale nel test set: impossibile "
                "valutare process_head (le sue etichette esistono solo per "
                "il vero segnale)."
            )

        y_true_proc_onehot = y_test["process_head"][is_true_signal]
        y_prob_proc = y_pred_probs[is_true_signal]
        y_true_proc = np.argmax(y_true_proc_onehot, axis=1)
        y_pred_proc = np.argmax(y_prob_proc, axis=1)

        w_proc = w_test["process_head"][is_true_signal].flatten()
        w_proc_raw = w_test_raw["process_head"][is_true_signal].flatten()
        w_proc_phys = w_test_physical["process_head"][is_true_signal].flatten()

        # 3. Confusion matrix 5x5 - training-diagnostic + physical
        plot_conf_matrix(
            y_true=y_true_proc, y_pred=y_pred_proc,
            class_names=PROCESS_CLASS_NAMES,
            output_dir=config.output_dir, filename="confusion_matrix_process_head.png",
            sample_weight=w_proc, sample_weight_raw=w_proc_raw, total_events=total_events,
            threshold_info="(Solo veri eventi di segnale, nessun taglio S/B: testa sb_head assente)",
        )
        plot_conf_matrix(
            y_true=y_true_proc, y_pred=y_pred_proc,
            class_names=PROCESS_CLASS_NAMES,
            output_dir=physical_dir, filename="confusion_matrix_process_head.png",
            sample_weight=w_proc_phys, sample_weight_raw=w_proc_raw, total_events=total_events,
            threshold_info="(Solo veri eventi di segnale, nessun taglio S/B: testa sb_head assente)",
        )

        # 4. ROC multiclasse One-vs-Rest - training-diagnostic + physical
        plot_roc_multiclass(
            y_true_onehot=y_true_proc_onehot, y_prob=y_prob_proc,
            class_names=PROCESS_CLASS_NAMES, output_dir=config.output_dir,
            sample_weight=w_proc,
        )
        plot_roc_multiclass(
            y_true_onehot=y_true_proc_onehot, y_prob=y_prob_proc,
            class_names=PROCESS_CLASS_NAMES, output_dir=physical_dir,
            sample_weight=w_proc_phys,
        )

        plot_prediction_composition(
            y_true=y_true_proc, y_pred=y_pred_proc,
            class_names=list(PROCESS_CLASS_NAMES),
            output_dir=config.output_dir, filename_prefix="prediction_composition",
            sample_weight=w_proc, sample_weight_raw=w_proc_raw, total_events=total_events,
            title_suffix="(Solo veri eventi di segnale, nessuna cascata S/B)",
            bkg_label="__nessuna_classe_di_fondo__",
        )
        plot_prediction_composition(
            y_true=y_true_proc, y_pred=y_pred_proc,
            class_names=list(PROCESS_CLASS_NAMES),
            output_dir=physical_dir, filename_prefix="prediction_composition",
            sample_weight=w_proc_phys, sample_weight_raw=w_proc_raw, total_events=total_events,
            title_suffix="(Solo veri eventi di segnale, nessuna cascata S/B)",
            bkg_label="__nessuna_classe_di_fondo__",
        )

        # 6. Signal Region per singolo canale - training-diagnostic + physical
        for idx in range(len(PROCESS_CLASS_NAMES)):
            analyze_signal_region(
                y_true=y_true_proc, y_probs=y_prob_proc, sample_weight=w_proc,
                class_names=PROCESS_CLASS_NAMES, target_class_idx=idx,
                threshold=cascade_threshold, report_path=report_path,
            )
        for idx in range(len(PROCESS_CLASS_NAMES)):
            analyze_signal_region(
                y_true=y_true_proc, y_probs=y_prob_proc, sample_weight=w_proc_phys,
                class_names=PROCESS_CLASS_NAMES, target_class_idx=idx,
                threshold=cascade_threshold, report_path=physical_report_path,
            )

    evaluation = model.evaluate(
        X_test, y_test[head], sample_weight=w_test[head], verbose=0
    )
    if not isinstance(evaluation, (list, tuple)):
        evaluation = [evaluation]
    metric_names = model.metrics_names
    return dict(zip(metric_names, evaluation))

# -----------------------------
# Example main
# -----------------------------
def main() -> None:
    config = parse_args_to_config()
    os.makedirs(config.output_dir, exist_ok=True)
    config_filepath = os.path.join(config.output_dir, "run_config.txt")
    save_config_to_txt(config, config_filepath)

    feature_all = [
        "m", "dphi", "deta", "dr", "pt", "px", "py", "pz", "px_i", "px_j", "py_i", "py_j", "pz_i", "pz_j",
        "neutrinos_pt", "neutrinos_phi", "neutrinos_eta", "eta", "phi", 
        "pt_i", "pt_j", "phi_i", "phi_j", "eta_i", "eta_j", "pdg_i", "pdg_j"
    ]
    feature_raw = [
        "pt_i", "pt_j", "phi_i", "phi_j", "eta_i", "eta_j", "pdg_i", "pdg_j", "px_i", "px_j", "py_i", "py_j", "pz_i", "pz_j",
    ]
    feature_digerite = [
        "m", "dphi", "deta", "dr", "pt", "px", "py", "pz", "pdg_i", "pdg_j",
        "neutrinos_pt", "neutrinos_phi", "neutrinos_eta", "eta", "phi"
    ]
    feature_4chlep2nu = [
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
    ]
    feature_dict = {
        "all": feature_all,
        "raw": feature_raw,
        "digerite": feature_digerite,
        "4l2n": feature_4chlep2nu,
    }

    if len(config.input_files) == 2:
        df = load_root_to_dataframe_per2coppie(
            file_coppia1=config.input_files[0],
            file_coppia2=config.input_files[1],
            tree_name=config.tree_name,
            feature_branches=feature_dict[config.feature_set],
            label_branch=config.label_branch,
            weight_branch=config.weight_branch,
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
    else:
        df = load_root_to_dataframe(
            root_files=config.input_files,
            tree_name=config.tree_name,
            feature_branches=feature_dict[config.feature_set],
            label_branch=config.label_branch,
            weight_branch=config.weight_branch,
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )        

    if df.empty:
        raise ValueError("Il DataFrame è vuoto dopo il caricamento dei file ROOT.")

    df = run_feature_engineering(df)
  
    feature_columns = [c for c in df.columns if c not in [config.label_branch, config.weight_branch]]

    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test, w_train_raw, w_val_raw, w_test_raw, preproc = prepare_data(
        df=df,
        feature_columns=feature_columns,
        label_column=config.label_branch,
        weight_column=config.weight_branch,
        config=config,
    )

    total_events = len(df)

    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        w_val_raw=w_val_raw,
        total_events=total_events,
        config=config,
    )

    if len(model.output_names) == 1:
        results = evaluate_model(
            model=model, 
            X_test=X_test, 
            y_test=y_test, 
            w_test=w_test,
            w_test_raw=w_test_raw, 
            history=history, 
            config=config,
            total_events=total_events
        )
    else:
        results = evaluate_cascaded_model(
            model=model, 
            X_test=X_test, 
            y_test=y_test, 
            w_test=w_test,
            w_test_raw=w_test_raw, 
            history=history, 
            config=config,
            total_events=total_events
            )

    print("\nEvaluation metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()