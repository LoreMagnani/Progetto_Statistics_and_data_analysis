import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import uproot
import joblib
from tensorflow import keras
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURAZIONE
# =====================================================================
LUMINOSITY: float = 10_800_000.0  # pb^-1

SIGNAL_NAMES: List[str] = ["WW+H", "ZZ+H", "ZZ", "WZ", "WW"]

# kind_events -> indice canale di segnale (0..4, stesso ordine di SIGNAL_NAMES)
KIND_TO_SIGNAL_IDX: Dict[int, int] = {
    -5: 0,  # WW+H
    -4: 1,  # ZZ+H
    -3: 2,  # ZZ
    -2: 3,  # WZ
    -1: 4,  # WW
}
# kind_events che identificano il fondo.
BKG_KIND_VALUES: Sequence[int] = (1, 2)

# La testa multiclass ha 5 uscite "pure" (schema classico, cascata via
# sola soglia SB) o 6 uscite con una classe "Fondo" appresa
MULTICLASS_HAS_BKG_OUTPUT: bool = True


TREE_NAME: str = "Events" 
LABEL_BRANCH: str = "kind_events"
WEIGHT_BRANCH: str = "Weight_weight"

# Feature per la testa SB e per la testa multiclass,
RAW_FEATURE_BRANCHES: List[str] = [
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

@dataclass
class InputSample:
    """Un file ROOT di input con il proprio numero di eventi generati
    (serve a scalare weight_weight alla luminosità: processi diversi
    hanno statistiche di generazione diverse)."""
    path: str
    total_events: int

INPUT_SAMPLES: List[InputSample] = [
    InputSample(path="../rootfiles/4chlep2nu.root", total_events=300000),

]

SB_MODEL_PATH: str = "../ML_classifier/Modelli_migliori/4l/outputs_bktrain_1M_1/sb_only/best_model.keras"       
MULTI_MODEL_PATH: str = "../ML_classifier/Modelli_migliori/4l/outputs_bktrain_1M_1/multi_only/best_model.keras"  

PREPROCESSOR_PATH: str = "../ML_classifier/Modelli_migliori/4l/outputs_bktrain_1M_1/preprocessor.pkl"

CASCADE_THRESHOLD: float = 0.3
SIGNAL_REGION_THRESHOLD: float = 0.85  
OUTPUT_DIR: str = "output_classificazione_cascata"
PREDICTIONS_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, "predizioni_evento_per_evento.csv")


# =====================================================================
# CARICAMENTO DATI
# =====================================================================
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

    df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"Caricamento completato! Dimensioni finali dataset: {df.shape}\n")

    return df

def scale_sample_weight_to_luminosity(
    sample_weight: np.ndarray,
    total_events: int,
    luminosity: float = LUMINOSITY,
) -> np.ndarray:
    if total_events <= 0:
        raise ValueError("total_events must be positive")
    return sample_weight.astype(np.float64) / float(total_events) * float(luminosity)


# =====================================================================
# FEATURE ENGINEERING HELPERS
# =====================================================================
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
            or c.startswith("mt ")  # "mT <rango> nu"
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

        # "dphi" e' un prefisso/sottostringa sicuro qui: nessuna colonna phi
        # "pura" (gia' gestita da add_trig_phi_features) lo contiene, perche'
        # quelle iniziano per "phi", non per "dphi".
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

# Queste sono per il caso a 6l
def handle_missing_reconstruction(
    df: pd.DataFrame,
    exclude_columns: Sequence[str] = ("kind_events", "Weight_weight"),
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]

    nan_mask_any = df[numeric_cols].isna().any(axis=1)
    df["has_full_reco"] = (~nan_mask_any).astype(np.float32)

    if nan_mask_any.any():
        cols_with_nan = [c for c in numeric_cols if df[c].isna().any()]
        print(
            f"[feature engineering] Imputo con la media {len(cols_with_nan)} colonne per "
            f"{int(nan_mask_any.sum())} eventi senza ricostruzione completa "
            f"({100 * nan_mask_any.mean():.2f}%); aggiunta colonna 'has_full_reco'."
        )
        for col in cols_with_nan:
            df[col] = df[col].fillna(df[col].mean())

    return df

def run_feature_engineering_6l(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = handle_missing_reconstruction(df)

    phi_cols = [c for c in df.columns if c.startswith("phi")]
    if phi_cols:
        df = add_trig_phi_features(df, phi_cols)
        df = df.drop(columns=phi_cols)

    df = process_pdg_features(df)
    df = apply_sqrt_to_separations(df)
    df = scale_kinematic_features(df)
    df = transform_dphi_features(df)

    return df



# =====================================================================
# COSTRUZIONE DATASET + PREPROCESSOR
# =====================================================================
def load_dataset(samples: Sequence[InputSample]) -> pd.DataFrame:
    global MULTI_MODEL_PATH, PREPROCESSOR_PATH, SB_MODEL_PATH

    if not samples:
        raise ValueError("INPUT_SAMPLES è vuoto.")

    frames = []
    for sample in samples:

        df = load_root_to_dataframe(
            root_files=[sample.path],
            tree_name=TREE_NAME,
            feature_branches=RAW_FEATURE_BRANCHES,
            label_branch=LABEL_BRANCH,
            weight_branch=WEIGHT_BRANCH,
        )

        df = run_feature_engineering(df)

        df["peso_fisico"] = scale_sample_weight_to_luminosity(
            df[WEIGHT_BRANCH].to_numpy(), sample.total_events, LUMINOSITY
        )
        df["file_origine"] = sample.path
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)

def build_true_labels(df: pd.DataFrame) -> np.ndarray:
    kind = df[LABEL_BRANCH].to_numpy()
    n_signal = len(SIGNAL_NAMES)
    bkg_idx = n_signal

    y_true = np.full(kind.shape, fill_value=-1, dtype=int)
    for code, idx in KIND_TO_SIGNAL_IDX.items():
        y_true[kind == code] = idx
    for code in BKG_KIND_VALUES:
        y_true[kind == code] = bkg_idx

    n_unmapped = int(np.sum(y_true == -1))
    if n_unmapped > 0:
        codici_ignoti = sorted(set(kind[y_true == -1].tolist()))
        print(
            f"[ATTENZIONE] {n_unmapped} eventi hanno un kind_events non mappato "
            f"in KIND_TO_SIGNAL_IDX/BKG_KIND_VALUES: {codici_ignoti}. "
            "Verranno esclusi dalla valutazione."
        )
    return y_true

def extract_features_for_inference(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    feature_columns = [
        c for c in df.columns
        if c not in [LABEL_BRANCH, WEIGHT_BRANCH, "peso_fisico", "file_origine"]
        and (
            np.issubdtype(df[c].dtype, np.number)
            or np.issubdtype(df[c].dtype, np.bool_)
        )
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    return X, feature_columns

def apply_preprocessor(X: np.ndarray) -> np.ndarray:
    preprocessor_module_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ML_classifier/Codice/4l")
    )
    if preprocessor_module_dir not in sys.path:
        sys.path.insert(0, preprocessor_module_dir)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Controllo difensivo: se in futuro una colonna derivata (label,
    # predizione, ...) finisse per errore tra le feature PRIMA di questa
    # chiamata, qui esplode con un messaggio chiaro invece di un errore
    # sklearn criptico (o, peggio, un mismatch silenzioso se il conteggio
    # combaciasse per caso).
    n_expected = getattr(preprocessor.scaler, "n_features_in_", None)
    if n_expected is not None and X.shape[1] != n_expected:
        raise ValueError(
            f"X ha {X.shape[1]} colonne ma lo scaler salvato in "
            f"'{PREPROCESSOR_PATH}' si aspetta {n_expected} feature (quelle "
            "viste in training). Probabile causa: una colonna derivata "
            "(y_true_6, una predizione, ...) e' stata aggiunta al DataFrame "
            "PRIMA di extract_features_for_inference() ed e' stata raccolta "
            "per errore come se fosse una feature."
        )

    X_new = preprocessor.transform_x(X)
    return X_new


# =====================================================================
# MODELLI E INFERENZA A CASCATA
# =====================================================================
def load_models(sb_path: str, multi_path: str):
    sb_model = keras.models.load_model(sb_path)
    multi_model = keras.models.load_model(multi_path)
    return sb_model, multi_model

def run_cascade_inference(
    sb_model,
    multi_model,
    X_scaled: np.ndarray,
    threshold: float = CASCADE_THRESHOLD,
):
    n_signal = len(SIGNAL_NAMES)
    bkg_idx = n_signal

    # SB inference
    y_prob_sb = np.asarray(sb_model.predict(X_scaled, batch_size=8192)).reshape(-1)
    if y_prob_sb.ndim != 1:
        raise ValueError("La testa SB deve avere una sola uscita sigmoid.")

    # Multiclass inference
    y_prob_multi = np.asarray(multi_model.predict(X_scaled, batch_size=8192))
    n_outputs = y_prob_multi.shape[1]

    # Controllo difensivo
    if MULTICLASS_HAS_BKG_OUTPUT:
        if n_outputs != n_signal + 1:
            raise ValueError(
                f"Modello multiclass ha {n_outputs} output, attesi {n_signal+1}."
            )
        y_pred_multi_signal_idx = np.argmax(y_prob_multi[:, :n_signal], axis=1)
        y_pred_multi_full_idx = np.argmax(y_prob_multi, axis=1)
    else:
        if n_outputs != n_signal:
            raise ValueError(
                f"Modello multiclass ha {n_outputs} output, attesi {n_signal}."
            )
        y_pred_multi_signal_idx = np.argmax(y_prob_multi, axis=1)
        y_pred_multi_full_idx = y_pred_multi_signal_idx

    # Cascata
    y_pred_cascade = np.where(
        y_prob_sb >= threshold,
        y_pred_multi_full_idx,
        bkg_idx
    )

    return {
        "y_prob_sb": y_prob_sb,
        "y_prob_multi": y_prob_multi,
        "y_pred_multi_signal_idx": y_pred_multi_signal_idx,
        "y_pred_multi_full_idx": y_pred_multi_full_idx,
        "y_pred_cascade": y_pred_cascade,
    }


# =====================================================================
# FUNZIONI DI PLOT / ANALISI
# =====================================================================
def plot_conf_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: str,
    filename: str,
    sample_weight: Optional[np.ndarray] = None,
    threshold_info: str = ""
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)

    cm_true = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, normalize="true")
    cm_pred = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, normalize="pred")

    raw_true_counts = np.array([np.sum(y_true == i) for i in range(num_classes)], dtype=int)
    raw_pred_counts = np.array([np.sum(y_pred == i) for i in range(num_classes)], dtype=int)

    if sample_weight is not None:
        phys_true_counts = np.array([np.sum(sample_weight[y_true == i]) for i in range(num_classes)], dtype=float)
        phys_pred_counts = np.array([np.sum(sample_weight[y_pred == i]) for i in range(num_classes)], dtype=float)
    else:
        phys_true_counts = np.zeros(num_classes, dtype=float)
        phys_pred_counts = np.zeros(num_classes, dtype=float)

    row_labels = [
        f"{class_names[i]}\nraw={raw_true_counts[i]}\nfis={phys_true_counts[i]:.1f}"
        for i in range(num_classes)
    ]
    col_labels = [
        f"{class_names[i]}\nraw={raw_pred_counts[i]}\nfis={phys_pred_counts[i]:.1f}"
        for i in range(num_classes)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    title_true = "Normalizzata per Classe Vera\n(Efficienza / Righe = 1)"
    title_pred = "Normalizzata per Classe Predetta\n(Purità / Colonne = 1)"
    if threshold_info:
        title_true += f"\n{threshold_info}"
        title_pred += f"\n{threshold_info}"

    disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=class_names)
    disp_true.plot(ax=ax1, cmap="Blues", values_format=".3f", colorbar=True, xticks_rotation=45)
    ax1.set_title(title_true)
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))
    ax1.set_yticklabels(row_labels)
    ax1.set_xticklabels(col_labels, rotation=45, ha="right")

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

def analyze_binary_signal_region(
    y_true: np.ndarray,
    y_prob_sig: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float = 0.85,
    report_path: Optional[str] = None
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
    report_path: Optional[str] = None
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
    title_suffix: str = "",
    bkg_label: str = "Fondo Totale"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)

    cm_raw = confusion_matrix(y_true, y_pred)
    cm_abs = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    has_weights = sample_weight is not None

    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm_row = np.divide(cm_abs, row_sums, out=np.zeros_like(cm_abs, dtype=float), where=row_sums != 0)

    def _raw_phys_str(raw_value, phys_value=None) -> str:
        raw_str = f"raw={int(round(raw_value))}"
        if phys_value is None:
            return raw_str
        return f"{raw_str}, fis={phys_value:.1f}"

    signal_indices = [i for i, name in enumerate(class_names) if name != bkg_label]
    colors_true = plt.cm.tab10(np.linspace(0, 1, num_classes))
    x_positions = np.arange(num_classes)

    y_label_abs = "Eventi Fisici (Pesi scalati alla luminosità)" if has_weights else "Numero di Eventi Grezzi"

    for true_class_idx in signal_indices:
        true_name = class_names[true_class_idx]
        heights = cm_abs[true_class_idx, :]
        heights_raw = cm_raw[true_class_idx, :]
        row_integral = np.sum(heights)
        row_integral_raw = np.sum(heights_raw)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_positions, heights, color=colors_true[true_class_idx], alpha=0.85, edgecolor='black', linewidth=1.0, width=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Classe Predetta", fontsize=12)
        ax.set_ylabel(y_label_abs, fontsize=12)
        ax.set_title(f"Template - Vera Classe: {true_name}\n{title_suffix}", fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for j, h in enumerate(heights):
            if h > 0:
                fmt = f"{h:.1f}" if has_weights else f"{int(h)}"
                ax.text(j, h, fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

        integral_str = _raw_phys_str(row_integral_raw, row_integral if has_weights else None)
        ax.text(0.98, 0.96, f"Totale Eventi Veri ({true_name})\n{integral_str}", transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))
        plt.tight_layout()
        clean_name = true_name.replace("+", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_template_{clean_name}.png"), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    bottoms_abs = np.zeros(num_classes)
    for i in signal_indices:
        heights_true_i = cm_abs[i, :]
        ax.bar(x_positions, heights_true_i, bottom=bottoms_abs, label=f"Vero: {class_names[i]}", color=colors_true[i], edgecolor='black', linewidth=0.5, width=0.55)
        bottoms_abs += heights_true_i

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Classe Predetta", fontsize=12)
    ax.set_ylabel(y_label_abs, fontsize=12)
    ax.set_title(f"Composizione Assoluta per Canale Predetto\n{title_suffix}", fontsize=14)
    ax.legend(title="Vera Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_composizione_assoluta.png"), dpi=200)
    plt.close(fig)

    col_sums = cm_abs.sum(axis=0, keepdims=True)
    cm_norm_col = np.divide(cm_abs, col_sums, out=np.zeros_like(cm_abs, dtype=float), where=col_sums != 0)

    fig, ax = plt.subplots(figsize=(12, 7))
    bottoms_purity = np.zeros(num_classes)
    for i in range(num_classes):
        fraction_true_i = cm_norm_col[i, :]
        ax.bar(x_positions, fraction_true_i, bottom=bottoms_purity, label=f"Vero: {class_names[i]}", color=colors_true[i], edgecolor='black', linewidth=0.5, width=0.55)
        bottoms_purity += fraction_true_i

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Classe Predetta", fontsize=12)
    ax.set_ylabel("Frazione della Composizione (Purezza)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Composizione Normalizzata per Canale Predetto (Purezza)\n{title_suffix}", fontsize=14)
    ax.legend(title="Vera Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_composizione_normalizzata_per_canale.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions_sig = np.arange(len(signal_indices))
    signal_class_names = [class_names[i] for i in signal_indices]
    bottoms_norm = np.zeros(len(signal_indices))

    for j in range(num_classes):
        fraction_pred_j = cm_norm_row[signal_indices, j]
        ax.bar(x_positions_sig, fraction_pred_j, bottom=bottoms_norm, label=f"Predetto: {class_names[j]}", color=plt.cm.tab10(j / num_classes), edgecolor='black', linewidth=0.5, width=0.55)
        bottoms_norm += fraction_pred_j

    ax.set_xticks(x_positions_sig)
    ax.set_xticklabels(signal_class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Vera Classe di Segnale", fontsize=12)
    ax.set_ylabel("Frazione di Eventi Smistati", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Distribuzione Relativa delle Predizioni per Vera Classe (Efficienza)\n{title_suffix}", fontsize=14)
    ax.legend(title="Classe Predetta", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_efficienza_normalizzata_per_vera_classe.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    bin_edges = np.arange(num_classes + 1) - 0.5
    for true_class_idx in signal_indices:
        true_name = class_names[true_class_idx]
        heights_norm = cm_norm_row[true_class_idx, :]
        original_integral = np.sum(cm_abs[true_class_idx, :])
        yield_str = f"{original_integral:.1f}" if has_weights else f"{int(original_integral)}"

        ax.stairs(heights_norm, bin_edges, label=f"Vero: {true_name} (Int: {yield_str})", color=colors_true[true_class_idx], linewidth=2.2)
        ax.fill_between(np.arange(num_classes), heights_norm, step="mid", color=colors_true[true_class_idx], alpha=0.08)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Classe Predetta", fontsize=12)
    ax.set_ylabel("Frazione Relativa", fontsize=12)
    ax.set_ylim(0, max(1.05, cm_norm_row.max() * 1.1))
    ax.set_title(f"Template Overlayed (Normalizzati a 1)\n{title_suffix}", fontsize=14)
    ax.legend(title="Vera Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_template_overlay.png"), dpi=200)
    plt.close(fig)


# =====================================================================
# ORCHESTRAZIONE
# =====================================================================
def split_dataframe_50_50(
    df: pd.DataFrame,
    label_branch: Optional[str] = None,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("[INFO] Avvio suddivisione 50-50 del DataFrame...")
 
    stratify_target = None
    if stratify:
        if label_branch and label_branch in df.columns:
            stratify_target = df[label_branch]
            print(f" -> Stratificazione attiva sul branch: '{label_branch}'")
        else:
            print(" -> [WARNING] Branch di label non trovato o non specificato. Suddivisione senza stratificazione.")
 
    df_half1, df_half2 = train_test_split(
        df,
        test_size=0.5,
        random_state=random_state,
        stratify=stratify_target
    )
 
    df_half1 = df_half1.reset_index(drop=True)
    df_half2 = df_half2.reset_index(drop=True)
 
    print("Suddivisione completata:")
    print(f" - Sotto-dataset 1: {len(df_half1)} eventi ({len(df_half1) / len(df) * 100:.1f}%)")
    print(f" - Sotto-dataset 2: {len(df_half2)} eventi ({len(df_half2) / len(df) * 100:.1f}%)\n")
 
    return df_half1, df_half2

def process_and_save_nn_results(
    y_true_6: np.ndarray,
    y_pred_6: np.ndarray,
    weights: np.ndarray,
    class_names: Sequence[str],
    output_parquet_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_events = pd.DataFrame({
        "y_true_idx": y_true_6,
        "y_pred_idx": y_pred_6,
        "weight": weights,
        "y_true_name": [class_names[idx] for idx in y_true_6],
        "y_pred_name": [class_names[idx] for idx in y_pred_6],
    })

    df_events.to_parquet(output_parquet_path, index=False)
    print(f"[INFO] Risultati per-evento salvati su: {output_parquet_path}")

    counts_matrix = pd.crosstab(
        index=df_events["y_true_name"],
        columns=df_events["y_pred_name"],
        values=df_events["weight"],
        aggfunc="sum",
        dropna=False
    ).fillna(0.0)
    counts_matrix = counts_matrix.reindex(index=class_names, columns=class_names, fill_value=0.0)

    response_template = counts_matrix.div(counts_matrix.sum(axis=1), axis=0).fillna(0.0)
    composition_template = counts_matrix.div(counts_matrix.sum(axis=0), axis=1).fillna(0.0)

    return df_events, response_template, composition_template

def _template_e_composizione_per_meta(
    df_half: pd.DataFrame,
    class_names_6: List[str],
    output_subdir: str,
    filename_prefix: str,
) -> None:
    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    y_true_6 = df_half["y_true_6"].to_numpy()
    y_pred_cascade = df_half["y_pred_cascade"].to_numpy()
    w_phys = df_half["peso_fisico"].to_numpy()

    plot_prediction_composition(
        y_true=y_true_6, y_pred=y_pred_cascade, class_names=class_names_6,
        output_dir=output_dir, filename_prefix=filename_prefix, sample_weight=w_phys*2,
        title_suffix=f"(Cascata P(S/B) >= {CASCADE_THRESHOLD}, {filename_prefix})",
    )
    
    df_events, response_template, composition_template = process_and_save_nn_results(
        y_true_6=y_true_6, y_pred_6=y_pred_cascade, weights=w_phys,
        class_names=class_names_6,
        output_parquet_path=os.path.join(output_dir, f"{filename_prefix}_eventi.parquet"),
    )
    
    response_template.to_csv(os.path.join(output_dir, f"{filename_prefix}_template_risposta.csv"))
    composition_template.to_csv(os.path.join(output_dir, f"{filename_prefix}_template_composizione.csv"))
    
    response_template.to_json(os.path.join(output_dir, f"{filename_prefix}_template_risposta.json"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Caricamento dataset...")
    df = load_dataset(INPUT_SAMPLES)

    print("[INFO] Estrazione feature per inferenza...")
    X_raw, feature_columns = extract_features_for_inference(df)

    print("[INFO] Applicazione preprocessor...")
    X_scaled = apply_preprocessor(X_raw)

    print("[INFO] Costruzione etichette vere a 6 classi...")
    df["y_true_6"] = build_true_labels(df)
    weights = df["peso_fisico"].to_numpy()

    print("[INFO] Caricamento modelli Keras...")
    sb_model, multi_model = load_models(SB_MODEL_PATH, MULTI_MODEL_PATH)

    print("[INFO] Inferenza a cascata...")
    cascade_outputs = run_cascade_inference(
        sb_model=sb_model,
        multi_model=multi_model,
        X_scaled=X_scaled,
        threshold=CASCADE_THRESHOLD,
    )

    df["y_prob_sb"] = cascade_outputs["y_prob_sb"]
    df["y_pred_cascade"] = cascade_outputs["y_pred_cascade"]
    df["y_pred_multi_signal"] = cascade_outputs["y_pred_multi_signal_idx"]
    
    for i, name in enumerate(SIGNAL_NAMES):
        df[f"prob_{name}"] = cascade_outputs["y_prob_multi"][:, i]

    y_true_6 = df["y_true_6"].to_numpy()
    y_pred_cascade = df["y_pred_cascade"].to_numpy()
    y_prob_sb = df["y_prob_sb"].to_numpy()
    y_pred_multi_signal = df["y_pred_multi_signal"].to_numpy()
    y_pred_multi_full = cascade_outputs["y_pred_multi_full_idx"]

    class_names_6 = SIGNAL_NAMES + ["Fondo Totale"]
    y_true_binary = (y_true_6 < len(SIGNAL_NAMES)).astype(int)
    y_pred_binary = (y_prob_sb >= CASCADE_THRESHOLD).astype(int)

    print("[INFO] Generazione Confusion Matrices e ROC...")
    plot_conf_matrix(y_true_binary, y_pred_binary, ["Fondo Totale", "Segnale VBS"], OUTPUT_DIR, "confusion_matrix_SB.png", weights, f"SB threshold = {CASCADE_THRESHOLD}")
    
    true_signal_mask = y_true_6 < len(SIGNAL_NAMES)
    plot_conf_matrix(y_true_6[true_signal_mask], y_pred_multi_signal[true_signal_mask], SIGNAL_NAMES, OUTPUT_DIR, "confusion_matrix_multiclass.png", weights[true_signal_mask], "Multiclass (solo segnali veri)")

    plot_conf_matrix(y_true_6, y_pred_cascade, class_names_6, OUTPUT_DIR, "confusion_matrix_cascade_6x6.png", weights, f"Cascade threshold = {CASCADE_THRESHOLD}")

    plot_roc_binary(y_true_binary, y_prob_sb, OUTPUT_DIR, weights)

    analyze_binary_signal_region(y_true_binary, y_prob_sb, weights, threshold=SIGNAL_REGION_THRESHOLD)

    print("[INFO] Split 50/50 del dataset con colonne incorporate...")
    df_half1, df_half2 = split_dataframe_50_50(df, label_branch="y_true_6")

    print("[INFO] Generazione template e composizione...")
    _template_e_composizione_per_meta(df_half1, class_names_6, output_subdir="meta_template", filename_prefix="template")
    _template_e_composizione_per_meta(df_half2, class_names_6, output_subdir="meta_composizione", filename_prefix="composizione")

    print("[INFO] Salvataggio completo evento-per-evento...")
    df.to_parquet(PREDICTIONS_OUTPUT_PATH.replace(".csv", ".parquet"), index=False)
    print("[INFO] Pipeline completata con successo.")

if __name__ == "__main__":
    main()