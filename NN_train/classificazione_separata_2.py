import os
import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from sb_threshold_scan import scan_sb_threshold, plot_sb_threshold_scan
from multi_threshold_scan import scan_multiclass_thresholds, plot_multiclass_threshold_scan
from feature_ranking import rank_features_all_heads
from bkg_class_multiclass import prepare_multiclass_with_bkg_class
from evaluate_cascade_learned_bkg import evaluate_cascade_with_learned_bkg_class

from multiclassifier_progressivo import (
    Config,
    parse_args_to_config,
    save_config_to_txt,
    train_model,
    scale_sample_weight_to_luminosity,
    plot_history,
    plot_multitask_confusion_matrices,
    plot_multitask_roc_curves,
    plot_prediction_composition,
    plot_conf_matrix,
    plot_roc_binary,
    plot_roc_multiclass,
    analyze_binary_signal_region,
    analyze_cascaded_signal_region,
    analyze_signal_region,
    PROCESS_CLASS_NAMES,
    WarmUpCosineDecay,
)

from shared_data import (
    FEATURE_DICT,
    load_and_split_shared,
    prepare_for_network,
    _build_run_config,
)

import shutil
from confidence_filter_analysis import evaluate_cascade_with_confidence_filter


# Se True, salta il training di multi_only: copia i risultati dell'ultimo
# invece di riallenare. Utile mentre si itera solo su sb_only.
siamoMultiFelici = False
PREVIOUS_GOOD_MULTI_DIR = "outputs/multi_only"    
PREVIOUS_GOOD_MULTI_MODEL_FILENAME = "best_model.keras"  
siamoSbFelici = False
PREVIOUS_GOOD_SB_DIR = "outputs/sb_only"    
PREVIOUS_GOOD_SB_MODEL_FILENAME = "best_model.keras" 
FamoSei = True
if FamoSei:
    siamoMultiFelici = False
    siamoSbFelici = False

SB_OVERRIDES = {
    # Di base non si toccano______SB
    "train_stage": "sb_only",
    "use_batch_norm": "False",
    "use_layer_norm": "True",
    "optimizer_type": "AdamW",
    "batch_size": 256,
    "random_state": 42,
    "epochs": 200,
    "tree_name": "Events",
    "label_branch": "kind_events",
    "weight_branch": "Weight_weight",
    "chunk_size": 250000,
    "max_entries": "None",
    "test_size": 0.2,
    "val_size": 0.1,

    # Inutili per questo______SB
    "hidden_layers_multi": "",
    "pretrain_epochs": 50,
    "freeze_shared_after_pretrain": "True",
    "freeze_sb_after_pretrain": "True",
    "pretrained_weights_path": "pretrained_sb_weights.weights.h5",
    "fine_tune_lr": 1e-5,
    "two_stage_auto": "False",
    "use_focal_loss_multi": "False",
    "focal_gamma_multi": 2.0,
    "focal_alpha_multi": 0.25,
    "use_residual_in_multi": "False",
    "norm_weight": "per_class",

    
    # Raramente da toccare --> parametri di altri______SB
    "dropout_rate": 0.1,
    "weight_decay": 1e-4,
    "one_cycle_max_lr": 1e-3,
    "sgd_momentum": 0.9,
    "step_lr_period": 50,
    "step_lr_gamma": 0.5,
    "warmup_steps": 1000,

    # Importanti da cambiare______SB
    "output_dir": "outputs/sb_only",
    "feature_set": "4l2n",

    "hidden_layers": "", # Qua se non vuoi layers puoi lasciare vuoto. Stiamo bypassando il parser______SB
    "hidden_layers_sb": "256, 512, 1024, 256, 128, 64",

    # A volte da cambiare______SB
    "use_dropout": "True",
    "bootstrap": "False",
    "scheduler_type": "StepLR",
    "norm_weights_sb": "per_class_sublinear",
    "learning_rate": 1e-3,
    "use_warmup": "False",
    "grad_accum_steps": 1,
    "global_clipnorm": 4.0,
    "sb_pos_weight_multiplier": 1,
    "reweight_for_sb_head": "True",   # Ripesa sulle classi prima di passare il reweight scelto
    "use_class_weight_sb": "False",   # Annulla quello che fa reweight_for_sb_head
    "norm_weight_sb_alpha": 0.5,      # Alpha per "per_class_sublinear"
    "norm_pre_weight_sb_alpha": 0.3,  # Alpha per "per_class_sublinear" nel pre-reweighting per pareggiare i canali -

    # Loss paramether______SB
    "use_focal_loss_sb": "False",
    "focal_gamma_sb": 2.0,
    "focal_alpha_sb": 0.25,
}
MULTI_OVERRIDES = {
    # Di base non si toccano______MULTI
    "train_stage": "multi_only",
    "use_batch_norm": "False",
    "use_layer_norm": "True",
    "optimizer_type": "AdamW",
    "batch_size": 256,
    "random_state": 42,
    "epochs": 200,
    "tree_name": "Events",
    "label_branch": "kind_events",
    "weight_branch": "Weight_weight",
    "chunk_size": 250000,
    "max_entries": "None",
    "test_size": 0.2,
    "val_size": 0.1,

    # Inutili per questo______MULTI
    "hidden_layers_sb": "",
    "reweight_for_sb_head": "True",
    "use_class_weight_sb": "True",
    "pretrain_epochs": 50,
    "freeze_shared_after_pretrain": "True",
    "freeze_sb_after_pretrain": "True",
    "pretrained_weights_path": "pretrained_sb_weights.weights.h5",
    "fine_tune_lr": 1e-5,
    "two_stage_auto": "False",
    "use_focal_loss_sb": "False",
    "focal_gamma_sb": 2.0,
    "focal_alpha_sb": 0.25,  
    "norm_weights_sb": "per_class",

    
    # Raramente da toccare______MULTI
    "learning_rate": 1e-3,
    "dropout_rate": 0.2,
    "weight_decay": 1e-4,
    "one_cycle_max_lr": 1e-3,
    "sgd_momentum": 0.9,
    "step_lr_period": 20,
    "step_lr_gamma": 0.1,
    "warmup_steps": 1000,
    "norm_weight_sb_alpha": 0.5,


    # Importanti da cambiare______MULTI
    "output_dir": "outputs/multi_only",
    "feature_set": "4l2n",

    "hidden_layers": "",
    "hidden_layers_multi": "256, 512, 1024, 512, 128, 64",

    # A volte da cambiare______MULTI
    "bootstrap": "True",
    "use_dropout": "True",
    "scheduler_type": "ReduceLROnPlateau",
    "norm_weight": "per_class_sublinear",
    "use_warmup": "False",
    "grad_accum_steps": 1,
    "global_clipnorm": 4.0,
    "use_residual_in_multi": "True",
    "norm_weight_sb_alpha": 0.5,

    # Loss paramether______MULTI
    "use_focal_loss_multi": "False",
    "focal_gamma_multi": 2.0,
    "focal_alpha_multi": 0.25,
}
CASCADE_OUTPUT_DIR = "outputs/cascade"
CASCADE_THRESHOLD = 0.5


class BkgSeparationMonitor(keras.callbacks.Callback):
    def __init__(self, X_val, y_val_process_head, bkg_class_idx, batch_size=2048, verbose=True):
        super().__init__()
        self.X_val = X_val
        self.y_true = np.argmax(y_val_process_head, axis=1)
        self.bkg_class_idx = bkg_class_idx
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_bkg_true = self.y_true == bkg_class_idx
        self.history = {"val_bkg_recall": [], "val_bkg_precision": [], "val_bkg_auc": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        preds = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        y_prob = preds["process_head"] if isinstance(preds, dict) else preds
        y_pred = np.argmax(y_prob, axis=1)
        is_bkg_pred = y_pred == self.bkg_class_idx

        n_true_bkg = int(self.is_bkg_true.sum())
        n_pred_bkg = int(is_bkg_pred.sum())
        n_correct = int((self.is_bkg_true & is_bkg_pred).sum())

        recall = n_correct / n_true_bkg if n_true_bkg > 0 else float("nan")
        precision = n_correct / n_pred_bkg if n_pred_bkg > 0 else float("nan")

        bkg_score = y_prob[:, self.bkg_class_idx]
        if 0 < n_true_bkg < len(self.y_true):
            auc_val = roc_auc_score(self.is_bkg_true, bkg_score)
        else:
            auc_val = float("nan")

        self.history["val_bkg_recall"].append(recall)
        self.history["val_bkg_precision"].append(precision)
        self.history["val_bkg_auc"].append(auc_val)

        logs["val_bkg_recall"] = recall
        logs["val_bkg_precision"] = precision
        logs["val_bkg_auc"] = auc_val

        if self.verbose:
            print(
                f" - val_bkg_recall: {recall:.4f} - val_bkg_precision: {precision:.4f} "
                f"- val_bkg_auc: {auc_val:.4f}"
            )

def plot_bkg_monitor_history(bkg_monitor: "BkgSeparationMonitor", output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(bkg_monitor.history["val_bkg_recall"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, bkg_monitor.history["val_bkg_recall"], label="val_bkg_recall")
    ax.plot(epochs, bkg_monitor.history["val_bkg_precision"], label="val_bkg_precision")
    ax.plot(epochs, bkg_monitor.history["val_bkg_auc"], label="val_bkg_auc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Separazione classe 'Bkg' (process_head) per epoca")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bkg_separation_history.png"), dpi=150)
    plt.close(fig)

def train_one(
    base_config: Config,
    overrides: dict,
    data,
    total_events: int,
    extra_callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> Tuple[keras.Model, keras.callbacks.History, Config]:
    (
        X_train, X_val, _X_test,
        y_train, y_val, _y_test,
        w_train, w_val, _w_test,
        _w_train_raw, w_val_raw, _w_test_raw,
        _preproc,
    ) = data

    run_config = _build_run_config(base_config, overrides)
    os.makedirs(run_config.output_dir, exist_ok=True)
    save_config_to_txt(run_config, os.path.join(run_config.output_dir, "run_config.txt"))

    model, history = train_model(
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, w_val=w_val, w_val_raw=w_val_raw,
        total_events=total_events, config=run_config,
        extra_callbacks=extra_callbacks,
    )
    return model, history, run_config

def evaluate_cascade_two_models(
    model_sb: keras.Model,
    model_multi: keras.Model,
    X_test_sb: np.ndarray,
    X_test_multi: np.ndarray,
    y_test: Dict[str, np.ndarray],
    w_test_raw: Dict[str, np.ndarray],
    history_sb: Optional[keras.callbacks.History],
    history_multi: Optional[keras.callbacks.History],
    output_dir: str,
    total_events: int,
    batch_size: int,
    feature_columns_sb: list,
    feature_columns_multi: list,
    cascade_threshold: float = CASCADE_THRESHOLD,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    if history_sb is not None:
        plot_history(history_sb, os.path.join(output_dir, "history_sb"))
    if history_multi is not None:
        plot_history(history_multi, os.path.join(output_dir, "history_multi"))

    y_prob_sb = model_sb.predict(X_test_sb, batch_size=batch_size, verbose=0)
    y_prob_multi = model_multi.predict(X_test_multi, batch_size=batch_size, verbose=0)
    y_pred_probs_dict = {"sb_head": y_prob_sb, "process_head": y_prob_multi}


    w_test_physical = {
        head: scale_sample_weight_to_luminosity(w_test_raw[head], total_events)
        for head in w_test_raw
    }

    plot_multitask_confusion_matrices(
        y_true_dict=y_test, y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test_physical, w_raw_dict=w_test_raw,
        signal_class_names=PROCESS_CLASS_NAMES, output_dir=output_dir,
        total_events=total_events, cascade_threshold=cascade_threshold,
    )
    plot_multitask_roc_curves(
        y_true_dict=y_test, y_pred_probs_dict=y_pred_probs_dict,
        w_dict=w_test_physical, signal_class_names=PROCESS_CLASS_NAMES,
        output_dir=output_dir,
    )

    y_true_sb = y_test["sb_head"].flatten()
    y_true_process_int = np.argmax(y_test["process_head"], axis=1)
    y_pred_process_int = np.argmax(y_pred_probs_dict["process_head"], axis=1)

    bkg_class_idx = len(PROCESS_CLASS_NAMES)
    y_true_cascaded = np.where(y_true_sb == 1, y_true_process_int, bkg_class_idx)
    y_pred_cascaded = np.where(
        y_pred_probs_dict["sb_head"].flatten() >= cascade_threshold,
        y_pred_process_int, bkg_class_idx,
    )

    plot_prediction_composition(
        y_true=y_true_cascaded, y_pred=y_pred_cascaded,
        class_names=list(PROCESS_CLASS_NAMES) + ["Fondo Totale"],
        output_dir=output_dir, filename_prefix="prediction_composition",
        sample_weight=w_test_physical["sb_head"].flatten(),
        sample_weight_raw=w_test_raw["sb_head"].flatten(),
        total_events=total_events,
        title_suffix=f"(Cascata con P(S/B) >= {cascade_threshold}) - due reti indipendenti",
    )

    report_path = os.path.join(output_dir, "analysis_report.txt")

    analyze_binary_signal_region(
        y_true=y_true_sb, y_prob_sig=y_pred_probs_dict["sb_head"].flatten(),
        sample_weight=w_test_physical["sb_head"].flatten(),
        threshold=cascade_threshold, report_path=report_path,
    )

    threshold_scan_df = scan_sb_threshold(
        y_true=y_true_sb, y_prob=y_pred_probs_dict["sb_head"].flatten(),
        sample_weight_physical=w_test_physical["sb_head"].flatten(),
        current_threshold=cascade_threshold,
    )
    threshold_scan_df.to_csv(os.path.join(output_dir, "sb_threshold_scan.csv"), index=False)
    plot_sb_threshold_scan(
        threshold_scan_df,
        output_path=os.path.join(output_dir, "sb_threshold_scan.png"),
        current_threshold=cascade_threshold,
    )

    is_true_signal = y_true_sb == 1
    threshold_scan_multi_results = scan_multiclass_thresholds(
        y_true_int=y_true_process_int[is_true_signal],
        y_prob=y_pred_probs_dict["process_head"][is_true_signal],
        sample_weight_physical=w_test_physical["process_head"][is_true_signal].flatten(),
        class_names=PROCESS_CLASS_NAMES,
    )
    for class_name, df_class in threshold_scan_multi_results.items():
        safe_name = class_name.replace("+", "_plus_").replace(" ", "_")
        df_class.to_csv(os.path.join(output_dir, f"process_threshold_scan_{safe_name}.csv"), index=False)
    plot_multiclass_threshold_scan(threshold_scan_multi_results, output_dir=output_dir)

    feature_ranking_sb_df = rank_features_all_heads(
        X_test=X_test_sb, feature_names=feature_columns_sb, y_test=y_test,
        w_test_physical=w_test_physical, class_names=PROCESS_CLASS_NAMES,
    )
    feature_ranking_sb_df.to_csv(os.path.join(output_dir, "feature_ranking_sb.csv"), index=False)

    feature_ranking_multi_df = rank_features_all_heads(
        X_test=X_test_multi, feature_names=feature_columns_multi, y_test=y_test,
        w_test_physical=w_test_physical, class_names=PROCESS_CLASS_NAMES,
    )
    feature_ranking_multi_df.to_csv(os.path.join(output_dir, "feature_ranking_multi.csv"), index=False)

    analyze_cascaded_signal_region(
        y_true_multi_int=y_true_process_int, y_prob_bin=y_pred_probs_dict["sb_head"].flatten(),
        y_prob_multi=y_pred_probs_dict["process_head"], class_names=PROCESS_CLASS_NAMES,
        threshold=cascade_threshold, report_path=report_path,
    )

    for idx in range(len(PROCESS_CLASS_NAMES)):
        analyze_signal_region(
            y_true=y_true_process_int, y_probs=y_pred_probs_dict["process_head"],
            sample_weight=w_test_physical["process_head"].flatten(),
            class_names=PROCESS_CLASS_NAMES, target_class_idx=idx,
            threshold=cascade_threshold, report_path=report_path,
        )

    eval_sb = model_sb.evaluate(X_test_sb, y_test["sb_head"], sample_weight=w_test_physical["sb_head"], verbose=0)
    if not isinstance(eval_sb, (list, tuple)):
        eval_sb = [eval_sb]
    results = {f"sb_{name}": val for name, val in zip(model_sb.metrics_names, eval_sb)}

    eval_multi = model_multi.evaluate(X_test_multi, y_test["process_head"], sample_weight=w_test_physical["process_head"], verbose=0)
    if not isinstance(eval_multi, (list, tuple)):
        eval_multi = [eval_multi]
    results.update({f"multi_{name}": val for name, val in zip(model_multi.metrics_names, eval_multi)})

    return results


def main() -> None:
    base_config = parse_args_to_config()

    # Caricamento + split UNA SOLA VOLTA, condiviso tra le due reti (vedi
    # shared_data_split.py) - garantisce lo stesso split di eventi anche se
    # le due reti hanno feature_set diversi.
    shared = load_and_split_shared(base_config, SB_OVERRIDES, MULTI_OVERRIDES)

    print("[SEZIONE 1] Preparazione + training sb_only...")
    sb_config = _build_run_config(base_config, SB_OVERRIDES)
    data_sb, feature_columns_sb = prepare_for_network(shared, sb_config)
    if siamoSbFelici:
        print(f"  -> siamoSbFelici=True: copio {PREVIOUS_GOOD_SB_DIR} in "
              f"{sb_config.output_dir} e ricarico il modello, nessun training.")
        if os.path.abspath(PREVIOUS_GOOD_SB_DIR) != os.path.abspath(sb_config.output_dir):
            if os.path.exists(sb_config.output_dir):
                shutil.rmtree(sb_config.output_dir)
            shutil.copytree(PREVIOUS_GOOD_SB_DIR, sb_config.output_dir)
        model_sb = keras.models.load_model(
            os.path.join(sb_config.output_dir, PREVIOUS_GOOD_SB_MODEL_FILENAME),
            custom_objects={"WarmUpCosineDecay": WarmUpCosineDecay},  # vedi nota sotto
        )
        history_sb = None
    else:
        model_sb, history_sb, config_sb = train_one(base_config, SB_OVERRIDES, data_sb, shared.total_events)

    print("[SEZIONE 2] Preparazione + training multi_only...")
    multi_config = _build_run_config(base_config, MULTI_OVERRIDES)
    if siamoMultiFelici:
        print(f"  -> siamoMultiFelici=True: copio {PREVIOUS_GOOD_MULTI_DIR} in "
              f"{multi_config.output_dir} e ricarico il modello, nessun training.")
        if FamoSei:
            data_multi6, feature_columns_multi, class_names_6 = prepare_multiclass_with_bkg_class(
                shared=shared, multi_run_config=multi_config, model_sb=model_sb,
                sb_preproc=data_sb[12], sb_feature_set=sb_config.feature_set,
                sb_batch_size=base_config.batch_size,
                bkg_class_name="Bkg",  # bkg_sample_size non specificato -> default: canale di segnale più numeroso
            )
        else:
            data_multi, feature_columns_multi = prepare_for_network(shared, multi_config)
        if os.path.abspath(PREVIOUS_GOOD_MULTI_DIR) != os.path.abspath(multi_config.output_dir):
            if os.path.exists(multi_config.output_dir):
                shutil.rmtree(multi_config.output_dir)
            shutil.copytree(PREVIOUS_GOOD_MULTI_DIR, multi_config.output_dir)
        model_multi = keras.models.load_model(
            os.path.join(multi_config.output_dir, PREVIOUS_GOOD_MULTI_MODEL_FILENAME),
            custom_objects={"WarmUpCosineDecay": WarmUpCosineDecay},  # vedi nota sotto
        )
        history_multi = None
        bkg_monitor = None
    elif FamoSei:
        print("  -> FamoSei=True: alleno con anche bkg difficile...")
        data_multi6, feature_columns_multi, class_names_6 = prepare_multiclass_with_bkg_class(
            shared=shared, multi_run_config=multi_config, model_sb=model_sb,
            sb_preproc=data_sb[12], sb_feature_set=sb_config.feature_set,
            sb_batch_size=base_config.batch_size,
            bkg_class_name="Bkg",  # bkg_sample_size non specificato -> default: canale di segnale più numeroso
        )
        bkg_monitor = BkgSeparationMonitor(
            X_val=data_multi6[1],
            y_val_process_head=data_multi6[4]["process_head"],
            bkg_class_idx=len(PROCESS_CLASS_NAMES),  # indice 5, coerente col resto del codice FamoSei
        )
        model_multi, history_multi, config_multi = train_one(
            base_config, MULTI_OVERRIDES, data_multi6, shared.total_events,
            extra_callbacks=[bkg_monitor],
        )
    else:
        data_multi, feature_columns_multi = prepare_for_network(shared, multi_config)
        model_multi, history_multi, config_multi = train_one(base_config, MULTI_OVERRIDES, data_multi, shared.total_events)

    print("[SEZIONE 3] Valutazione a cascata (due reti indipendenti)...")
    if FamoSei:
        cascade_threshold_famosei = 0.3   

        summary = evaluate_cascade_with_learned_bkg_class(
            model_sb=model_sb, model_multi_6class=model_multi,
            X_test_sb=data_sb[2], X_test_multi=data_multi6[2],
            y_test=data_multi6[5], w_test_raw=data_multi6[11],
            class_names_6=class_names_6, total_events=shared.total_events,
            output_dir=os.path.join(CASCADE_OUTPUT_DIR, "learned_bkg"),
            batch_size=base_config.batch_size,
            cascade_threshold=cascade_threshold_famosei,
        )

        if history_sb is not None:
            plot_history(history_sb, os.path.join(CASCADE_OUTPUT_DIR, "history_sb"))
        if history_multi is not None:
            plot_history(history_multi, os.path.join(CASCADE_OUTPUT_DIR, "history_multi"))
        if bkg_monitor is not None:
            plot_bkg_monitor_history(bkg_monitor, os.path.join(CASCADE_OUTPUT_DIR, "bkg_monitor"))

        eval_output_dir = os.path.join(CASCADE_OUTPUT_DIR, "learned_bkg_evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)

        X_test_sb = data_sb[2]
        X_test_multi = data_multi6[2]
        y_test = data_multi6[5]          # dizionari di test a 6 classi (sb_head e process_head)
        w_test_raw = data_multi6[11]     # pesi grezzi coerenti con y_test (fondo test NON azzerato)

        y_prob_sb = model_sb.predict(X_test_sb, batch_size=base_config.batch_size, verbose=0)
        y_prob_multi = model_multi.predict(X_test_multi, batch_size=base_config.batch_size, verbose=0)

        w_test_physical = {
            head: scale_sample_weight_to_luminosity(w_test_raw[head], shared.total_events)
            for head in w_test_raw
        }

        y_true_sb = y_test["sb_head"].flatten()
        y_pred_sb_bin = (y_prob_sb.flatten() >= cascade_threshold_famosei).astype(int)

        plot_conf_matrix(
            y_true=y_true_sb, y_pred=y_pred_sb_bin,
            class_names=["Fondo Totale", "Segnale VBS"],
            output_dir=eval_output_dir, filename="confusion_matrix_sb_head.png",
            sample_weight=w_test_physical["sb_head"].flatten(),
            sample_weight_raw=w_test_raw["sb_head"].flatten(),
            total_events=shared.total_events,
            threshold_info=f"(Taglio P(S/B) >= {cascade_threshold_famosei})",
        )
        plot_roc_binary(
            y_true=y_true_sb, y_prob=y_prob_sb.flatten(),
            output_dir=eval_output_dir,
            sample_weight=w_test_physical["sb_head"].flatten(),
        )

        y_true_process_full = np.argmax(y_test["process_head"], axis=1)
        y_pred_process_full = np.argmax(y_prob_multi, axis=1)

        plot_conf_matrix(
            y_true=y_true_process_full, y_pred=y_pred_process_full,
            class_names=class_names_6, output_dir=eval_output_dir,
            filename="confusion_matrix_process_head.png",
            sample_weight=w_test_physical["process_head"].flatten(),
            sample_weight_raw=w_test_raw["process_head"].flatten(),
            total_events=shared.total_events,
            threshold_info="(process_head nativo, nessun filtro S/B, include il vero fondo)",
        )
        plot_roc_multiclass(
            y_true_onehot=y_test["process_head"], y_prob=y_prob_multi,
            class_names=class_names_6, output_dir=eval_output_dir,
            sample_weight=w_test_physical["process_head"].flatten(),
        )

        y_pred_sb_flat = y_prob_sb.flatten()
        y_pred_process_int = np.argmax(y_prob_multi, axis=1)

        bkg_class_idx = len(PROCESS_CLASS_NAMES)  

        y_true_cascaded = np.where(y_true_sb == 1, y_true_process_full + 1, 0)

        pass_net1 = y_pred_sb_flat >= cascade_threshold_famosei
        is_signal_net2 = y_pred_process_int != bkg_class_idx

        y_pred_cascaded = np.zeros_like(y_true_cascaded)
        final_valid_mask = pass_net1 & is_signal_net2
        y_pred_cascaded[final_valid_mask] = y_pred_process_int[final_valid_mask] + 1

        all_classes_total = ["Fondo Totale"] + list(PROCESS_CLASS_NAMES)

        plot_prediction_composition(
            y_true=y_true_cascaded, y_pred=y_pred_cascaded,
            class_names=all_classes_total,
            output_dir=eval_output_dir, filename_prefix="prediction_composition_cascade_6class",
            sample_weight=w_test_physical["sb_head"].flatten(),
            sample_weight_raw=w_test_raw["sb_head"].flatten(),
            total_events=shared.total_events,
            title_suffix=f"(Cascata 6 Classi con Hard Negative Mining P(S/B) >= {cascade_threshold_famosei})",
        )
        print(f"Valutazione a cascata completata. Risultati salvati in: {eval_output_dir}")

    else:
        X_test_sb = data_sb[2]
        X_test_multi = data_multi[2]
        y_test = data_sb[5]
        w_test_raw = data_sb[11]

        results = evaluate_cascade_two_models(
            model_sb=model_sb, model_multi=model_multi,
            X_test_sb=X_test_sb, X_test_multi=X_test_multi,
            y_test=y_test, w_test_raw=w_test_raw,
            history_sb=history_sb, history_multi=history_multi,
            output_dir=CASCADE_OUTPUT_DIR, total_events=shared.total_events,
            batch_size=base_config.batch_size,
            feature_columns_sb=feature_columns_sb, feature_columns_multi=feature_columns_multi,
            cascade_threshold=CASCADE_THRESHOLD,
        )

        y_prob_sb = model_sb.predict(X_test_sb, batch_size=base_config.batch_size, verbose=0)
        y_prob_multi = model_multi.predict(X_test_multi, batch_size=base_config.batch_size, verbose=0)

        evaluate_cascade_with_confidence_filter(
            y_prob_sb=y_prob_sb, y_prob_multi=y_prob_multi, y_test=y_test,
            w_test_raw=w_test_raw, total_events=shared.total_events,
            output_dir=os.path.join(CASCADE_OUTPUT_DIR, "confidence_filter"),
            cascade_threshold=CASCADE_THRESHOLD,
            confidence_thresholds=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7),
        )

        print("\nMetriche finali (cascata, due reti indipendenti):")
        for name, value in results.items():
            print(f"  {name}: {value:.6f}")

if __name__ == "__main__":
    main()