"""
Experiment script, copy this script into another file when running the experiment.
Make function changes in the baseline (AAAA_t10track.py) and fix the configurations (cfg) in this script.
Choose real or smoke data

Purpose:
    FILL IN THE PURPOSE HERE

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings)

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track

def main():

    print("\n======================================")
    print("EXPERIMENT Preprocess - Test different preprocessing filter")
    print("======================================")

    SEED = 0
    SUBSET = 2000

    USE_REAL = True
    track = ECGCinC2017Track()

    # ----------------------------------------------
    # DOWNLOAD AND LOAD REAL ECG DATA
    # ----------------------------------------------

    if USE_REAL:
        cache_dir = PROJECT_ROOT / "data_cache" / "ecg_B0"
        print("\nDownloading ECG recordings...")
        track.download(str(cache_dir), subset = SUBSET, seed = SEED)
        records = track.load(str(cache_dir))
        track.assert_no_holdout(records)
    else:
        records = track.smoke()
        print(f"Generated {len(records)} synthetic ECG recordings")


    #This print function together with the other print functions are defined below the main function
    printDataInfo(track, records)

#     cfg = {
#         "preprocess": [
#             {"preprocess":"denoise","broadband":"savgol","savgol_order":2,"win":9},
#             #{"preprocess":"denoise","powerline":"notch","f0":50.0,"q":40,"harmonics":(1,2)},
#             #{"preprocess":"bandpass","ecg_band":"sym4","order":5}
#         ],
#     }

    configs = {
        "P0_none": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "P1_bandpass": {
            #Configuration 2
            "seed" : SEED,
            "preprocess": "bandpass",
            "ecg_band": (0.5, 40),
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "P2_wavelet": {
            #Configuration 3
            "seed" : SEED,
            "preprocess": "wavelet",
            "wavelet": "db4",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "P3_notch": {
            #Configuration 4
            "seed" : SEED,
            "preprocess": "bandpass",
            "notch": 50,
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "P4_bandpass_notch": {
            #Configuration 5
            "seed" : SEED,
            "preprocess": "bandpass",
            "ecg_band": (0.5, 40),
            "notch": 50,
            "wavelet_mode": "soft",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },
        "P6_bandpass_wavelet": {
            "seed": SEED,

            "preprocess": [
                {
                    "preprocess": "bandpass",
                    "ecg_band": (0.5, 40)
                },
                {
                    "preprocess": "wavelet",
                    "wavelet": "db4"
                }
            ],

            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "P7_notch_wavelet": {
            "seed": SEED,

            "preprocess": [
                {
                    "preprocess": "bandpass",
                    "notch": 50
                },
                {
                    "preprocess": "wavelet",
                    "wavelet": "db4"
                }
            ],

            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },
    }

    results = []
    confusion_results = []

    for name, cfg in configs.items():
        print("\n======================================")
        print(f"EXPERIMENT: {name}")
        print("======================================")

        print("\nConfiguration")
        print(cfg)

        current_track = ECGCinC2017Track()

        # ----------------------------------------------
        # PREPROCESSING + FEATURE EXTRACTION
        # ----------------------------------------------

        print("\nExtracting features...")

        X, y, groups = current_track.build_dataset(records, cfg = cfg)

        printFeatureInfo(X, y, groups)

        # ----------------------------------------------
        # 4. MACHINE LEARNING + CROSS-VALIDATION
        # ----------------------------------------------

        print("\nRunning 5-fold GroupKFold...")

        report = current_track.evaluate(X, y, groups, cfg = cfg)

        # ----------------------------------------------
        # EVALUATION RESULTS
        # ----------------------------------------------

        print("\n======================================")
        print("B0 — CLASSIFICATION RESULTS")
        print("======================================")

        printEvaluationScores(current_track, report)
        printConfusionMatrix(current_track, report)
        printPerClassPerformance(current_track, report)

        results.append({
        "Experiment": name,
        "Macro-F1": report["macro_f1"],
        "Kappa": report["cohens_kappa"],
        })

        # ----------------------------------------------
        # SAVE CONFUSION MATRIX FOR VISUALIZATION
        # ----------------------------------------------

        cm = confusion_matrix(
            report["y_true"],
            report["y_pred"],
            labels=current_track.meta.classes
        )

        confusion_results.append({
            "name": name,
            "matrix": cm,
            "macro_f1": report["macro_f1"]
        })

    import pandas as pd

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Macro-F1",
        ascending=False
    )

    print("\n======================================")
    print("PREPROCESSING EXPERIMENT SUMMARY")
    print("======================================")

    print(results_df.to_string(index=False))

    
    # ----------------------------------------------
    # PLOT ALL CONFUSION MATRICES
    # ----------------------------------------------

    print("\nGenerating confusion matrix comparison...")

    # Create results folder
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # File path
    save_path = results_dir / "preprocessing_confusion_matrices.png"

    # Generate figure
    plotConfusionMatrices(
        confusion_results,
        save_path=save_path
    )

# ----------------------------------------------
# These are a bunch of print function, so print can easily be removed when just testing stuff
# ----------------------------------------------

def printDataInfo(track, records):
    print(f"Number of recordings: {len(records)}")

    print("\nClass distribution")
    labels = np.array([rec.labels[0] for rec in records])
    for cls in track.meta.classes:
        print(f"{cls}: {np.sum(labels==cls)}")


def printFeatureInfo(X, y, groups):
    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Unique recordings: {len(set(groups))}")


def printEvaluationScores(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    print(f"\nMacro-F1: {macro_f1:.4f}")
    print(f"Cohen's kappa: {report['cohens_kappa']:.4f}")

    fold_scores = []

    for fold in report["per_fold"]:
        print(
            f"Fold {fold['fold']}: "
            f"Macro-F1 = {fold['macro_f1']:.4f}"
        )

        fold_scores.append(fold["macro_f1"])

    fold_scores = np.array(fold_scores)

    print("\nMacro-F1 across folds:")

    print(f"Mean: {np.mean(fold_scores):.4f}")
    print(f"SD: {np.std(fold_scores):.4f}")
    print(f"Min: {np.min(fold_scores):.4f}")
    print(f"Max: {np.max(fold_scores):.4f}")


def printConfusionMatrix(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print("\nConfusion matrix:")
    print("Class order:", classes)
    print(cm)


def printPerClassPerformance(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    print("\nPer-class performance:")

    print(classification_report(y_true, y_pred, labels=classes, target_names=["Normal", "AF", "Other", "Noisy"], zero_division=0))

    print("\nExperiment B0 completed!")

def plotConfusionMatrices(confusion_results, save_path=None):

    """
    Plot confusion matrices for all preprocessing experiments.

    Each matrix displays:
        - Number of predictions
        - Percentage of the true class

    All matrices use the same color scale for comparison.
    """

    class_names = ["Normal", "AF", "Other", "Noisy"]

    n_experiments = len(confusion_results)

    # Number of plots per row
    ncols = 3

    # Calculate required rows
    nrows = math.ceil(n_experiments / ncols)

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        constrained_layout=True,
        squeeze=False
    )

    axes = axes.flatten()

    for idx, experiment in enumerate(confusion_results):

        ax = axes[idx]

        cm = experiment["matrix"]

        name = experiment["name"]

        macro_f1 = experiment["macro_f1"]

        # ----------------------------------------------
        # NORMALIZE CONFUSION MATRIX BY TRUE CLASS
        # ----------------------------------------------

        row_sums = cm.sum(axis=1, keepdims=True)

        cm_normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )

        # ----------------------------------------------
        # HEATMAP
        # ----------------------------------------------

        im = ax.imshow(
            cm_normalized,
            interpolation="nearest",
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        # ----------------------------------------------
        # TITLE
        # ----------------------------------------------

        ax.set_title(
            f"{name}\nMacro-F1 = {macro_f1:.3f}",
            fontsize=13,
            fontweight="bold",
            pad=12
        )

        # ----------------------------------------------
        # AXIS LABELS
        # ----------------------------------------------

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))

        ax.set_xticklabels(
            class_names,
            rotation=35,
            ha="right"
        )

        ax.set_yticklabels(class_names)

        ax.set_xlabel(
            "Predicted class",
            fontsize=11
        )

        ax.set_ylabel(
            "True class",
            fontsize=11
        )

        # ----------------------------------------------
        # ANNOTATE CELLS
        # ----------------------------------------------

        for i in range(4):

            for j in range(4):

                count = cm[i, j]

                percentage = cm_normalized[i, j] * 100

                # White text on dark blue cells
                text_color = (
                    "white"
                    if cm_normalized[i, j] > 0.5
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{count}\n({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold"
                )

        # Draw grid between cells
        ax.set_xticks(
            np.arange(-0.5, 4, 1),
            minor=True
        )

        ax.set_yticks(
            np.arange(-0.5, 4, 1),
            minor=True
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=2
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False
        )

    # ----------------------------------------------
    # HIDE UNUSED SUBPLOTS
    # ----------------------------------------------

    for idx in range(n_experiments, len(axes)):

        axes[idx].axis("off")

    # ----------------------------------------------
    # SHARED COLORBAR
    # ----------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        shrink=0.75,
        pad=0.02
    )

    cbar.set_label(
        "Proportion of true class",
        fontsize=11
    )

    # ----------------------------------------------
    # MAIN TITLE
    # ----------------------------------------------

    fig.suptitle(
        "ECG Classification — Preprocessing Comparison",
        fontsize=18,
        fontweight="bold"
    )

    # ----------------------------------------------
    # SAVE FIGURE
    # ----------------------------------------------

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"\nConfusion matrices saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    main()