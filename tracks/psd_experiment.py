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
from scipy.signal import welch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track

def main():

    print("\n======================================")
    print("EXPERIMENT B0 — ORIGINAL ECG BASELINE")
    print("======================================")

    SEED = 0
    SUBSET = 2000

    USE_REAL = True

    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"

    track = ECGCinC2017Track()

    # ----------------------------------------------
    # 2. DOWNLOAD AND LOAD REAL ECG DATA
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
        "P5_bandpass_notch_with_q": {
            #Configuration 5
            "seed" : SEED,
            "preprocess": "denoise",
            "powerline": "notch",
            "q": 100, "harmonics": (1, 2),
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
    compare_configs_PSD_welch(track, records, configs, f_min=0, f_max= 150)

def compare_configs_PSD_welch(track, records, configs, f_min = 0, f_max = 150):

    class_names = {
        "N": "Normal",
        "A": "AF",
        "O": "Other",
        "~": "Noisy"
    }

    fig, axes = plt.subplots(len(track.meta.classes), 1, figsize=(12, 12), sharex=True, constrained_layout=True)


    for ax, cls in zip(axes, track.meta.classes):
        rec = next((r for r in records if r.labels[0] == cls), None)
        
        if rec is None: 
            ax.set_visible(False)
            continue

        for name, cfg in configs.items():
            if cfg is not None:
                rec = track.preprocess(rec, cfg=cfg)
            ecg = rec.epochs['ecg'][0]
            fs = rec.fs

            # ----------------------------------------------
            # WELCH
            # ----------------------------------------------
            f_w, psd_w = welch(
                ecg,
                fs=fs,
                window="hann",
                nperseg=2048,
                noverlap=1024,
                detrend="constant"
            )

            ax.plot(
                10 * np.log(f_w), 
                10 * np.log10(psd_w +1e-12),
                label = name,
                linewidth=1.5
            )

        ax.set_title(
            f"{class_names[cls]} — Record {rec.group}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlim(f_min, 10* np.log(f_max))

        ax.set_ylabel("PSD (dB/Hz)")
        ax.grid(True, alpha=0.3)

        ax.legend()

    axes[-1].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "Preprocessing comparison — ECG Power Spectral Density",
        fontsize=16,
        fontweight="bold"
    )

    plt.show()


    
def compare_bartlett_to_welch(track, records, cfg=None, f_min=0, f_max = 150):

    class_names = {
        "N": "Normal",
        "A": "AF",
        "O": "Other",
        "~": "Noisy"
    }

    fig, axes = plt.subplots(len(track.meta.classes), 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    for ax, cls in zip(axes, track.meta.classes):
        rec = next((r for r in records if r.labels[0] == cls), None)
        
        if rec is None: 
            ax.set_visible(False)
            continue
        if cfg is not None:
            rec = track.preprocess(rec, cfg=cfg)
        
        ecg = rec.epochs['ecg'][0]
        fs = rec.fs

        # ----------------------------------------------
        # BARTLETT
        # ----------------------------------------------
        f_b, psd_b = welch(
            ecg,
            fs=fs,
            window="boxcar",
            nperseg=2048,
            noverlap=0,
            detrend="constant"
        )

        # ----------------------------------------------
        # WELCH
        # ----------------------------------------------
        f_w, psd_w = welch(
            ecg,
            fs=fs,
            window="hann",
            nperseg=2048,
            noverlap=1024,
            detrend="constant"
        )

        # ----------------------------------------------
        # PLOT
        # ----------------------------------------------

        ax.plot(
            f_b, 
            10 * np.log10(psd_b +1e-12),
            label = "Bartlett",
            linewidth=1.5
        )
        ax.plot(
            f_w, 
            10 * np.log10(psd_w +1e-12),
            label = "Welch",
            linewidth=1.5
        )

        ax.set_title(
            f"{class_names[cls]} — Record {rec.group}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlim(f_min, f_max)

        ax.set_ylabel("PSD (dB/Hz)")
        ax.grid(True, alpha=0.3)

        ax.legend()

    axes[-1].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "Bartlett vs Welch — ECG Power Spectral Density",
        fontsize=16,
        fontweight="bold"
    )

    plt.show()






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



if __name__ == "__main__":
    main()