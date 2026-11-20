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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track, _rpeaks_improved, _rpeaks

def main():

    print("\n======================================")
    print("EXPERIMENT B0 — ORIGINAL ECG BASELINE")
    print("======================================")

    SEED = 0
    SUBSET = 2000

    USE_REAL = True

    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"
    cfg = {
        "seed" : SEED,
        "preprocess": "none",
        "select": "none",
        "imbalance": "balanced",
        "cv_max_splits": 5,
    }

    # cfg = {
    #     "seed": 42,
        # "preprocess": [
        #     {"preprocess": "denoise", "powerline": "notch",
        #      "f0": 50.0, "q": 40, "harmonics": (1, 2)},
        # ],
    #     "select": "none",  # keep all 12 features
    #     "rf_n_estimators": 300,
    #     "rf_max_depth": 15,
    #     "rf_min_samples_split": 4,
    #     "rf_min_samples_leaf": 2,
    #     "rf_max_features": "sqrt",
    #     "rf_class_weight": "balanced_subsample",
    # }

    track = ECGCinC2017Track()
    print("\nConfiguration")
    print(cfg)

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



    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(len(track.meta.classes), 1, figsize=(9, 6), sharex=True)

    for ax, cls in zip(axes, track.meta.classes):

        rec = next((r for r in records if r.labels[0] == cls), None)

        if rec is None: 
            ax.set_visible(False)
            continue

        x = rec.epochs['ecg'][0]
        fs = rec.fs
        n = min(len(x), int(6*fs))
        t = np.arange(n)/fs

        ax.plot(t, x[:n], lw=0.8)
        pk = _rpeaks_improved(x, fs)
        pk = pk[pk < n]

        ax.plot(pk/fs, x[pk], 'r.', ms=8)
        ax.set_ylabel(f'class {cls}')

    axes[-1].set_xlabel('time (s)')
    axes[0].set_title('CinC-2017 rhythm strips (red = R-peaks)')
    plt.tight_layout(); plt.show()


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