"""
Evaluation script, copy this script into another file when running evaluating iteration.
Do not run this script to do experiments, this script only uses evaluation data, experimenting on evaluation data can inflate the evaluation score

Iteration:
    Iteration 2 - Preprocessing, feature extraction, feature selection

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the evaluation set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Result:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.91 | 0.01 | 0.07 | 0.01 | 1200 |
    | **A** | 0.07 | 0.66 | 0.26 | 0.01 | 173 |
    | **O** | 0.37 | 0.05 | 0.56 | 0.02 | 567 |
    | **~** | 0.34 | 0.02 | 0.20 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.663 (sd 0.022, range 0.633-0.689 across 5 folds); pooled 0.664
    - cohens_kappa: pooled 0.577, mean over folds 0.575
    - balanced_accuracy: pooled 0.644, mean over folds 0.643
    - accuracy: pooled 0.777, mean over folds 0.777
    - worst fold: 2 (macro_f1 0.633, n=400) — read this one's errors before the mean's
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track
from report import summarize_report, plot_confusion
import matplotlib.pyplot as plt

def main():

    print("\n======================================")
    print("ENTER EXPERIMENT NAME — ENTER PURPOSE")
    print("======================================")

    SEED = 0
    DEV_SUBSET = 2000
    EVAL_SUBSET = 2000

    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"
    cfg = {
        "seed" : SEED,
        "preprocess": "none",
        "r_peak": "R2",
        "select": "anova",
        "select_k": 11,
        "imbalance": "balanced",
        "cv_max_splits": 5,
    }

    track = ECGCinC2017Track()
    print("\nConfiguration")
    print(cfg)

    # ----------------------------------------------
    # 2. DOWNLOAD AND LOAD REAL ECG DATA, MAKE SURE ONLY EVALUATION DATA IS BEING USED
    # ----------------------------------------------
    cache_dir_dev = PROJECT_ROOT / "data_cache" / "dev"
    cache_dir_eval = PROJECT_ROOT / "data_cache" / "eval"

    print("\nDownloading ECG recordings for development...")
    track.download(str(cache_dir_dev), subset = DEV_SUBSET, seed = SEED)
    dev_records = track.load(str(cache_dir_dev))
    dev_ids = {rec.group for rec in dev_records}
    track.assert_no_holdout(dev_records)

    print("\nDownloading ECG recordings for evaluation...")
    track.download(str(cache_dir_eval), subset = EVAL_SUBSET, seed = SEED, exclude_ids=dev_ids)
    eval_records = track.load(str(cache_dir_eval))
    eval_ids = {rec.group for rec in eval_records}

    if not dev_ids.isdisjoint(eval_ids):
        raise ValueError("Rerords found in both develepment set and evaluation set")
    print("No developmental data found inside evaluation data")
    
    track.assert_no_holdout(eval_records)

    records = eval_records
    
    #This print function together with the other print functions are defined below the main function
    printDataInfo(track, records)

    # ----------------------------------------------
    # 3. PREPROCESSING + FEATURE EXTRACTION
    # ----------------------------------------------

    print("\nExtracting features...")

    X, y, groups = track.build_dataset(records, cfg = cfg)

    printFeatureInfo(X, y, groups)

    # ----------------------------------------------
    # 4. MACHINE LEARNING + CROSS-VALIDATION
    # ----------------------------------------------

    print("\nRunning 5-fold GroupKFold...")

    report = track.evaluate(X, y, groups, cfg = cfg)

    # ----------------------------------------------
    # 5. EVALUATION RESULTS
    # ----------------------------------------------

    track.report(report)

    plot_confusion(
        report["y_true"],
        report["y_pred"],
        labels=track.meta.classes,
        title="ECG Classification — Confusion Matrix"
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





if __name__ == "__main__":
    main()