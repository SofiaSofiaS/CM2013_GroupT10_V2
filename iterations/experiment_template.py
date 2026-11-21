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

    USE_REAL = False 
    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"
    cfg = {
        "seed" : SEED,
        "preprocess": "none",
        "select": "none",
        "imbalance": "balanced",
        "cv_max_splits": 5,
    }

    track = ECGCinC2017Track()
    print("\nConfiguration")
    print(cfg)

    # ----------------------------------------------
    # 2. DOWNLOAD AND LOAD REAL ECG DATA
    # ----------------------------------------------
    if USE_REAL:
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
        print("Developmental and evaluation records as disjoint")
        
        track.assert_no_holdout(eval_records)
        records = dev_records

    else:
        records = track.smoke()
        print(f"Generated {len(records)} synthetic ECG recordings")

    
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

    summarize_report(report)

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