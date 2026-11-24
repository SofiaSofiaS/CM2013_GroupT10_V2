"""
Test different classifiers perfromance on baseline, classifiers tested are:
    1. kNN
    2. SVM
    3. Random forest

Purpose:
    FILL IN THE PURPOSE HERE

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results:

    Comments:
    This showed some interesting results between the different classifiers.
    First: The default classifier already uses Random forest with n = 200, so having no classifier and random forest with n = 200 shows the same
    results since they use the same classifier. This random forest showed the highest macro-F1 but the worse classification of minority classes (AF and noisy)

    Second: SVM showed the lowest macro-F1 score while having the best classification of minority classes. This shows that between these classifiers
    there seems to be a trade-off between correctly classifying minority classes and having a high macro-F1 score. We also see that increasing 
    C from 1 to 10 increased the macro-F1 a bit but decreases the minority class classification. 

    Next step would be to tune the classifiers and see if the classification could be improved:
    Random forest: Set max depth, try 5 and 10; increase min samples per leaf, try 5
    k-NN: Increase/decrease k, try 3 and 10 keeping distance weights
    SVM: keep C=1, change gamma, try auto
    

    None:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.92 | 0.00 | 0.07 | 0.01 | 1200 |
    | **A** | 0.08 | 0.69 | 0.23 | 0.01 | 173 |
    | **O** | 0.40 | 0.05 | 0.55 | 0.01 | 567 |
    | **~** | 0.39 | 0.02 | 0.12 | 0.47 | 59 |

    **macro_f1** — mean macro_f1 0.688 (sd 0.043, range 0.621-0.733 across 5 folds); pooled 0.693
    - cohens_kappa: pooled 0.579, mean over folds 0.578
    - balanced_accuracy: pooled 0.659, mean over folds 0.656
    - accuracy: pooled 0.780, mean over folds 0.780
    - worst fold: 1 (macro_f1 0.621, n=400) — read this one's errors before the mean's

    Random forest:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.92 | 0.00 | 0.07 | 0.01 | 1200 |
    | **A** | 0.08 | 0.69 | 0.23 | 0.01 | 173 |
    | **O** | 0.40 | 0.05 | 0.55 | 0.01 | 567 |
    | **~** | 0.39 | 0.02 | 0.12 | 0.47 | 59 |

    **macro_f1** — mean macro_f1 0.688 (sd 0.043, range 0.621-0.733 across 5 folds); pooled 0.693
    - cohens_kappa: pooled 0.579, mean over folds 0.578
    - balanced_accuracy: pooled 0.659, mean over folds 0.656
    - accuracy: pooled 0.780, mean over folds 0.780
    - worst fold: 1 (macro_f1 0.621, n=400) — read this one's errors before the mean's

    KNN 5 (Uniform):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.89 | 0.01 | 0.09 | 0.01 | 1200 |
    | **A** | 0.09 | 0.72 | 0.20 | 0.00 | 173 |
    | **O** | 0.40 | 0.09 | 0.50 | 0.01 | 567 |
    | **~** | 0.41 | 0.05 | 0.05 | 0.49 | 59 |

    **macro_f1** — mean macro_f1 0.662 (sd 0.051, range 0.610-0.719 across 5 folds); pooled 0.667
    - cohens_kappa: pooled 0.531, mean over folds 0.530
    - balanced_accuracy: pooled 0.650, mean over folds 0.651
    - accuracy: pooled 0.752, mean over folds 0.752
    - worst fold: 1 (macro_f1 0.610, n=400) — read this one's errors before the mean's

    KNN 5 (distance):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.88 | 0.01 | 0.10 | 0.01 | 1200 |
    | **A** | 0.10 | 0.71 | 0.20 | 0.00 | 173 |
    | **O** | 0.39 | 0.08 | 0.52 | 0.01 | 567 |
    | **~** | 0.39 | 0.03 | 0.05 | 0.53 | 59 |

    **macro_f1** — mean macro_f1 0.665 (sd 0.042, range 0.628-0.733 across 5 folds); pooled 0.669
    - cohens_kappa: pooled 0.535, mean over folds 0.534
    - balanced_accuracy: pooled 0.658, mean over folds 0.660
    - accuracy: pooled 0.753, mean over folds 0.753
    - worst fold: 0 (macro_f1 0.628, n=400) — read this one's errors before the mean's

    SVM (C=1):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.78 | 0.03 | 0.13 | 0.06 | 1200 |
    | **A** | 0.01 | 0.86 | 0.12 | 0.01 | 173 |
    | **O** | 0.26 | 0.13 | 0.57 | 0.04 | 567 |
    | **~** | 0.10 | 0.07 | 0.14 | 0.69 | 59 |

    **macro_f1** — mean macro_f1 0.632 (sd 0.023, range 0.610-0.665 across 5 folds); pooled 0.633
    - cohens_kappa: pooled 0.534, mean over folds 0.533
    - balanced_accuracy: pooled 0.728, mean over folds 0.730
    - accuracy: pooled 0.727, mean over folds 0.727
    - worst fold: 0 (macro_f1 0.610, n=400) — read this one's errors before the mean's

    SVM (C=10):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.83 | 0.02 | 0.12 | 0.04 | 1200 |
    | **A** | 0.02 | 0.76 | 0.20 | 0.02 | 173 |
    | **O** | 0.27 | 0.13 | 0.58 | 0.02 | 567 |
    | **~** | 0.24 | 0.03 | 0.14 | 0.59 | 59 |

    **macro_f1** — mean macro_f1 0.637 (sd 0.035, range 0.604-0.684 across 5 folds); pooled 0.640
    - cohens_kappa: pooled 0.549, mean over folds 0.548
    - balanced_accuracy: pooled 0.691, mean over folds 0.692
    - accuracy: pooled 0.744, mean over folds 0.744
    - worst fold: 4 (macro_f1 0.604, n=399) — read this one's errors before the mean's

"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track
from report import summarize_report, plot_confusion
import matplotlib.pyplot as plt
import math
def main():

    print("\n======================================")
    print("ENTER EXPERIMENT NAME — ENTER PURPOSE")
    print("======================================")

    SEED = 0
    DEV_SUBSET = 2000
    EVAL_SUBSET = 2000

    USE_REAL = True
    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"

    configs = {
        "None": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },
        "Random forest, 200": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
        },

        "5 Nearest Neighbors (uniform)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "knn",
            "n_neighbors": 5,
            "weights": "uniform"
        },

        "5 Nearest Neighbours (weights)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "knn",
            "n_neighbors": 5,
            "weights": "distance"
        },

        "SVM (RBF, C=1)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "svm",
            "kernel": "rbf",
            "C":1,
        },

        "SVM (RBF, C=10)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "svm",
            "kernel": "rbf",
            "C": 10,
        },
    }


    track = ECGCinC2017Track()

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

    printDataInfo(track, records)
    
    n_configs = len(configs)
    n_cols = 3
    n_rows = math.ceil(n_configs / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (name, cfg) in enumerate(configs.items()):
        print("\nConfiguration")
        print(cfg)
        current_track = ECGCinC2017Track()
        print("\nExtracting features...")
        X, y, groups = current_track.build_dataset(records, cfg=cfg)
        report = current_track.evaluate(X, y, groups, cfg=cfg)
        current_track.report(report)
        plot_confusion(
            report["y_true"],
            report["y_pred"],
            labels=track.meta.classes,
            title=name,
            ax=axes[i]
        )

    for ax in axes[len(configs):]:
        ax.set_visible(False)

    plt.tight_layout()
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