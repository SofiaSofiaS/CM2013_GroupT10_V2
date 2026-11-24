"""
Test different classifiers perfromance on baseline, all classifiers are random forest with:
    Max depth-> |  None      |  5        |  10      |
    Min leaf    |            |           |          |
        |  1    |  None, 1   |  5, 1     | 10, 1    |
        V  3    |  None, 3   |  5, 5     | 10, 3    |
           5    |  None, 5   |  5, 10    | 10, 5    |

Total classifiers test: 9
N_estimator = 200 for all classifiers

Purpose:
    Further testing of classifiers, trying to tune them

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results:

    Comments:
    This is part 3, check part 1 and 2 of classifier experiments first
    These comments on the results regard the developmental set cross validation only, peformance on evaluation set corss validation can differ.

    The highest macro-F1 we got was 0.691 (from max depth = 10) beating the previous iterations macro-F1 (0.688) on this data with these folds (not the score on the evaluation data)
    with an even better sd (0.021 vs 0.043) and range (0.663-0.718 vs 0.621-0.733). Accuracy of minory classes improved as well, not bu a substantial amount
    but i little bit for every class other than "normal" which decreased with 0.03. 

    Increasing min samples per leaf increased classification of noisy records. The random forest classifier with a min samples per leaf of 5
    and max depth of 5 predicted the noisy classes the best (0.8, 0.33 increase compare to previous) but sacrificing classification of the normal class
    (0.78, 0.14 decrease compared to previous) which brought down it's macro-F1 (0.635).

    With min samples per leaf = 5, and max depth = None or 10, we got a balance between classification of minority classes and keeping the accuracy of the normal class. 
    These two classifier also had the best balanced accuracy (0.749 and 0.746) while maintaining a high macro-F1  (0.674 and 0.675).

    This macro-F1 is lower than the previous baseline (on this data with these folds), so here it is worth considering if the improved balanced accuracy 
    is worth the decreased macro-F1. And we would say that it is since the noisy class was heavily missclassified before, and AF was better but still bad.
    With this decreased macro-F1 (due to worse classification of a majority class) we greatly improve the accruacy of these minority classes. 


    And it even managed to predict noisy and "AF" classes more accuratly 

    Max depth = None, Min samples per leaf = 1 (this is also the default classifier):
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

    Max depth = None, Min samples per leaf = 3:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.87 | 0.01 | 0.09 | 0.03 | 1200 |
    | **A** | 0.03 | 0.77 | 0.18 | 0.01 | 173 |
    | **O** | 0.32 | 0.07 | 0.57 | 0.04 | 567 |
    | **~** | 0.22 | 0.03 | 0.14 | 0.61 | 59 |

    **macro_f1** — mean macro_f1 0.672 (sd 0.017, range 0.647-0.693 across 5 folds); pooled 0.674
    - cohens_kappa: pooled 0.580, mean over folds 0.579
    - balanced_accuracy: pooled 0.705, mean over folds 0.705
    - accuracy: pooled 0.769, mean over folds 0.769
    - worst fold: 1 (macro_f1 0.647, n=400) — read this one's errors before the mean's

    Max depth = None, Min samples per leaf = 5:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.85 | 0.01 | 0.10 | 0.04 | 1200 |
    | **A** | 0.03 | 0.79 | 0.16 | 0.01 | 173 |
    | **O** | 0.29 | 0.09 | 0.57 | 0.05 | 567 |
    | **~** | 0.14 | 0.02 | 0.08 | 0.76 | 59 |

    **macro_f1** — mean macro_f1 0.674 (sd 0.034, range 0.623-0.715 across 5 folds); pooled 0.675
    - cohens_kappa: pooled 0.581, mean over folds 0.580
    - balanced_accuracy: pooled 0.745, mean over folds 0.749
    - accuracy: pooled 0.764, mean over folds 0.764
    - worst fold: 1 (macro_f1 0.623, n=400) — read this one's errors before the mean's

    Max depth = 5, Min samples per leaf = 1:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.79 | 0.02 | 0.16 | 0.03 | 1200 |
    | **A** | 0.02 | 0.83 | 0.14 | 0.01 | 173 |
    | **O** | 0.25 | 0.14 | 0.56 | 0.05 | 567 |
    | **~** | 0.15 | 0.03 | 0.12 | 0.69 | 59 |

    **macro_f1** — mean macro_f1 0.637 (sd 0.039, range 0.587-0.681 across 5 folds); pooled 0.639
    - cohens_kappa: pooled 0.523, mean over folds 0.523
    - balanced_accuracy: pooled 0.718, mean over folds 0.719
    - accuracy: pooled 0.723, mean over folds 0.723
    - worst fold: 1 (macro_f1 0.587, n=400) — read this one's errors before the mean's

    Max depth = 5, Min samples per leaf = 3:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.78 | 0.02 | 0.16 | 0.04 | 1200 |
    | **A** | 0.02 | 0.86 | 0.12 | 0.01 | 173 |
    | **O** | 0.25 | 0.14 | 0.55 | 0.05 | 567 |
    | **~** | 0.10 | 0.03 | 0.14 | 0.73 | 59 |

    **macro_f1** — mean macro_f1 0.637 (sd 0.033, range 0.584-0.666 across 5 folds); pooled 0.638
    - cohens_kappa: pooled 0.524, mean over folds 0.523
    - balanced_accuracy: pooled 0.730, mean over folds 0.732
    - accuracy: pooled 0.721, mean over folds 0.721
    - worst fold: 1 (macro_f1 0.584, n=400) — read this one's errors before the mean's

    Max depth = 5, Min samples per leaf = 5:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.78 | 0.02 | 0.15 | 0.05 | 1200 |
    | **A** | 0.01 | 0.84 | 0.13 | 0.02 | 173 |
    | **O** | 0.25 | 0.15 | 0.55 | 0.05 | 567 |
    | **~** | 0.08 | 0.02 | 0.10 | 0.80 | 59 |

    **macro_f1** — mean macro_f1 0.635 (sd 0.039, range 0.575-0.670 across 5 folds); pooled 0.635
    - cohens_kappa: pooled 0.519, mean over folds 0.519
    - balanced_accuracy: pooled 0.741, mean over folds 0.744
    - accuracy: pooled 0.717, mean over folds 0.717
    - worst fold: 1 (macro_f1 0.575, n=400) — read this one's errors before the mean's

    Max depth = 10, Min samples per leaf = 1:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.89 | 0.01 | 0.09 | 0.01 | 1200 |
    | **A** | 0.06 | 0.72 | 0.21 | 0.01 | 173 |
    | **O** | 0.34 | 0.06 | 0.59 | 0.01 | 567 |
    | **~** | 0.31 | 0.02 | 0.15 | 0.53 | 59 |

    **macro_f1** — mean macro_f1 0.691 (sd 0.021, range 0.663-0.718 across 5 folds); pooled 0.696
    - cohens_kappa: pooled 0.592, mean over folds 0.591
    - balanced_accuracy: pooled 0.683, mean over folds 0.681
    - accuracy: pooled 0.781, mean over folds 0.781
    - worst fold: 1 (macro_f1 0.663, n=400) — read this one's errors before the mean's

    Max depth = 10, Min samples per leaf = 3:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.86 | 0.01 | 0.10 | 0.03 | 1200 |
    | **A** | 0.03 | 0.80 | 0.16 | 0.01 | 173 |
    | **O** | 0.29 | 0.08 | 0.59 | 0.04 | 567 |
    | **~** | 0.22 | 0.02 | 0.15 | 0.61 | 59 |

    **macro_f1** — mean macro_f1 0.679 (sd 0.025, range 0.653-0.712 across 5 folds); pooled 0.681
    - cohens_kappa: pooled 0.592, mean over folds 0.591
    - balanced_accuracy: pooled 0.716, mean over folds 0.716
    - accuracy: pooled 0.774, mean over folds 0.774
    - worst fold: 1 (macro_f1 0.653, n=400) — read this one's errors before the mean's

    Max depth = 10, Min samples per leaf = 5:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.85 | 0.01 | 0.10 | 0.04 | 1200 |
    | **A** | 0.03 | 0.80 | 0.16 | 0.01 | 173 |
    | **O** | 0.28 | 0.09 | 0.58 | 0.05 | 567 |
    | **~** | 0.15 | 0.02 | 0.08 | 0.75 | 59 |

    **macro_f1** — mean macro_f1 0.675 (sd 0.031, range 0.627-0.711 across 5 folds); pooled 0.676
    - cohens_kappa: pooled 0.584, mean over folds 0.584
    - balanced_accuracy: pooled 0.744, mean over folds 0.746
    - accuracy: pooled 0.766, mean over folds 0.766
    - worst fold: 1 (macro_f1 0.627, n=400) — read this one's errors before the mean's

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

        "Random forest, min samples leaf = 3": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 3,
        },
        "Random forest, min samples leaf = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 5,
        },
        "Random forest, max depth = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "max_depth": 5,
        },
        
        "Random forest, max depth = 5 + min samples leaf = 3": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 3,
            "max_depth": 5
        },        
        "Random forest, max depth = 5 + min samples leaf = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 5,
            "max_depth": 5
        },
                
        "Random forest, max depth = 10": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "max_depth": 10
        },        
        "Random forest, max depth = 10 + min samples leaf = 3": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 3,
            "max_depth": 10
        },        
        "Random forest, max depth = 10 + min samples leaf = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 5,
            "max_depth": 10
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