"""
Testing adding different feature extraction metrics, see if they improve current baseline
Since this is just testing what happens when we add more features, only a single baseline is run at a time. Between runs, another feature is added in the track
This test was performed on 5 runs, the improved r-peak detection is used:
    1. Original 11 features, to establish something to compare against
    2. RR-interval entropy is added, 12 features in total
    3. RR-interval difference entropy is added, 13 features in total
    4. QRS-morphology sd is added, 14 features in total
    5. QRS-morphology median is added, 15 features in total
    6. QRS-morphology median is but QRS-morphology is removed, 14 features in total

Purpose:
    To test how RR-interval entropy, RR-interval difference entropy and QRS-morphology impacts classification, confusion matrix and macro-F1 score

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results:
    1. 11 orignal features: 
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.01 | 0.06 | 0.00 | 1200 |
    | **A** | 0.10 | 0.61 | 0.27 | 0.02 | 173 |
    | **O** | 0.41 | 0.06 | 0.52 | 0.01 | 567 |
    | **~** | 0.41 | 0.02 | 0.14 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.657 (sd 0.042, range 0.599-0.710 across 5 folds); pooled 0.664
    - cohens_kappa: pooled 0.557, mean over folds 0.557
    - balanced_accuracy: pooled 0.626, mean over folds 0.628
    - accuracy: pooled 0.772, mean over folds 0.772
    - worst fold: 4 (macro_f1 0.599, n=399) — read this one's errors before the mean's

    2. 12 features (11 original + RR-interval entropy):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.01 | 0.06 | 0.00 | 1200 |
    | **A** | 0.10 | 0.65 | 0.24 | 0.02 | 173 |
    | **O** | 0.41 | 0.05 | 0.53 | 0.01 | 567 |
    | **~** | 0.42 | 0.03 | 0.10 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.671 (sd 0.035, range 0.629-0.712 across 5 folds); pooled 0.677
    - cohens_kappa: pooled 0.570, mean over folds 0.570
    - balanced_accuracy: pooled 0.638, mean over folds 0.638
    - accuracy: pooled 0.779, mean over folds 0.779
    - worst fold: 4 (macro_f1 0.629, n=399) — read this one's errors before the mean's

    3. 13 features (11 original + RR-interval entropy + RR-interval difference entropy)
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.92 | 0.00 | 0.07 | 0.00 | 1200 |
    | **A** | 0.08 | 0.70 | 0.21 | 0.02 | 173 |
    | **O** | 0.39 | 0.05 | 0.55 | 0.01 | 567 |
    | **~** | 0.39 | 0.03 | 0.14 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.689 (sd 0.020, range 0.666-0.711 across 5 folds); pooled 0.693
    - cohens_kappa: pooled 0.585, mean over folds 0.584
    - balanced_accuracy: pooled 0.654, mean over folds 0.656
    - accuracy: pooled 0.784, mean over folds 0.784
    - worst fold: 3 (macro_f1 0.666, n=400) — read this one's errors before the mean's

    4. 14 features (11 original + RR-interval entropy + RR-interval difference entropy + QRS morphology sd)
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.00 | 0.06 | 0.01 | 1200 |
    | **A** | 0.07 | 0.71 | 0.21 | 0.01 | 173 |
    | **O** | 0.40 | 0.04 | 0.55 | 0.01 | 567 |
    | **~** | 0.44 | 0.03 | 0.10 | 0.42 | 59 |

    **macro_f1** — mean macro_f1 0.683 (sd 0.023, range 0.657-0.711 across 5 folds); pooled 0.687
    - cohens_kappa: pooled 0.592, mean over folds 0.591
    - balanced_accuracy: pooled 0.654, mean over folds 0.655
    - accuracy: pooled 0.788, mean over folds 0.788
    - worst fold: 4 (macro_f1 0.657, n=399) — read this one's errors before the mean's
    
    5. 15 features (11 original + RR-interval entropy + RR-interval difference entropy + QRS morphology sd + QRS morphology median)
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.94 | 0.00 | 0.06 | 0.01 | 1200 |
    | **A** | 0.08 | 0.69 | 0.22 | 0.01 | 173 |
    | **O** | 0.38 | 0.04 | 0.57 | 0.01 | 567 |
    | **~** | 0.42 | 0.02 | 0.10 | 0.46 | 59 |

    **macro_f1** — mean macro_f1 0.698 (sd 0.022, range 0.671-0.733 across 5 folds); pooled 0.702
    - cohens_kappa: pooled 0.607, mean over folds 0.606
    - balanced_accuracy: pooled 0.664, mean over folds 0.661
    - accuracy: pooled 0.796, mean over folds 0.796
    - worst fold: 4 (macro_f1 0.671, n=399) — read this one's errors before the mean's

    6. 14 features (11 original + RR-interval entropy + RR-interval difference entropy + QRS morphology median)
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.00 | 0.06 | 0.00 | 1200 |
    | **A** | 0.07 | 0.69 | 0.24 | 0.01 | 173 |
    | **O** | 0.38 | 0.04 | 0.57 | 0.01 | 567 |
    | **~** | 0.41 | 0.02 | 0.12 | 0.46 | 59 |

    **macro_f1** — mean macro_f1 0.699 (sd 0.026, range 0.664-0.730 across 5 folds); pooled 0.703
    - cohens_kappa: pooled 0.605, mean over folds 0.603
    - balanced_accuracy: pooled 0.662, mean over folds 0.662
    - accuracy: pooled 0.795, mean over folds 0.795
    - worst fold: 4 (macro_f1 0.664, n=399) — read this one's errors before the mean's

Notes:
    For each iteration that was added the baselines macro-F1 score increased, indicating that every added feature might be beneficial for classification.
    When adding RR-interval entropy we saw an increase in correct classification for AF (Normal, other and Noisy saw little to no change),
    and at the same time increased macro-F1 and decreased the sd and range for macro-F1 across different scores. 
    
    When adding RR-interval difference entropy the correct classification of AF increased even more, and "other" classification improved by 0.03,
    we saw the same type of improvement that we also got when adding RR-interval entropy (an increase in macro-F1 and decrease in macro-F1 sd and range)

    QRS-morphology sd showed worse classification for the noisy class, but slight improvment for AF and Normal. At the same time the macro-F1 score decreased.
    By looking at the confusion matrix, this did not seem to be beneficial for our classifier since this decreased noisy classification which we want to keep.

    QRS-morphology median also showed great improvement. The macro-F1 score increased yet again ot highest of the 5 different versions tested here. 
    Although we see a decrease in AF classifcation, all of the other classes improved

    The last version was used to see if removing QRS-morphology sd but keeping QRS-morphology median would improve the model (since this would also simplify it)
    And the score showed promising results since this increases the macro-F1 a bit. But looking at the diagonals on the confusion matrix, this added no extra
    to the correct classification (each cell is normalized so in total amount there might be some change but not in normalized value). So we cant say that
    this version actually is better, since this might just be an effect of tuning the features to the data and folds used. But since this had the best result and also 
    used one less feature than previous version, this model may be of better use.
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

    USE_REAL = True 
    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"
    cfg = {
        "seed" : SEED,
        "preprocess": "none",
        "r_peak": "R2",
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