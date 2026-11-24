# Results log — `<ecg_cinc2017>`, `<T10>`

**Track:** `ecg_cinc2017`
**Split unit:** `record`
**Primary metric:** `macro-F1` 
**Evaluation mode(s):** `new-record`

## Iteration log

Paste the metric straight from the harness — `rep["summary"]` prints the required shape, e.g.
`mean cohens_kappa 0.61 (sd 0.12, range 0.34-0.73 across 8 subjects)`. A pooled number with no
spread is half a result.

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-11-21 | supplied baseline, unchanged — establish the floor using 1999 records | mean Macro-F1 0.5963 (sd 0.044, range 0.557-0.663 across 5 folds) | — (baseline) | — | 2880f6b |
| 2 | 2026-11-21 | Improved r-peak detection by adding adaptive thresholding, r-peak localization and missed r-peak detection to improve rr-interval accuracy, as well as adding features for abnormal (other and AF) ecg classification and added Anova (k=11) for feature selection to improve noisy class while maintaining other calssification | mean macro_f1 0.663 (sd 0.022, range 0.633-0.689 across 5 folds) | yes |  | 4113e6 | 
| 3 | 2026-11-23 | Compared random forest, kNN and SVM, then tuned random forest depth and leaf size to improve minority class recall, selected random forest with min_samples_leaf = 5 | mean macro_f1 0.679 (sd 0.024, range 0.637-0.695 across 5 folds) | yes |  |  |
| 4 |  |  |  | yes / no |  |  |

*"Better" means better under the same honest harness — same split unit, same evaluation mode,
same seed. A change that lowers the metric can still be the right call (simpler, faster, more
robust across subjects, removes a leak). Say so in the column instead of quietly reverting it:
"kept — κ fell 0.02 but the worst-subject κ rose from 0.18 to 0.31" is a stronger result than a
silent higher mean.*

## ⚠️ Before you fill in many rows — the garden of forking paths

The harness guarantees that **no single run** leaks a subject. It cannot guarantee anything
about the *sequence* of runs. Every row in the table above is scored on the **same**
group-aware folds, and those are the same folds that produce the number you will report. Try
twelve configurations against one held-out set and report the best, and the winner was chosen
partly because it happens to suit *these particular subjects*. Its margin over the runner-up is
optimistically biased, and nothing in the report will look wrong. This is a **selection effect**,
not a leak — which is exactly why the leakage guard cannot catch it, and why it has to be
handled by how you *plan* the work rather than by the code.

It bites hardest here because the cohorts are small: a dozen comparisons on five or six subjects
is a great deal of selection pressure on very little data.

Two defensible ways to handle it. **Say in the report which one you used** — Criterion 7 grades
the quality of the comparison, and this is part of that quality:

| approach | what you do | what it costs |
|---|---|---|
| **an explicit development set** | hold out a few groups *up front* as a DEV set; compare every design option on DEV only; run the chosen pipeline **once** on the untouched evaluation folds and report that | fewer groups serving each purpose — painful on these cohorts — and a noisier DEV estimate |
| **a comparison budget** | decide in advance how many configurations you will compare (a handful, not a sweep), write them into this file **before** you run them, and report the count | you may miss a better option, and you have to resist re-opening the budget once you have seen the numbers — which is the whole discipline |

Either way, two habits that cost nothing:

- **Record the count.** "We compared 6 configurations" belongs in the report. A team that
  compares six and says so is doing better science than one that compares twenty and shows a
  longer table.
- **Read a small margin as no margin.** If the difference between two options is smaller than
  the spread across your subjects/folds, it is not a result. Say "no measurable difference; we
  kept the simpler one" — that sentence earns marks, and claiming the 0.01 does not.

**Related, and equally invisible: *where* your preprocessing is fitted.** The notebook builds
features for every recording **before** `evaluate()` cuts the folds. That is safe for everything
the scaffold ships, because all of it is **stateless** — a fixed filter, or a threshold estimated
from one epoch's own samples. It stops being safe the moment you add a stage that **learns**:
CSP, PCA/ICA, a cohort-wide z-score, a learned artifact template. Fit one of those over all the
recordings and every held-out subject has already shaped the transform that built the training
features — and `assert_no_subject_leak` will still pass, because it checks group ids and cannot
see what a filter was learned from. A learned stage must be fitted **inside** the fold: as a step
in the classifier pipeline (cloned and refit per fold) or via `select_features()`. If you added
one, record here which of the two you used.

## Decision log — the choices behind the numbers

Rows above say *what happened*; this says *what you chose and why*, which is what §16.4 asks you
to make traceable and what the report's defence is built from. There is no single correct
pipeline here — the scaffold deliberately ships options, not answers. One line per decision;
add rows as the pipeline grows, and note the alternative you rejected.
 
| Pipeline module | Option chosen | Alternatives considered | Why this one | Iteration | Revised later? |
|---|---|---|---|---|---|
| Data loading | 1,999 real PhysioNet ECG recordings | Synthetic smoke data | Establish performance on real ECG recordings | 1 | — |
| Preprocessing | `none` | Bandpass, notch, wavelet | Preserve original preprocessing as the reference | 1 | — |
| Feature extraction | Original 11 features | Additional HRV and quality features | Establish performance before modifying features | 1 | — |
| Feature selection | `none` | ANOVA, mutual information | Preserve original baseline configuration | 1 | — |
| Classification | Supplied Random Forest, balanced | Tuned RF, SVM | Establish reference performance before model comparisons | 1 | — |
| Inference | Original implementation | — | No modifications required for reference baseline | 1 | — |
| Reporting | Macro-F1, kappa, confusion matrix, per-class recall | — | Follow ECG evaluation requirements | 1 | — |
| Feature extraction | improved r-peak detection | peak original rpeak detection | Improved the r-peak detection, also improved macro-F1 compared to supplied baseline (0.5963 to 0.6642)| 2 | No |
| Preprocessing | None | Bandpass, Notch, Bandpass + Notch, Bandpass + Wavelet, Notch + Wavelet, Savgol | This improved the macro-F1 and kappa more than other classes, confusion matrix also showed better results for all four classes compare to other filters | 2 | Yes|
| Feature extraction | RR-entropy, RR-difference entropy, median QRS-morphology | SD QRS-morphology | These features helped identified recordings of class "other" and "AF", features describing normal ecg already exists with supplied baseline, SD morphology seemed to confuse the classifier more than help | 2 | - |
| Feature selection | Anova (k=11) | None, Variance, Anova (k=8), Tree, Lasso | This was the only classifier that improved the noisy class WHILE not sacreficing the other classes | 2 | - |
| Classification | Keep defualt random forest classifier | kNN and SVM  with different settings | This option maintained a good macro-F1 compared to other calssifiers | 3 | Yes |
| Classification | Random forest (min smaples per leaf = 5) | kNN, SVM and Random forest with different settings | This option gave a good balance between macro-F1 and minority class recall | 3 | - |
| Classification | Keep random forest (min smaples per leaf = 5) | Differend random forests with changed max depth and min sample leaf size | Other classifiers had worse macro-F1 or worse balanced accuracy | 3 | - |
|---|---|---|---|---|---|


## Revisions — the ones that went **backwards** (Criterion 9's actual evidence)

Adding iterations forward is a to-do list. What this section wants is the place a number
downstream sent you back **up** the pipeline. One row is enough; two is a good project.
The notebook's "Decision points on this track" section has a symptom → stage table to
diagnose from.

| # | The downstream result that triggered it | Which earlier decision it indicted | What you changed | What happened to the metric |
|---|---|---|---|---|
| 1 | Realised that developmental data was also used to evaluated model | stage 2 — no per-recording normalisation | added another 1999 records, that are only used for evaluation | decreaseda a bit but now by a substantial amount |
| 2 | Default random forest acheived developmet macro-F1 0.688 but noisy recall was only 0.47 suggesting room for improvement | Earlier decision to retain the deafult random forest classifier | Revisited configuration adn tested random forest depth and min leaf size | Developmental macro-F1 decreased (0.688 -> 0.674) but balanced accuracy increased (0.656 ->  0.749), AF recall increased (0.69 -> 0.79) and noisy recall increased (0.47 -> 0.76) |

## Who did what

One line per person per iteration — the honest-disclosure requirement of §16.7, and the evidence
`INDIVIDUAL_ASSESSMENT.md` asks for. The seven modules do **not** have to map one-to-one onto
people: one person may own several modules, two people may share one, and ownership may rotate
between iterations. Record what actually happened.

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Hampus | The whole group ran the default baseline in the beginning of the project but results got lost, Hampus ran the supplied baseline again i python virtual enviroment while having the correct library versions and asserting no holdouts |  |
| 2 | Hampus Eriksson | Data loading, classification, Reporting | Bingchen Wei|
| 2 | Sofia Remle | Preprocessing, Reporting | Hampus Eriksson |
| 2 | Bingchen Wei | Feature extraction, feature selection | Sofia Remle |
| 3 | Hampus Eriksson |---|---|
| 3 | Sofia Remle |---|---|
| 3 | Bingchen Wei |---|---|
| 4 | Hampus Eriksson |---|---|
| 4 | Sofia Remle |---|---|
| 4 | Bingchen Wei |---|---|

## Final numbers (fill in once, at the end)

| | Value | Under what split |
|---|---|---|
| Primary metric, development (mean + spread) |  |  |
| Secondary metrics (macro-F1 / balanced accuracy / per-class recall) |  |  |
| Held-out set — **scored once, never tuned against** |  |  |
| Supplied baseline, same harness |  |  |
| Yardstick from the dataset card (human ceiling / benchmark / chance) |  |  |
| **Configurations compared** before settling (a number) |  | on DEV folds / on the evaluation folds — say which |
| How option comparison was kept separate from final reporting |  | *e.g.* "separate DEV groups" / "budget of 6, fixed in advance" |

*Grading: `CAPSTONE_REPORT_RUBRIC.md` (team, 30 pts) and `INDIVIDUAL_ASSESSMENT.md`
(individual, 10 pts).*
