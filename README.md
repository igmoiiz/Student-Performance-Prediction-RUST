# Student Performance Prediction (Rust + linfa)

Predict a student's final grade (`A`/`B`/`C`/`D`/`F`) from study habits,
attendance, and background features. Two model families are provided, both
built in Rust with the [linfa](https://github.com/rust-ml/linfa) ecosystem:

- **Logistic regression** (`linfa-logistic`, L-BFGS) — `cargo run -- train`
- **Tree-based models** (`linfa-trees` + a custom random forest) — `cargo run -- train-tree`:
  a decision tree, a class-weighted random forest, and a soft-voting
  ensemble of logistic regression + random forest.

Data loading (`csv`), linear algebra (`ndarray`), metrics, and visualizations
(`plotters`, SVG output) — with zero Python.

## Results

### Test set (200 held-out samples, seed 42)

| Model | Accuracy | F1-A | F1-B | F1-C | F1-D | F1-F |
| --- | --- | --- | --- | --- | --- | --- |
| Majority-class baseline (always `B`) | 0.365 | — | — | — | — | — |
| Decision tree (depth 6, leaf 3) | 0.525 | 0.596 | 0.552 | 0.533 | 0.077 | 0.000 |
| Random forest (300 trees, class-weighted) | 0.560 | 0.695 | 0.527 | 0.510 | 0.458 | 0.000 |
| Ensemble (logit 0.6 + forest 0.4) | 0.595 | 0.685 | 0.573 | 0.605 | 0.357 | 0.000 |
| **Logistic regression** | **0.605** | 0.692 | 0.597 | 0.612 | 0.308 | 0.000 |

> Note: tree results drift by a few points between runs. `linfa-trees` resolves
> tied class counts in leaf nodes by iterating a `HashMap`, whose iteration
> order is randomized per process (logistic regression is fully deterministic).
> The table above is a representative run; expect roughly ±1-2 percentage
> points on the tree rows across runs.

**What improved:** the model's worst failure was the rare `D` grade (F1 0.308
with logistic, which mostly blurred `D` into `C`). Class-weighting the random
forest's bootstrap sampling (sampling rows inversely proportional to class
frequency) lifts `D` F1 to **0.458** — a 49% relative gain — while keeping
accuracy within 4.5 points of the best model. The ensemble matches logistic's
accuracy closely while also improving `D` (0.357).

**Honest caveat:** on this near-linear, ordinal dataset (grades track
`previous_grade`/`attendance` almost monotonically) the linear model remains
the accuracy leader; tree models' value here is diversity and minority-class
handling, not raw accuracy. `F` has only 12 samples in the whole dataset and 2
in the test set — no model can realistically recover it.

### Why the random forest is class-weighted

The dataset is imbalanced (A: 284, B: 354, C: 261, D: 89, F: 12). A plain
random forest quickly collapses toward the majority classes. By bootstrapping
rows with probability inversely proportional to class frequency, rare grades
appear in more trees, so the ensemble can actually emit minority predictions.

## Pipeline

```
data/dataset.csv
      │  csv crate
      ▼
data::load  ->  Vec<Record> (1000 students, 12 fields)
      │  rand seeded shuffle (seed 42)
      ▼
80% train / 20% test
      │  Preprocessor::fit on TRAIN ONLY (no test leakage)
      ▼
Feature engineering -> 12 columns
      │
      ├──▶ MultiLogisticRegression (L-BFGS)         ──▶ outputs/model.bin
      │
      └──▶ DecisionTree  (linfa-trees, Gini)
           RandomForest (custom bagging + feature subsetting,
                         inverse-frequency bootstrap)
           Ensemble = 0.6·P(logit) + 0.4·P(forest)   ──▶ outputs/model_tree.bin
      │
      ▼
metrics + confusion matrix + per-class P/R/F1  ->  prints + predictions.csv
      │
      ▼
outputs/plots/*.svg (13 labelled charts)
```

The fitted `Preprocessor` is serialized with each model bundle, so the
`predict`/`predict-tree` subcommands apply exactly the same scaling and
one-hot encoding used during training. `predict-tree` reproduces the ensemble
(its blend weight is stored in the bundle).

## Feature engineering

| Type | Columns | Encoding |
| --- | --- | --- |
| Numeric (4) | `study_time_hours`, `attendance_percent`, `sleep_hours`, `previous_grade` | z-score normalisation (mean 0, std 1) |
| Categorical (gender) | `gender` | one-hot (`gender_Male`) |
| Categorical (parental_education) | `parental_education` | one-hot, 4 columns, `None` dropped as baseline |
| Flags (3) | `internet_access`, `extracurricular_activities`, `part_time_job` | binary `Yes`/`No` |

**Dropped:** `final_exam_score`. It directly determines the final grade, so
including it would leak the target (inflate accuracy and make the model
useless for real prediction). `student_id` is an identifier, not a feature.

## Getting started

Requires a recent stable Rust toolchain (`cargo` 1.94+).

```bash
# 1. Train the logistic regression, evaluate on the held-out test set
cargo run -- train

# 2. Train the tree-based models (decision tree + class-weighted random
#    forest + logistic/forest ensemble) and the model comparison
cargo run -- train-tree

# 3. Predict a grade from a single CSV line (paste at the prompt)
cargo run -- predict        # logistic model
cargo run -- predict-tree   # logistic + random forest ensemble
# e.g.  999,Female,4.6,85.3,8.1,High School,No,No,Yes,52.2,82.2,B

# 4. Run the unit tests
cargo test
```

## Project structure

```
data/dataset.csv            Input dataset (1000 students, 12 fields)
src/main.rs                 CLI: train/train-tree + predict/predict-tree, model bundles
src/data.rs                 CSV loading + Record struct
src/features.rs             Preprocessor: z-scoring, one-hot encoding, grade mapping
src/metrics.rs              Confusion matrix, accuracy, per-class precision/recall/F1
src/forest.rs               Random forest (bagging + feature subsetting, class weights)
src/plots.rs                plotters-based charts (SVG)
outputs/predictions.csv     Logistic test-set predictions with actual grades
outputs/tree_predictions.csv Ensemble test-set predictions with actual grades
outputs/model.bin           Serialized logistic model + preprocessor (gitignored)
outputs/model_tree.bin      Serialized forest + tree + logit + preprocessor (gitignored)
outputs/plots/*.svg         Generated charts
LICENSE.md                  MIT license
```

## Charts (`outputs/plots/`)

SVG output renders text with the system font stack (no bundled fonts needed),
and displays in any browser or on GitHub.

| File | What it shows |
| --- | --- |
| `01_grade_distribution.svg` | Count of students per final grade. |
| `02_numeric_feature_histograms.svg` | Raw distributions of the 4 numeric features. |
| `03_attendance_vs_study_time_scatter.svg` | Attendance % vs study time, coloured by grade. |
| `04_previous_grade_by_final_grade.svg` | Mean previous grade (+/- 1 std) per grade — how well the feature separates classes. |
| `05_scaled_feature_histograms.svg` | The same features after z-score scaling. |
| `06_confusion_matrix.svg` | Logistic confusion matrix on the test set. |
| `07_per_class_metrics.svg` (+ `_recall`, `_f1`) | Per-class precision, recall, F1 for logistic. |
| `08_feature_importance.svg` | Logistic mean |weight| per feature. |
| `09_confusion_matrix_decision_tree.svg` | Decision tree confusion matrix. |
| `10_confusion_matrix_random_forest.svg` | Random forest confusion matrix. |
| `11_confusion_matrix_ensemble.svg` | Ensemble confusion matrix. |
| `12_model_comparison.svg` | Accuracy + per-class F1 grouped bars for all four models. |
| `13_tree_feature_importance.svg` | Random forest mean relative impurity decrease per feature. |

## Rust vs Python for machine learning

Both ecosystems can build this model. The trade-offs matter mostly at scale
and in production:

| Concern | Rust (`linfa`) | Python (`scikit-learn`) |
| --- | --- | --- |
| Dependencies | Compile-time, static binary | Runtime, `pip` environment management |
| Speed | Very fast; training here runs in well under a second | Fast, but interpreter + numpy overhead per call |
| Deployment | Single static-ish binary, no runtime | Needs a Python runtime, venv, model pickling |
| Ecosystem breadth | Smaller; growing (`linfa`, `burn`, `candle`, `ndarray`) | Huge: sklearn, torch, xgboost, etc. |
| Interop | Trivial to call from other Rust code | Best-in-class for notebooks / experimentation |
| Type safety | Compiler catches shape/mismatch bugs at build time | Errors surface at runtime |

For a small tabular problem like this one, the two are interchangeable in
accuracy. Rust wins when you need a self-contained, fast, embeddable artifact
(e.g. a CLI tool or a server module) without babysitting a Python runtime;
Python wins when you want the fastest path to experimentation with the largest
collection of ready-made libraries. Note that `scikit-learn` ships a mature
`RandomForestClassifier` with `class_weight="balanced_subsample"` built in —
the equivalent here is implemented by hand in `src/forest.rs`.

## License

[MIT](LICENSE.md) — Copyright (c) 2026 Moiz Baloch.
