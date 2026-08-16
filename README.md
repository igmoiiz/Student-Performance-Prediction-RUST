# Student Performance Prediction (Rust + linfa)

Predict a student's final grade (`A`/`B`/`C`/`D`/`F`) from study habits,
attendance, and background features using a multinomial logistic-regression
model trained with the [linfa](https://github.com/rust-ml/linfa) ecosystem.

Built entirely in Rust: data loading (`csv`), linear algebra (`ndarray`),
model training (`linfa-logistic`, L-BFGS), metrics, and visualizations
(`plotters`, SVG output) — with zero Python.

## Results

| Metric | Value |
| --- | --- |
| Test accuracy (200 held-out samples) | **0.605** |
| Majority-class baseline (always predict `B`) | 0.365 |
| Classes | A: 284, B: 354, C: 261, D: 89, F: 12 (dataset) |

The model comfortably beats the majority baseline. Class `F` has only 12
samples, so its recall is naturally poor; see the per-class metrics plots for
the full breakdown.

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
      ▼
MultiLogisticRegression (L-BFGS, L2 alpha=1.0, 500 iterations max)
      │
      ▼
metrics + confusion matrix + per-class P/R/F1  ->  prints + predictions.csv
      │
      ▼
outputs/model.bin  (bincode, model + preprocessor together)
      │
      ▼
outputs/plots/*.svg (8 labelled charts)
```

The trained model and the fitted `Preprocessor` are serialized together into
`outputs/model.bin`, so the `predict` subcommand applies exactly the same
scaling and one-hot encoding used during training.

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
# 1. Train the model, evaluate it on the held-out test set, write artifacts
cargo run -- train

# 2. Predict a grade from a single CSV line (paste at the prompt)
cargo run -- predict
# e.g.  999,Female,4.6,85.3,8.1,High School,No,No,Yes,52.2,82.2,B

# 3. Run the unit tests
cargo test
```

## Project structure

```
data/dataset.csv          Input dataset (1000 students, 12 fields)
src/main.rs               CLI: train + predict, model bundle, metrics printing
src/data.rs               CSV loading + Record struct
src/features.rs           Preprocessor: z-scoring, one-hot encoding, grade mapping
src/metrics.rs            Confusion matrix, accuracy, per-class precision/recall/F1
src/plots.rs              plotters-based charts (SVG)
outputs/predictions.csv   Test-set predictions with actual grades
outputs/model.bin         Serialized model + preprocessor (gitignored)
outputs/plots/*.svg       Generated charts
LICENSE.md                MIT license
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
| `06_confusion_matrix.svg` | Confusion matrix on the test set. |
| `07_per_class_metrics.svg` (+ `_recall`, `_f1`) | Per-class precision, recall, F1. |
| `08_feature_importance.svg` | Mean absolute logistic-regression weight per feature. |

## Rust vs Python for machine learning

Both ecosystems can build this model. The trade-offs matter mostly at scale
and in production:

| Concern | Rust (`linfa`) | Python (`scikit-learn`) |
| --- | --- | --- |
| Dependencies | Compile-time, static binary | Runtime, `pip` environment management |
| Speed | Very fast; L-BFGS training here runs in well under a second | Fast, but interpreter + numpy overhead per call |
| Deployment | Single static-ish binary, no runtime | Needs a Python runtime, venv, model pickling |
| Ecosystem breadth | Smaller; growing (`linfa`, `burn`, `candle`, `ndarray`) | Huge: sklearn, torch, xgboost, etc. |
| Interop | Trivial to call from other Rust code | Best-in-class for notebooks / experimentation |
| Type safety | Compiler catches shape/mismatch bugs at build time | Errors surface at runtime |

For a small tabular problem like this one, the two are interchangeable in
accuracy. Rust wins when you need a self-contained, fast, embeddable artifact
(e.g. a CLI tool or a server module) without babysitting a Python runtime;
Python wins when you want the fastest path to experimentation with the largest
collection of ready-made libraries.

## License

[MIT](LICENSE.md) — Copyright (c) 2026 Moiz Baloch.
