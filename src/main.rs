mod data;
mod features;
mod metrics;
mod plots;

use features::Preprocessor;
use linfa::prelude::*;
use linfa_logistic::{MultiFittedLogisticRegression, MultiLogisticRegression};
use metrics::ClassMetrics;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

const DATASET_PATH: &str = "data/dataset.csv";
const MODEL_PATH: &str = "outputs/model.bin";
const PREDICTIONS_PATH: &str = "outputs/predictions.csv";
const TEST_FRACTION: f64 = 0.2;
const SEED: u64 = 42;
const N_CLASSES: usize = 5;

/// Serialized together so the `predict` subcommand can re-apply the exact same
/// preprocessing (scaling + one-hot encoding) as training.
#[derive(Serialize, Deserialize)]
struct ModelBundle {
    model: MultiFittedLogisticRegression<f64, i32>,
    preprocess: Preprocessor,
}

struct Split {
    train: Vec<data::Record>,
    test: Vec<data::Record>,
}

fn split_train_test(records: &[data::Record], test_fraction: f64, seed: u64) -> Split {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..records.len()).collect();
    indices.shuffle(&mut rng);
    let n_test = (records.len() as f64 * test_fraction).round() as usize;
    let test: Vec<_> = indices[..n_test].iter().map(|&i| records[i].clone()).collect();
    let train: Vec<_> = indices[n_test..].iter().map(|&i| records[i].clone()).collect();
    Split { train, test }
}

fn print_metrics(
    actual: &[i32],
    predicted: &[i32],
    cm: &Array2<usize>,
    per_class: &[(i32, ClassMetrics)],
) {
    let correct = actual.iter().zip(predicted).filter(|(a, p)| a == p).count();
    let acc = metrics::accuracy(actual, predicted);
    println!();
    println!("==================== TEST-SET EVALUATION ====================");
    println!("Samples: {}", actual.len());
    println!("Accuracy: {:.3} ({correct}/{total})", acc, total = actual.len());
    println!();
    println!("Confusion matrix (rows = actual grade, cols = predicted grade):");
    print!("        ");
    for j in 0..N_CLASSES {
        print!("{:>6}", features::ordinal_to_grade(j as i32));
    }
    println!();
    for i in 0..N_CLASSES {
        print!("{:>4}   ", features::ordinal_to_grade(i as i32));
        for j in 0..N_CLASSES {
            print!("{:>6}", cm[[i, j]]);
        }
        println!();
    }
    println!();
    println!("{:<6} {:>10} {:>10} {:>10} {:>10}", "Grade", "Precision", "Recall", "F1", "Support");
    for (label, m) in per_class {
        println!(
            "{:<6} {:>10.3} {:>10.3} {:>10.3} {:>10}",
            features::ordinal_to_grade(*label),
            m.precision,
            m.recall,
            m.f1,
            m.support
        );
    }
    println!();
    println!("Baseline to beat: always predicting the majority class (B) gives {:.3}", majority_baseline(actual));
    println!("==============================================================");
}

fn majority_baseline(actual: &[i32]) -> f64 {
    let mut counts = vec![0usize; N_CLASSES];
    for a in actual {
        counts[*a as usize] += 1;
    }
    let (majority, _) = counts.iter().enumerate().max_by_key(|(_, c)| **c).unwrap();
    counts[majority] as f64 / actual.len() as f64
}

fn save_predictions(
    test: &[data::Record],
    actual: &[i32],
    predicted: &[i32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(PREDICTIONS_PATH)?;
    wtr.write_record(["student_id", "actual_grade", "predicted_grade"])?;
    for (rec, (a, p)) in test.iter().zip(actual.iter().zip(predicted)) {
        wtr.write_record([
            rec.student_id.to_string(),
            features::ordinal_to_grade(*a).to_string(),
            features::ordinal_to_grade(*p).to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn run_train() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("outputs")?;
    let records = data::load(DATASET_PATH)?;
    println!("Loaded {} student records", records.len());

    let split = split_train_test(&records, TEST_FRACTION, SEED);
    println!("Split (seed {}): {} train / {} test", SEED, split.train.len(), split.test.len());

    // Fit preprocessing on the train split only -> no test information leaks.
    let pre = Preprocessor::fit(&split.train);
    println!("Feature engineering -> {} features", pre.feature_names.len());
    println!("  z-scored numerics: {:?}", features::NUMERIC_FEATURES);
    println!("  one-hot encoded: gender, parental_education, internet_access, extracurricular_activities, part_time_job");
    println!("  dropped: final_exam_score (target leakage - it directly determines the grade)");

    let (x_train, y_train) = pre.transform(&split.train);
    let (x_test, y_test) = pre.transform(&split.test);

    let train_ds = Dataset::new(x_train.clone(), y_train.clone());
    let model = MultiLogisticRegression::new()
        .max_iterations(500)
        .gradient_tolerance(1e-4)
        .fit(&train_ds)?;
    println!("Trained multinomial logistic regression (L-BFGS, L2 alpha=1.0)");

    let predicted = model.predict(&x_test);

    let actual: Vec<i32> = y_test.to_vec();
    let predicted_vec: Vec<i32> = predicted.to_vec();
    let cm = metrics::confusion_matrix(&actual, &predicted_vec, N_CLASSES);
    let per_class = metrics::per_class_metrics(&cm, &[0, 1, 2, 3, 4]);
    print_metrics(&actual, &predicted_vec, &cm, &per_class);

    save_predictions(&split.test, &actual, &predicted_vec)?;
    println!("Saved predictions -> {PREDICTIONS_PATH}");

    let bundle = ModelBundle {
        model: model.clone(),
        preprocess: pre.clone(),
    };
    std::fs::write(MODEL_PATH, bincode::serialize(&bundle)?)?;
    println!("Saved model bundle -> {MODEL_PATH}");

    plots::render_all(&records, &pre, &cm, &model)?;
    println!("Saved 8 labelled plots -> outputs/plots/");

    Ok(())
}

fn run_predict() -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("Model not found at {MODEL_PATH}. Run `cargo run -- train` first.");
        std::process::exit(1);
    }
    println!("Paste a student row as a single CSV line (12 fields, same column order as data/dataset.csv;");
    println!("`final_grade` may be any value, it is ignored). Example:");
    println!("  999,Female,4.6,85.3,8.1,High School,No,No,Yes,52.2,82.2,B");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let record = data::from_csv_line(&line)?;

    let bytes = std::fs::read(MODEL_PATH)?;
    let bundle: ModelBundle = bincode::deserialize(&bytes)?;

    let row = bundle.preprocess.transform_record(&record);
    let features = Array2::from_shape_vec((1, row.len()), row)?;
    let pred = bundle.model.predict(&features);
    println!();
    println!(
        "Student #{} -> predicted final grade: {}",
        record.student_id,
        features::ordinal_to_grade(pred[0])
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => run_train(),
        Some("predict") => run_predict(),
        _ => {
            println!("Student Performance Prediction (Rust + linfa)");
            println!();
            println!("Usage:");
            println!("  cargo run -- train       Train the model, evaluate it and write all artifacts");
            println!("  cargo run -- predict     Load outputs/model.bin and predict a grade from a CSV line");
            println!("  cargo test               Run unit tests");
            Ok(())
        }
    }
}
