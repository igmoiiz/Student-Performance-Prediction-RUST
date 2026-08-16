use crate::data::Record;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Ordered target classes.
pub const GRADES: [&str; 5] = ["A", "B", "C", "D", "F"];

/// Numeric features, in the order they appear in the design matrix.
pub const NUMERIC_FEATURES: [&str; 4] = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
];

/// `parental_education` levels encoded as one-hot columns ("None" is the
/// baseline and is dropped to avoid the dummy-variable trap).
pub const PARENTAL_EDUCATION_LEVELS: [&str; 4] = ["High School", "Bachelors", "Masters", "PhD"];

pub fn grade_to_ordinal(grade: &str) -> i32 {
    GRADES.iter().position(|g| *g == grade).unwrap_or(0) as i32
}

pub fn ordinal_to_grade(ordinal: i32) -> &'static str {
    GRADES.get(ordinal as usize).copied().unwrap_or("?")
}

/// The ordered list of column names of the design matrix (for feature
/// importance plots and documentation).
pub fn feature_names() -> Vec<String> {
    let mut names: Vec<String> = NUMERIC_FEATURES.iter().map(|n| n.to_string()).collect();
    names.push("gender_Male".to_string());
    for level in PARENTAL_EDUCATION_LEVELS.iter() {
        names.push(format!("parental_education_{}", level.replace(' ', "_")));
    }
    names.push("internet_access_Yes".to_string());
    names.push("extracurricular_activities_Yes".to_string());
    names.push("part_time_job_Yes".to_string());
    names
}

/// Encodes raw student records into a numeric design matrix.
///
/// * Numeric features are z-score normalized (mean 0, std 1) using statistics
///   computed on the training split only, so no test information leaks.
/// * Categorical features are one-hot encoded with the baseline category
///   dropped.
///
/// The resulting matrix has 12 columns:
///   `study_time_hours, attendance_percent, sleep_hours, previous_grade,
///    gender_Male, parental_education_* (4), internet_access_Yes,
///    extracurricular_activities_Yes, part_time_job_Yes`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preprocessor {
    pub feature_names: Vec<String>,
    pub numeric_mean: [f64; 4],
    pub numeric_std: [f64; 4],
}

impl Preprocessor {
    /// Fit the scaler statistics on a slice of records (train split only).
    pub fn fit(records: &[Record]) -> Self {
        let n = records.len() as f64;
        let mut sums = [0.0; 4];
        let mut sumsq = [0.0; 4];
        for r in records {
            let vals = [r.study_time_hours, r.attendance_percent, r.sleep_hours, r.previous_grade];
            for (i, v) in vals.iter().enumerate() {
                sums[i] += v;
                sumsq[i] += v * v;
            }
        }
        let mut mean = [0.0; 4];
        let mut std = [0.0; 4];
        for i in 0..4 {
            mean[i] = sums[i] / n;
            let variance = (sumsq[i] / n) - mean[i] * mean[i];
            std[i] = variance.max(0.0).sqrt();
            if std[i] < 1e-12 {
                std[i] = 1.0; // guard against constant features
            }
        }
        Self {
            feature_names: feature_names(),
            numeric_mean: mean,
            numeric_std: std,
        }
    }

    /// Encode a single record into one row of the design matrix.
    pub fn transform_record(&self, r: &Record) -> Vec<f64> {
        let mut row = Vec::with_capacity(self.feature_names.len());
        let vals = [r.study_time_hours, r.attendance_percent, r.sleep_hours, r.previous_grade];
        for (i, v) in vals.iter().enumerate() {
            row.push((v - self.numeric_mean[i]) / self.numeric_std[i]);
        }
        row.push(if r.gender == "Male" { 1.0 } else { 0.0 });
        for level in PARENTAL_EDUCATION_LEVELS.iter() {
            row.push(if r.parental_education == *level { 1.0 } else { 0.0 });
        }
        row.push(if r.internet_access == "Yes" { 1.0 } else { 0.0 });
        row.push(if r.extracurricular_activities == "Yes" { 1.0 } else { 0.0 });
        row.push(if r.part_time_job == "Yes" { 1.0 } else { 0.0 });
        row
    }

    /// Encode a slice of records into a feature matrix and ordinal targets.
    pub fn transform(&self, records: &[Record]) -> (Array2<f64>, Array1<i32>) {
        let n = records.len();
        let n_feat = self.feature_names.len();
        let mut features = Array2::<f64>::zeros((n, n_feat));
        let mut targets = Array1::<i32>::zeros(n);
        for (i, r) in records.iter().enumerate() {
            let row = self.transform_record(r);
            for (j, v) in row.iter().enumerate() {
                features[[i, j]] = *v;
            }
            targets[i] = grade_to_ordinal(&r.final_grade);
        }
        (features, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> Record {
        Record {
            student_id: 1,
            gender: "Male".to_string(),
            study_time_hours: 4.0,
            attendance_percent: 98.0,
            sleep_hours: 6.5,
            parental_education: "Bachelors".to_string(),
            internet_access: "Yes".to_string(),
            extracurricular_activities: "Yes".to_string(),
            part_time_job: "No".to_string(),
            previous_grade: 76.9,
            final_exam_score: 100.0,
            final_grade: "A".to_string(),
        }
    }

    #[test]
    fn grade_mapping_round_trips() {
        for (i, g) in GRADES.iter().enumerate() {
            assert_eq!(grade_to_ordinal(g), i as i32);
            assert_eq!(ordinal_to_grade(i as i32), *g);
        }
    }

    #[test]
    fn feature_matrix_has_12_columns() {
        let pre = Preprocessor::fit(&[sample_record()]);
        assert_eq!(pre.feature_names.len(), 12);
        let (features, targets) = pre.transform(&[sample_record()]);
        assert_eq!(features.shape(), &[1, 12]);
        assert_eq!(targets[0], 0); // "A"
    }

    #[test]
    fn one_hot_encoding_is_correct() {
        let pre = Preprocessor::fit(&[sample_record()]);
        let row = pre.transform_record(&sample_record());
        // 4 numerics, then gender_Male=1, parental (4), then 3 flags
        assert_eq!(row[4], 1.0); // gender Male
        assert_eq!(row[5], 0.0); // High School
        assert_eq!(row[6], 1.0); // Bachelors
        assert_eq!(row[9], 1.0); // internet Yes
        assert_eq!(row[10], 1.0); // extracurricular Yes
        assert_eq!(row[11], 0.0); // part-time No
    }

    #[test]
    fn z_score_normalizes_to_mean_zero_std_one() {
        let records: Vec<Record> = (0..100)
            .map(|i| {
                let mut r = sample_record();
                r.study_time_hours = i as f64 / 10.0; // varying values
                r
            })
            .collect();
        let pre = Preprocessor::fit(&records);
        let (features, _) = pre.transform(&records);
        let col = features.column(0);
        let mean = col.mean().unwrap();
        let variance =
            col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
        let std = variance.sqrt();
        assert!((mean).abs() < 1e-9, "mean={}", mean);
        assert!((std - 1.0).abs() < 1e-9, "std={}", std);
    }

    #[test]
    fn constant_feature_does_not_divide_by_zero() {
        let mut r = sample_record();
        r.sleep_hours = 5.0;
        let pre = Preprocessor::fit(&[r.clone(), r.clone()]);
        let row = pre.transform_record(&r);
        assert!(row.iter().all(|v| v.is_finite()));
    }
}
