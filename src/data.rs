use serde::Deserialize;
use std::path::Path;

/// One row of the student dataset (schema of `data/dataset.csv`).
#[derive(Debug, Clone, Deserialize)]
pub struct Record {
    pub student_id: i64,
    pub gender: String,
    pub study_time_hours: f64,
    pub attendance_percent: f64,
    pub sleep_hours: f64,
    pub parental_education: String,
    pub internet_access: String,
    pub extracurricular_activities: String,
    pub part_time_job: String,
    pub previous_grade: f64,
    /// Dropped during feature engineering: it is the variable that directly
    /// determines `final_grade`, so including it would leak the target.
    #[allow(dead_code)]
    pub final_exam_score: f64,
    pub final_grade: String,
}

/// Load all student records from a CSV file.
pub fn load(path: impl AsRef<Path>) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut records = Vec::new();
    for row in reader.deserialize() {
        records.push(row?);
    }
    Ok(records)
}

/// Parse a single student as a comma-separated line with the same column order
/// as `data/dataset.csv` (12 fields). Used by the `predict` subcommand.
pub fn from_csv_line(line: &str) -> Result<Record, Box<dyn std::error::Error>> {
    let f: Vec<&str> = line.trim().split(',').collect();
    if f.len() != 12 {
        return Err(format!(
            "expected 12 comma-separated fields, got {}: {}",
            f.len(),
            line
        )
        .into());
    }
    let num = |s: &str| -> Result<f64, Box<dyn std::error::Error>> {
        Ok(s.trim().parse::<f64>()?)
    };
    Ok(Record {
        student_id: f[0].trim().parse::<i64>()?,
        gender: f[1].trim().to_string(),
        study_time_hours: num(f[2])?,
        attendance_percent: num(f[3])?,
        sleep_hours: num(f[4])?,
        parental_education: f[5].trim().to_string(),
        internet_access: f[6].trim().to_string(),
        extracurricular_activities: f[7].trim().to_string(),
        part_time_job: f[8].trim().to_string(),
        previous_grade: num(f[9])?,
        final_exam_score: num(f[10])?,
        final_grade: f[11].trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_csv_line() {
        let r = from_csv_line("42,Male,4.0,98.0,6.5,Bachelors,Yes,Yes,No,76.9,100.0,A").unwrap();
        assert_eq!(r.student_id, 42);
        assert_eq!(r.gender, "Male");
        assert_eq!(r.final_grade, "A");
    }

    #[test]
    fn rejects_wrong_field_count() {
        assert!(from_csv_line("1,Male,4.0").is_err());
    }
}
