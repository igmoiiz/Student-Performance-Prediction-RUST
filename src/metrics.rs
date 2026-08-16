use ndarray::Array2;

/// Per-class classification metrics for a multiclass problem.
#[derive(Debug, Clone, Copy)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Build a `n_classes x n_classes` confusion matrix from ordinal targets.
/// `cm[[actual, predicted]]` counts how many rows with the true label
/// `actual` were predicted as `predicted`.
pub fn confusion_matrix(actual: &[i32], predicted: &[i32], n_classes: usize) -> Array2<usize> {
    let mut cm = Array2::<usize>::zeros((n_classes, n_classes));
    for (a, p) in actual.iter().zip(predicted) {
        cm[[*a as usize, *p as usize]] += 1;
    }
    cm
}

pub fn accuracy(actual: &[i32], predicted: &[i32]) -> f64 {
    if actual.is_empty() {
        return 0.0;
    }
    let correct = actual.iter().zip(predicted).filter(|(a, p)| a == p).count();
    correct as f64 / actual.len() as f64
}

/// Per-class precision, recall and F1 computed from a confusion matrix.
pub fn per_class_metrics(cm: &Array2<usize>, labels: &[i32]) -> Vec<(i32, ClassMetrics)> {
    let n = cm.nrows();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let tp = cm[[i, i]];
        let fp: usize = (0..n).map(|r| cm[[r, i]]).sum::<usize>() - tp;
        let fn_count: usize = (0..n).map(|c| cm[[i, c]]).sum::<usize>() - tp;
        let support: usize = (0..n).map(|c| cm[[i, c]]).sum();
        let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
        let recall = if tp + fn_count == 0 { 0.0 } else { tp as f64 / (tp + fn_count) as f64 };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        out.push((
            labels[i],
            ClassMetrics { precision, recall, f1, support },
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_matrix_counts_correctly() {
        let actual = [0, 0, 1, 1, 2];
        let predicted = [0, 1, 1, 1, 2];
        let cm = confusion_matrix(&actual, &predicted, 3);
        assert_eq!(cm[[0, 0]], 1);
        assert_eq!(cm[[0, 1]], 1);
        assert_eq!(cm[[1, 1]], 2);
        assert_eq!(cm[[2, 2]], 1);
    }

    #[test]
    fn per_class_metrics_are_correct() {
        // Class 0: tp=1 fp=0 fn=1  -> p=1.0 r=0.5 f1=2/3
        // Class 1: tp=2 fp=1 fn=0  -> p=2/3 r=1.0 f1=0.8
        let actual = [0, 0, 1, 1, 2];
        let predicted = [0, 1, 1, 1, 2];
        let cm = confusion_matrix(&actual, &predicted, 3);
        let metrics = per_class_metrics(&cm, &[0, 1, 2]);
        assert!((metrics[0].1.precision - 1.0).abs() < 1e-9);
        assert!((metrics[0].1.recall - 0.5).abs() < 1e-9);
        assert!((metrics[1].1.precision - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics[1].1.recall - 1.0).abs() < 1e-9);
        assert_eq!(metrics[2].1.support, 1);
    }

    #[test]
    fn accuracy_is_fraction_correct() {
        let actual = [0, 1, 2, 0];
        let predicted = [0, 1, 0, 0];
        assert!((accuracy(&actual, &predicted) - 0.75).abs() < 1e-9);
    }
}
