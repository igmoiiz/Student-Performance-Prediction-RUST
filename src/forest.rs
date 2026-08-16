use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array1, Array2, Axis};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A random forest built on `linfa-trees` decision trees.
///
/// Bagging is implemented by bootstrap-sampling the rows with replacement, and
/// the second source of randomness comes from training each tree on a random
/// subset of the feature columns. Every tree keeps the subset it was trained on
/// so predictions can be made on full feature rows.
#[derive(Serialize, Deserialize, Clone)]
pub struct RandomForest {
    members: Vec<TreeMember>,
    n_classes: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct TreeMember {
    /// Column indices of the feature subset this tree was trained on.
    features: Vec<usize>,
    tree: DecisionTree<f64, usize>,
}

pub struct RandomForestParams {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub min_weight_leaf: f32,
    pub n_features_subset: usize,
    /// Bootstrap rows with probability inversely proportional to class
    /// frequency, so rare classes appear in more bags (helps imbalance).
    pub class_weighted: bool,
    pub seed: u64,
}

impl RandomForest {
    pub fn train(
        x: &Array2<f64>,
        y: &Array1<usize>,
        params: &RandomForestParams,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = x.nrows();
        let n_total = x.ncols();
        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut members = Vec::with_capacity(params.n_trees);

        let weighted: Option<WeightedIndex<f64>> = if params.class_weighted {
            let mut counts = vec![0usize; 0];
            let n_classes = y.iter().cloned().max().unwrap_or(0) + 1;
            counts.resize(n_classes, 0);
            for &v in y {
                counts[v] += 1;
            }
            let mut w = Vec::with_capacity(n);
            for &v in y {
                w.push(1.0 / counts[v].max(1) as f64);
            }
            WeightedIndex::new(w).ok()
        } else {
            None
        };

        for _ in 0..params.n_trees {
            let indices: Vec<usize> = if let Some(dist) = &weighted {
                (0..n).map(|_| rng.sample(dist)).collect()
            } else {
                (0..n).map(|_| rng.gen_range(0..n)).collect()
            };
            let x_boot = x.select(Axis(0), &indices);
            let y_boot = y.select(Axis(0), &indices);

            let mut feats: Vec<usize> = (0..n_total).collect();
            feats.shuffle(&mut rng);
            let chosen = feats[..params.n_features_subset].to_vec();
            let x_sub = x_boot.select(Axis(1), &chosen);

            let ds = Dataset::new(x_sub, y_boot);
            let tree = DecisionTree::params()
                .split_quality(SplitQuality::Gini)
                .max_depth(params.max_depth)
                .min_weight_leaf(params.min_weight_leaf)
                .fit(&ds)?;
            members.push(TreeMember { features: chosen, tree });
        }

        let n_classes = y.iter().cloned().max().unwrap_or(0) as usize + 1;
        Ok(Self { members, n_classes })
    }

    /// Majority vote over the ensemble.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let n = x.nrows();
        let mut votes = vec![vec![0usize; self.n_classes]; n];
        for m in &self.members {
            let x_sub = x.select(Axis(1), &m.features);
            let preds = m.tree.predict(&x_sub);
            for (i, p) in preds.iter().enumerate() {
                votes[i][*p as usize] += 1;
            }
        }
        let mut out = Array1::zeros(n);
        for (i, row) in votes.iter().enumerate() {
            let best = row.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0;
            out[i] = best;
        }
        out
    }

    /// Class probabilities as the fraction of trees voting for each class.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let n_classes = self.n_classes;
        let mut votes = vec![vec![0usize; n_classes]; n];
        for m in &self.members {
            let x_sub = x.select(Axis(1), &m.features);
            let preds = m.tree.predict(&x_sub);
            for (i, p) in preds.iter().enumerate() {
                votes[i][*p] += 1;
            }
        }
        let total = self.members.len().max(1) as f64;
        Array2::from_shape_fn((n, n_classes), |(i, j)| votes[i][j] as f64 / total)
    }

    /// Average feature importance across trees. Each tree's relative impurity
    /// decrease is mapped back onto the full feature space before averaging.
    pub fn feature_importance(&self, n_features_total: usize) -> Vec<f64> {
        let mut acc = vec![0.0f64; n_features_total];
        for m in &self.members {
            let imp = m.tree.feature_importance();
            for (local, &global) in m.features.iter().enumerate() {
                if let Some(&v) = imp.get(local) {
                    acc[global] += v;
                }
            }
        }
        let scale = self.members.len().max(1) as f64;
        acc.iter().map(|v| v / scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_data() -> (Array2<f64>, Array1<usize>) {
        // Two clearly separated clusters, 4 rows each of class 0 and 1.
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.1, 0.2, 1.0, 0.2, 0.1, 1.0, 0.3, 0.2, 1.0, 0.1, 0.3, 1.0, //
                5.0, 5.1, 1.0, 5.1, 5.0, 1.0, 5.2, 5.1, 1.0, 5.0, 5.2, 1.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);
        (x, y)
    }

    #[test]
    fn forest_trains_and_predicts() {
        let (x, y) = toy_data();
        let params = RandomForestParams {
            n_trees: 10,
            max_depth: Some(4),
            min_weight_leaf: 1.0,
            n_features_subset: 2,
            class_weighted: false,
            seed: 1,
        };
        let forest = RandomForest::train(&x, &y, &params).unwrap();
        let preds = forest.predict(&x);
        assert_eq!(preds.len(), 8);
        let acc = preds
            .iter()
            .zip(y.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / 8.0;
        assert!(acc >= 0.875, "forest should learn the clusters, got {acc}");
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let (x, y) = toy_data();
        let params = RandomForestParams {
            n_trees: 5,
            max_depth: Some(4),
            min_weight_leaf: 1.0,
            n_features_subset: 2,
            class_weighted: true,
            seed: 2,
        };
        let forest = RandomForest::train(&x, &y, &params).unwrap();
        let proba = forest.predict_proba(&x);
        assert_eq!(proba.shape(), &[8, 2]);
        for row in proba.rows() {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-9, "probabilities must sum to 1, got {sum}");
        }
    }
}
