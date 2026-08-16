mod data;
mod features;

use features::Preprocessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let records = data::load("data/dataset.csv")?;
    let pre = Preprocessor::fit(&records);
    let (matrix, _targets) = pre.transform(&records);
    println!(
        "Feature engineering -> {} x {} feature matrix",
        matrix.nrows(),
        matrix.ncols()
    );
    println!("Features: {:?}", pre.feature_names);
    println!(
        "Scaled mean / std of first numeric column: {:.3} / {:.3}",
        matrix.column(0).mean().unwrap(),
        matrix.column(0).std(0.0)
    );
    Ok(())
}
