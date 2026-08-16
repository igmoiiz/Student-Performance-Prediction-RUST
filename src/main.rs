mod data;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let records = data::load("data/dataset.csv")?;
    println!("Loaded {} student records", records.len());
    println!("First record: {:?}", records[0]);
    Ok(())
}
