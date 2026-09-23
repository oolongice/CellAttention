use cell_attention::{config::RunConfig, training::train_from_config};

fn main() {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "train_config.json".to_string());
    let config = RunConfig::from_file(&config_path).unwrap_or_else(|error| {
        eprintln!("configuration error: {error}");
        std::process::exit(2);
    });
    if let Err(error) = train_from_config(&config) {
        eprintln!("training failed: {error}");
        std::process::exit(1);
    }
    println!("checkpoint written to {}", config.output_dir);
}
