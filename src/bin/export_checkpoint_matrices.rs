use cell_attention::{config::RunConfig, training::export_checkpoint_matrices};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && (args.len() != 4 || args[3] != "--inputs-only") {
        return Err(
            "usage: export_checkpoint_matrices TRAIN_CONFIG.json CHECKPOINT_DIR [--inputs-only]"
                .into(),
        );
    }
    let config = RunConfig::from_file(&args[1])?;
    export_checkpoint_matrices(&config, Path::new(&args[2]), args.len() == 4)
}
