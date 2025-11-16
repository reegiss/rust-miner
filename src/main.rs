use anyhow::Result;
use clap::Parser;
use colored::*;

mod cli;
mod gpu;

use cli::{Args, display_banner};
use gpu::{detect_gpus, select_gpus};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    // Display banner
    display_banner();

    // Parse command-line arguments
    let args = Args::parse();

    // Validate algorithm
    if args.algo.is_empty() {
        eprintln!("{}", "Error: --algo is required".red().bold());
        std::process::exit(1);
    }

    // Validate mining URL
    if args.url.is_empty() {
        eprintln!("{}", "Error: --url is required".red().bold());
        std::process::exit(1);
    }

    // Validate user (wallet address)
    if args.user.is_empty() {
        eprintln!("{}", "Error: --user (wallet address) is required".red().bold());
        std::process::exit(1);
    }

    // Display configuration
    println!("\n{}", "=== Mining Configuration ===".cyan().bold());
    println!("{:<15} {}", "Algorithm:".green(), args.algo.bright_white());
    println!("{:<15} {}", "Pool URL:".green(), args.url.bright_white());
    println!("{:<15} {}", "Wallet:".green(), args.user.bright_white());
    println!("{:<15} {}", "Password:".green(), args.pass.bright_white());
    
    if let Some(ref gpu) = args.gpu {
        println!("{:<15} {}", "GPU:".green(), gpu.bright_white());
    }
    
    if args.opencl {
        println!("{:<15} {}", "Backend:".green(), "OpenCL".bright_white());
    } else {
        println!("{:<15} {}", "Backend:".green(), "CUDA (default)".bright_white());
    }
    
    println!();

    // Detect GPUs
    tracing::info!("Detecting available GPUs...");
    
    let all_devices = match detect_gpus() {
        Ok(devices) => devices,
        Err(e) => {
            eprintln!("\n{}", "❌ GPU Detection Failed!".red().bold());
            eprintln!("{}", format!("   Error: {}", e).red());
            eprintln!("\n{}", "This application requires a GPU with CUDA or OpenCL support.".yellow());
            eprintln!("{}", "Please ensure:".yellow());
            eprintln!("{}", "  • GPU drivers are installed".yellow());
            eprintln!("{}", "  • CUDA Toolkit installed (for NVIDIA)".yellow());
            eprintln!("{}", "  • OpenCL runtime installed (for AMD/Intel)".yellow());
            eprintln!("\n{}", "See SETUP.md for installation instructions.".cyan());
            std::process::exit(1);
        }
    };
    
    // Display detected GPUs
    println!("\n{}", "=== Detected GPUs ===".cyan().bold());
    for device in &all_devices {
        println!("  {}", format!("{}", device).green());
    }
    
    // Select GPUs based on --gpu argument
    let selected_devices = match select_gpus(args.gpu.clone(), &all_devices) {
        Ok(devices) => devices,
        Err(e) => {
            eprintln!("\n{}", format!("Error selecting GPUs: {}", e).red().bold());
            std::process::exit(1);
        }
    };
    
    // Display selected GPUs for mining
    if selected_devices.len() < all_devices.len() {
        println!("\n{}", "=== Selected GPUs for Mining ===".cyan().bold());
        for device in &selected_devices {
            println!("  {}", format!("{}", device).green());
        }
    }
    
    println!();

    // TODO: Initialize mining backend
    tracing::info!("Initializing mining backend...");
    
    // TODO: Connect to pool
    tracing::info!("Connecting to pool: {}", args.url);
    
    // TODO: Start mining
    tracing::info!("Starting mining with algorithm: {}", args.algo);

    // Placeholder: keep running
    println!("{}", "Mining started! Press Ctrl+C to stop.".yellow().bold());
    
    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    
    println!("\n{}", "Shutting down miner...".yellow());

    Ok(())
}
