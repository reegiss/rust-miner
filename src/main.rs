use anyhow::Result;
use clap::Parser;
use colored::*;

mod algorithms;
mod cli;
#[cfg(feature = "cuda")]
mod cuda;
mod gpu;
mod mining;
mod stratum;

use cli::{Args, display_banner};
use gpu::{detect_gpus, select_gpus};
use mining::{mine_job_cpu, MiningStats};
use stratum::{StratumClient, StratumConfig};

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

    // Initialize Stratum client
    tracing::info!("Initializing Stratum connection...");
    
    let stratum_config = StratumConfig::new(
        args.url.clone(),
        args.user.clone(),
        args.pass.clone(),
    );
    
    let stratum_client = StratumClient::new(stratum_config);
    
    // Connect to pool
    match stratum_client.connect_and_login().await {
        Ok(_) => {
            println!("{}", "✓ Connected to pool successfully".green().bold());
        }
        Err(e) => {
            eprintln!("\n{}", "❌ Failed to connect to pool!".red().bold());
            eprintln!("{}", format!("   Error: {}", e).red());
            eprintln!("\n{}", "Please check:".yellow());
            eprintln!("{}", "  • Pool URL is correct and reachable".yellow());
            eprintln!("{}", "  • Wallet address is valid".yellow());
            eprintln!("{}", "  • Network connection is working".yellow());
            std::process::exit(1);
        }
    }
    
    println!("\n{}", "=== Mining Status ===".cyan().bold());
    println!("{}", "Waiting for jobs from pool...".yellow());
    
    // Mining statistics
    let mut stats = MiningStats::new();
    let mut extranonce2_counter: u32 = 0;
    
    // Main mining loop - wait for jobs
    let mut job_count = 0;
    loop {
        tokio::select! {
            Some(job) = stratum_client.get_job() => {
                job_count += 1;
                println!("\n{} {} {}", 
                    "📋 Job".cyan().bold(),
                    format!("#{}", job_count).bright_white(),
                    format!("(ID: {})", job.job_id).dimmed()
                );
                println!("   {} {}", "Previous Hash:".green(), job.prevhash);
                println!("   {} {}", "Network Time:".green(), job.ntime);
                println!("   {} {}", "Difficulty:".green(), job.nbits);
                println!("   {} {}", "Clean Jobs:".green(), job.clean_jobs);
                
                // Get extranonce1 from client
                let extranonce1 = match stratum_client.get_extranonce1().await {
                    Some(en1) => en1,
                    None => {
                        eprintln!("{}", "Error: No extranonce1 available".red());
                        continue;
                    }
                };
                
                // Create extranonce2
                let extranonce2 = stratum_client.create_extranonce2(extranonce2_counter).await;
                extranonce2_counter = extranonce2_counter.wrapping_add(1);
                
                println!("   {} {}", "Extranonce1:".green(), extranonce1);
                println!("   {} {}", "Extranonce2:".green(), hex::encode(&extranonce2));
                println!("\n{}", "⛏️  Mining...".yellow().bold());
                
                // NOTE: Using CPU mining temporarily ONLY for pool connectivity testing
                // TODO: Replace with GPU (CUDA) mining for production (1000x faster)
                let chunk_size = 50_000; // Small chunks for responsiveness to Ctrl+C
                let mut start_nonce = 0u32;
                
                let start_time = std::time::Instant::now();
                'mining: loop {
                    // Check if there's a new job waiting (non-blocking)
                    if stratum_client.has_pending_job().await {
                        println!("   {} Switching to new job", "🔄".yellow());
                        break 'mining;
                    }
                    
                    let end_nonce = start_nonce.saturating_add(chunk_size);
                    
                    match mine_job_cpu(&job, &extranonce1, &extranonce2, start_nonce, end_nonce, &mut stats) {
                        Ok(Some((nonce, hash))) => {
                            let elapsed = start_time.elapsed();
                            println!("\n{}", "🎉 SHARE FOUND!".green().bold());
                            println!("   {} 0x{:08x}", "Nonce:".green(), nonce);
                            println!("   {} {}", "Hash:".green(), hex::encode(hash));
                            println!("   {} {:.2}s", "Time:".green(), elapsed.as_secs_f64());
                            
                            // Submit share to pool
                            println!("   {} Submitting share...", "📤".yellow());
                            
                            let extranonce2_hex = hex::encode(&extranonce2);
                            let nonce_hex = format!("{:08x}", nonce);
                            
                            match stratum_client.submit_share(
                                &job.job_id,
                                &extranonce2_hex,
                                &job.ntime,
                                &nonce_hex
                            ).await {
                                Ok(true) => {
                                    stats.shares_accepted += 1;
                                    println!("   {} {}", "✅ Share accepted!".green().bold(), 
                                        format!("({}/{})", stats.shares_accepted, stats.shares_found).dimmed());
                                }
                                Ok(false) => {
                                    stats.shares_rejected += 1;
                                    println!("   {} {}", "❌ Share rejected".red().bold(),
                                        format!("({} rejected)", stats.shares_rejected).dimmed());
                                }
                                Err(e) => {
                                    stats.shares_rejected += 1;
                                    eprintln!("   {} Submit error: {}", "❌".red(), e);
                                }
                            }
                            
                            // Continue mining this job after submitting
                            start_nonce = end_nonce;
                            if start_nonce == 0 {
                                // Nonce space exhausted, move to next extranonce2
                                break 'mining;
                            }
                        }
                        Ok(None) => {
                            // No share found in this chunk, continue to next chunk
                            start_nonce = end_nonce;
                            if start_nonce == 0 {
                                // Nonce space exhausted (wrapped around)
                                let elapsed = start_time.elapsed();
                                let hashrate = if elapsed.as_secs() > 0 {
                                    stats.hashes as f64 / elapsed.as_secs_f64()
                                } else {
                                    0.0
                                };
                                println!("   {} Exhausted nonce space (4.3 billion hashes)", "ℹ️".dimmed());
                                println!("   {} {:.2} H/s", "Hashrate:".dimmed(), hashrate);
                                break 'mining;
                            }
                            
                            // Log progress every chunk
                            if start_nonce % (chunk_size * 10) == 0 {
                                let elapsed = start_time.elapsed();
                                let hashrate = if elapsed.as_secs() > 0 {
                                    stats.hashes as f64 / elapsed.as_secs_f64()
                                } else {
                                    0.0
                                };
                                println!("   {} Mined {} hashes | {:.2} H/s", 
                                    "⛏️".dimmed(),
                                    start_nonce.to_string().bright_white(),
                                    hashrate
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("\n{} {}", "❌ Mining error:".red().bold(), e);
                            break 'mining;
                        }
                    }
                }
                
                // Display statistics
                if job_count % 5 == 0 {
                    println!("\n{}", "=== Statistics ===".cyan());
                    println!("   {} {}", "Total Hashes:".green(), stats.hashes);
                    println!("   {} {}", "Shares Found:".green(), stats.shares_found);
                }
            }
            
            _ = tokio::signal::ctrl_c() => {
                println!("\n{}", "Shutting down miner...".yellow());
                
                // Final statistics
                println!("\n{}", "=== Final Statistics ===".cyan().bold());
                println!("   {} {}", "Total Hashes:".green(), stats.hashes);
                println!("   {} {}", "Shares Found:".green(), stats.shares_found);
                println!("   {} {}", "Shares Accepted:".green(), stats.shares_accepted);
                println!("   {} {}", "Shares Rejected:".green(), stats.shares_rejected);
                
                break;
            }
        }
    }

    Ok(())
}
