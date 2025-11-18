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

/// CPU mining fallback (small chunks for responsiveness)
async fn mine_cpu_fallback(
    stratum_client: &StratumClient,
    job: &stratum::protocol::StratumJob,
    extranonce1: &str,
    extranonce2: &[u8],
    stats: &mut MiningStats,
) -> Option<(u32, [u8; 32])> {
    let chunk_size = 50_000u32;
    let mut start_nonce = 0u32;
    
    loop {
        // Check for new job
        if stratum_client.has_pending_job().await {
            break;
        }
        
        let end_nonce = start_nonce.saturating_add(chunk_size);
        
        match mine_job_cpu(job, extranonce1, extranonce2, start_nonce, end_nonce, stats) {
            Ok(Some((nonce, hash))) => return Some((nonce, hash)),
            Ok(None) => {
                start_nonce = end_nonce;
                if start_nonce < chunk_size {
                    break; // Wrapped around
                }
            }
            Err(_) => break,
        }
    }
    
    None
}

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
    
    // Initialize CUDA miner if available
    #[cfg(feature = "cuda")]
    let cuda_miner = {
        tracing::info!("Initializing CUDA miner...");
        match crate::cuda::CudaMiner::new() {
            Ok(miner) => {
                let device_name = miner.device_name().unwrap_or_else(|_| "Unknown".to_string());
                let (cc_major, cc_minor) = miner.compute_capability().unwrap_or((0, 0));
                println!("\n{}", "=== GPU Mining Initialized ===".green().bold());
                println!("   {} {}", "Device:".green(), device_name.bright_white());
                println!("   {} {}.{}", "Compute Capability:".green(), cc_major, cc_minor);
                println!("   {} Using CUDA kernel", "Mode:".green());
                println!();
                Some(miner)
            }
            Err(e) => {
                eprintln!("\n{}", "⚠️  CUDA initialization failed!".yellow().bold());
                eprintln!("   {}", format!("Error: {}", e).yellow());
                eprintln!("   {}", "Falling back to CPU mining...".yellow());
                println!();
                None
            }
        }
    };
    
    #[cfg(not(feature = "cuda"))]
    let cuda_miner: Option<()> = None;
    
    // Mining statistics
    let mut stats = MiningStats::new();
    let mut extranonce2_counter: u32 = 0;
    
    // Ctrl+C handler with atomic flag
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        shutdown_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    
    // Main mining loop - wait for jobs
    let mut job_count = 0;
    loop {
        // Check for shutdown
        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            println!("\n{}", "Shutting down miner...".yellow());
            
            // Final statistics
            println!("\n{}", "=== Final Statistics ===".cyan().bold());
            println!("   {} {}", "Total Hashes:".green(), stats.hashes);
            println!("   {} {}", "Shares Found:".green(), stats.shares_found);
            println!("   {} {}", "Shares Accepted:".green(), stats.shares_accepted);
            println!("   {} {}", "Shares Rejected:".green(), stats.shares_rejected);
            
            break;
        }
        
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
                
                let start_time = std::time::Instant::now();
                
                // Choose mining method based on CUDA availability
                #[cfg(feature = "cuda")]
                let mining_result = if let Some(miner) = cuda_miner.clone() {
                    // GPU Mining with NON-BLOCKING polling (low CPU usage like wildrig-multi)
                    let chunk_size = 10_000_000u32; // 10M nonces per call (~500ms GPU time)
                    let mut current_nonce = 0u32;
                    let mut found_share = None;
                    let mut iterations = 0u32;
                    
                    'gpu_mining: loop {
                        // Check for shutdown signal
                        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                            break 'gpu_mining;
                        }
                        
                        // Check for new job every 10 iterations
                        if iterations % 10 == 0 && stratum_client.has_pending_job().await {
                            println!("   {} Switching to new job", "🔄".yellow());
                            break 'gpu_mining;
                        }
                        
                        // Clone data for blocking thread
                        let job_clone = job.clone();
                        let extranonce1_clone = extranonce1.clone();
                        let extranonce2_clone = extranonce2.clone();
                        let miner_clone = miner.clone();
                        let nonce_start = current_nonce; // Capture current value
                        
                        // Run GPU mining in blocking thread pool (doesn't block tokio runtime)
                        let result = tokio::task::spawn_blocking(move || {
                            let mut local_stats = MiningStats::new();
                            crate::mining::mine_job_gpu(
                                &miner_clone,
                                &job_clone,
                                &extranonce1_clone,
                                &extranonce2_clone,
                                nonce_start,
                                chunk_size,
                                &mut local_stats
                            )
                        }).await;
                        
                        match result {
                            Ok(Ok(Some((nonce, hash)))) => {
                                stats.hashes += chunk_size as u64;
                                stats.shares_found += 1;
                                found_share = Some((nonce, hash));
                                break 'gpu_mining;
                            }
                            Ok(Ok(None)) => {
                                // Continue to next chunk
                                iterations += 1;
                                stats.hashes += chunk_size as u64;
                                
                                // Print hashrate every 100 iterations
                                if iterations % 100 == 0 {
                                    let elapsed = start_time.elapsed();
                                    if elapsed.as_secs() > 0 {
                                        let hashrate = stats.hashes as f64 / elapsed.as_secs_f64();
                                        tracing::info!(
                                            "GPU: {} hashes | {:.2} MH/s",
                                            stats.hashes,
                                            hashrate / 1_000_000.0
                                        );
                                    }
                                }
                                
                                current_nonce = current_nonce.wrapping_add(chunk_size);
                                if current_nonce < chunk_size {
                                    // Wrapped around
                                    break 'gpu_mining;
                                }
                                // No artificial sleep - cuStreamQuery polling already includes 1ms sleep
                            }
                            Ok(Err(e)) => {
                                eprintln!("   {} GPU mining error: {}", "❌".red(), e);
                                tracing::error!("GPU error details: {:?}", e);
                                break 'gpu_mining;
                            }
                            Err(e) => {
                                eprintln!("   {} Task join error: {}", "❌".red(), e);
                                break 'gpu_mining;
                            }
                        }
                    }
                    
                    found_share
                } else {
                    // Fallback to CPU if CUDA init failed
                    mine_cpu_fallback(&stratum_client, &job, &extranonce1, &extranonce2, &mut stats).await
                };
                
                #[cfg(not(feature = "cuda"))]
                let mining_result = mine_cpu_fallback(&stratum_client, &job, &extranonce1, &extranonce2, &mut stats).await;
                
                // Process mining result
                if let Some((nonce, hash)) = mining_result {
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
                } else {
                    // No share found
                    let elapsed = start_time.elapsed();
                    let hashrate = if elapsed.as_secs() > 0 {
                        stats.hashes as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    
                    #[cfg(feature = "cuda")]
                    if cuda_miner.is_some() {
                        tracing::debug!("No share found | {:.2} MH/s", hashrate / 1_000_000.0);
                    } else {
                        tracing::debug!("No share found | {:.2} kH/s", hashrate / 1_000.0);
                    }
                    
                    #[cfg(not(feature = "cuda"))]
                    tracing::debug!("No share found | {:.2} kH/s", hashrate / 1_000.0);
                }
                
                // Display statistics every 5 jobs
                if job_count % 5 == 0 {
                    println!("\n{}", "=== Statistics ===".cyan());
                    println!("   {} {}", "Total Hashes:".green(), stats.hashes);
                    println!("   {} {}", "Shares Found:".green(), stats.shares_found);
                }
            }
        }
    }

    Ok(())
}
