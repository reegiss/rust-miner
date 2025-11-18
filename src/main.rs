use anyhow::Result;
use clap::Parser;
use colored::*;

mod algorithms;
mod backend;
mod cli;
mod cuda;
mod gpu;
mod mining;
mod stratum;

use backend::MiningBackend;
use cli::{Args, display_banner};
use gpu::{detect_gpus, select_gpus};
use mining::MiningStats;
use stratum::{StratumClient, StratumConfig};
use std::collections::VecDeque;
use std::process::Command;
use std::time::Duration;

// CPU fallback removed by design. This miner requires a compatible GPU.

/// Create mining backend based on algorithm
fn create_backend(algo: &str) -> Result<std::sync::Arc<dyn MiningBackend>> {
    match algo.to_lowercase().as_str() {
        "qhash" => {
            let backend = cuda::QHashCudaBackend::new()?;
            Ok(std::sync::Arc::new(backend))
        }
        _ => {
            anyhow::bail!(
                "Unknown or unsupported algorithm: '{}'. Supported: qhash",
                algo
            );
        }
    }
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
    
    println!("{:<15} {}", "Backend:".green(), "CUDA".bright_white());
    
    println!();

    // Detect GPUs
    tracing::info!("Detecting available GPUs...");
    
    let all_devices = match detect_gpus() {
        Ok(devices) => devices,
        Err(e) => {
            eprintln!("\n{}", "❌ GPU Detection Failed!".red().bold());
            eprintln!("{}", format!("   Error: {}", e).red());
            eprintln!("\n{}", "This application requires a GPU with CUDA support.".yellow());
            eprintln!("{}", "Please ensure:".yellow());
            eprintln!("{}", "  • GPU drivers are installed".yellow());
            eprintln!("{}", "  • CUDA Toolkit installed (for NVIDIA)".yellow());
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

    // Initialize mining backend (algorithm-specific) - BEFORE connecting to pool
    let backend = {
        tracing::info!("Initializing mining backend for algorithm: {}", args.algo);
        match create_backend(&args.algo) {
            Ok(backend) => {
                let device_name = backend.device_name().unwrap_or_else(|_| "Unknown".to_string());
                let (cc_major, cc_minor) = backend.compute_capability().unwrap_or((0, 0));
                println!("\n{}", "=== GPU Mining Initialized ===".green().bold());
                println!("   {}: {}", "Algorithm".cyan(), args.algo.yellow());
                println!("   {}: {}", "Device".cyan(), device_name.green());
                println!("   {}: {}.{}", "Compute Capability".cyan(), cc_major, cc_minor);
                println!("   {}: Using CUDA kernel", "Mode".cyan());
                backend
            }
            Err(e) => {
                eprintln!("\n{}", "❌ Failed to initialize mining backend!".red().bold());
                eprintln!("{}", format!("   Error: {}", e).red());
                eprintln!("\n{}", format!("Algorithm '{}' is not supported or unavailable.", args.algo).yellow());
                eprintln!("{}", "Supported algorithms:".yellow());
                eprintln!("{}", "  • qhash (QBit/QubitCoin PoW)".green());
                std::process::exit(1);
            }
        }
    };

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
    // Hashrate moving-average tracker (samples = recent per-kernel hash rates)
    let mut hr_window: VecDeque<f64> = VecDeque::with_capacity(8);
    const HR_WINDOW_SIZE: usize = 8;
    // Full history of samples for 10s/60s/15m windows
    let mut hr_history: VecDeque<(std::time::Instant, u64)> = VecDeque::new();
    // Long-term windows for 1h/6h/24h views (sample per chunk)
    let mut long_samples: VecDeque<(std::time::Instant, u64)> = VecDeque::new();
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
                println!("\n{} {}", 
                    "📋 Job".cyan().bold(),
                    format!("#{} (ID: {})", job_count, job.job_id).bright_white()
                );

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

                // Print WildRig-style stats at job start (best-effort)
                print_wildrig_stats(&backend, &hr_history, &stats)?;
                
                let start_time = std::time::Instant::now();
                
                // GPU mining via backend (CUDA-only implementation under the hood)
                let mining_result = {
                    let backend_clone = backend.clone();
                    // GPU Mining with adaptive batch sizing & polling
                    // Start with a larger batch to reduce host<->device churn
                    let mut chunk_size = 50_000_000u32; // initial 50M nonces (~>500ms target)
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
                        
                        let nonce_start = current_nonce;
                        
                        // Run GPU mining in blocking thread (off main async reactor)
                        let backend_for_task = backend_clone.clone();
                        let job_clone = job.clone();
                        let extranonce1_clone = extranonce1.clone();
                        let extranonce2_clone = extranonce2.clone();
                        let result = tokio::task::spawn_blocking(move || {
                            let mining_result = backend_for_task.mine_job(
                                &job_clone,
                                &extranonce1_clone,
                                &extranonce2_clone,
                                nonce_start,
                                chunk_size,
                            );
                            // Return result and num_nonces for stats
                            (mining_result, chunk_size as u64)
                        }).await;

                        match result {
                            Ok((Ok(Some((nonce, hash))), num_hashes)) => {
                                // Merge stats
                                stats.hashes += num_hashes;
                                stats.shares_found += 1;
                                found_share = Some((nonce, hash));
                                break 'gpu_mining;
                            }
                            Ok((Ok(None), num_hashes)) => {
                                // Merge stats
                                stats.hashes += num_hashes;
                                // Continue to next chunk
                                iterations += 1;
                                
                                // Adaptive chunk sizing based on last kernel duration
                                if let Some(ms) = Some(backend_clone.last_kernel_ms()) {
                                    if ms > 0 {
                                        // Target window ~700-900ms
                                        if ms < 400 && chunk_size < 150_000_000 {
                                            // Increase by 25%
                                            chunk_size = (chunk_size as f64 * 1.25) as u32;
                                        } else if ms > 1200 && chunk_size > 5_000_000 {
                                            // Decrease by 20%
                                            chunk_size = (chunk_size as f64 * 0.80) as u32;
                                        }
                                    }
                                }

                                // Print hashrate & tuning info every 25 iterations
                                if iterations % 25 == 0 {
                                    let elapsed = start_time.elapsed();
                                    if elapsed.as_secs() > 0 {
                                        let hashrate = stats.hashes as f64 / elapsed.as_secs_f64();
                                        // Compute recent moving-average based on last kernel samples
                                        let kernel_ms = backend_clone.last_kernel_ms();
                                        if kernel_ms > 0 {
                                            let sample = (chunk_size as f64) / (kernel_ms as f64 / 1000.0);
                                            hr_window.push_back(sample);
                                            if hr_window.len() > HR_WINDOW_SIZE {
                                                hr_window.pop_front();
                                            }
                                        }

                                        // Record a sample for more detailed windows (sample = num_hashes)
                                        hr_history.push_back((std::time::Instant::now(), num_hashes));
                                        // Discard samples older than 15 minutes to bound memory
                                        while let Some((t, _)) = hr_history.front() {
                                            if t.elapsed() > Duration::from_secs(15 * 60) {
                                                hr_history.pop_front();
                                            } else {
                                                break;
                                            }
                                        }

                                        // Compute average of samples if available, otherwise use cumulative rate
                                        let recent_avg = if !hr_window.is_empty() {
                                            hr_window.iter().copied().sum::<f64>() / hr_window.len() as f64
                                        } else {
                                            hashrate
                                        };

                                        // Simplified log: only MH/s and kernel latency
                                        tracing::info!(
                                            "GPU: {:.2} MH/s | last_kernel={}ms",
                                            hashrate / 1_000_000.0,
                                            kernel_ms
                                        );
                                    }
                                        // Also print a WildRig-like statistics block every time we log
                                        print_wildrig_stats(&backend_clone, &hr_history, &stats)?;
                                }
                                
                                current_nonce = current_nonce.wrapping_add(chunk_size);
                                if current_nonce < chunk_size {
                                    // Wrapped around
                                    break 'gpu_mining;
                                }
                            }
                            Ok((Err(e), _num_hashes)) => {
                                eprintln!("   {} GPU mining error: {}", "❌".red(), e);
                                tracing::error!("GPU error details: {:?}", e);
                                break 'gpu_mining;
                            }
                            Err(join_err) => {
                                eprintln!("   {} GPU task join error: {}", "❌".red(), join_err);
                                break 'gpu_mining;
                            }
                        }
                    }
                    
                    found_share
                };
                
                
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
                    
                    tracing::debug!("No share found | {:.2} MH/s (GPU)", hashrate / 1_000_000.0);
                }
                
                // Display statistics every 5 jobs
                if job_count % 5 == 0 {
                    println!("\n{}", "=== Statistics ===".cyan());
                    println!("   {} {}", "Total Hashes:".green(), stats.hashes);
                    println!("   {} {}", "Shares Found:".green(), stats.shares_found);
                    // Print WildRig-like statistics block (best-effort)
                    print_wildrig_stats(&backend, &hr_history, &stats)?;
                    // Add a long-sample for the last processed chunk to support 1h/6h/24h views
                    let now = std::time::Instant::now();
                    long_samples.push_back((now, stats.hashes));
                    // Purge samples older than 24h
                    while let Some(&(t, _)) = long_samples.front() {
                        if now.duration_since(t).as_secs() > 24 * 3600 {
                            long_samples.pop_front();
                        } else {
                            break;
                        }
                    }

                    // Compute hash-rate window functions
                    let calc_window = |secs: u64| -> f64 {
                        if long_samples.is_empty() { return 0.0; }
                        let cutoff = now - std::time::Duration::from_secs(secs);
                        // Sum bytes newer than cutoff
                        let mut sum: u64 = 0;
                        let mut earliest = now;
                        for &(t, h) in long_samples.iter().rev() { // iterate from newest
                            if t >= cutoff {
                                sum = sum.saturating_add(h);
                                earliest = t;
                            } else {
                                break;
                            }
                        }
                        let dur = now.duration_since(earliest).as_secs_f64();
                        if dur <= 0.0 { return 0.0; }
                        (sum as f64) / dur
                    };

                    // Print 1h/6h/24h averages in MH/s
                    let m1 = calc_window(3600) / 1_000_000.0;
                    let m6 = calc_window(3600 * 6) / 1_000_000.0;
                    let m24 = calc_window(3600 * 24) / 1_000_000.0;
                    println!("   {} {:.2} MH/s | {} {:.2} MH/s | {} {:.2} MH/s", "1h".green(), m1, "6h".green(), m6, "24h".green(), m24);
                }
            }
        }
    }

    Ok(())
}

// Try to get GPU stats from nvidia-smi (best-effort). Returns (temp, fan, power, gfx_clock, mem_clock)
fn get_gpu_stats(device_index: usize) -> Option<(Option<u32>, Option<u32>, Option<f64>, Option<u32>, Option<u32>)> {
    // Query: temperature.gpu,fan.speed,power.draw,clocks.gr,clocks.mem
    let out = Command::new("nvidia-smi")
        .arg("--query-gpu=temperature.gpu,fan.speed,power.draw,clocks.gr,clocks.mem")
        .arg("--format=csv,noheader,nounits")
        .arg("-i")
        .arg(format!("{}", device_index))
        .output();

    let Ok(out) = out else { return None; };
    if !out.status.success() { return None; }

    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
    if parts.len() < 5 { return None; }

    let temp = parts[0].parse::<u32>().ok();
    let fan = parts[1].parse::<u32>().ok();
    let power = parts[2].parse::<f64>().ok();
    let gfx = parts[3].parse::<u32>().ok();
    let mem = parts[4].parse::<u32>().ok();
    Some((temp, fan, power, gfx, mem))
}

// Print stats in WildRig-like format (single GPU supported currently)
fn print_wildrig_stats(backend: &std::sync::Arc<dyn MiningBackend>, hr_history: &VecDeque<(std::time::Instant, u64)>, stats: &MiningStats) -> anyhow::Result<()> {
    use std::fmt::Write as _;

    let device_name = backend.device_name().unwrap_or_else(|_| "Unknown GPU".to_string());
    let idx = 0usize; // single-device support for now

    // Calculate rates
    let now = std::time::Instant::now();
    let rate_for = |secs: u64| -> f64 {
        let window = Duration::from_secs(secs);
        let sum: u64 = hr_history
            .iter()
            .filter(|(t, _)| now.duration_since(*t) <= window)
            .map(|(_, n)| *n)
            .sum();
        sum as f64 / window.as_secs_f64()
    };

    let rate10 = rate_for(10);
    let rate60 = rate_for(60);
    let rate15m = rate_for(15 * 60);

    let gstat = get_gpu_stats(idx).unwrap_or((None, None, None, None, None));

    let mut buf = String::new();
    writeln!(buf, "--------------------------------------[Statistics]--------------------------------------").ok();
    writeln!(buf, " ID Name                         Hashrate Temp  Fan  Power   Eff CClk MClk     A   R   I").ok();
    writeln!(buf, "----------------------------------------------------------------------------------------").ok();
    // Per-device row
    writeln!(buf, " #{} {:26} {:9.2} MH/s  {}C  {}%  {:.1}W   -   {}   {}     -   -   - ",
        idx,
        format!("{}", device_name),
        rate10 / 1_000_000.0,
        gstat.0.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string()),
        gstat.1.map(|f| f.to_string()).unwrap_or_else(|| "-".to_string()),
        gstat.2.unwrap_or(0.0),
        gstat.3.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string()),
        gstat.4.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string())
    ).ok();
    writeln!(buf, "----------------------------------------------------------------------------------------").ok();
    writeln!(buf, " 10s: {:>28.2} MH/s Power: {:>7.1}W            Accepted: {:8} ", rate10 / 1_000_000.0, gstat.2.unwrap_or(0.0), stats.shares_accepted).ok();
    writeln!(buf, " 60s: {:>32.2} MH/s                             Rejected: {:8} ", rate60 / 1_000_000.0, stats.shares_rejected).ok();
    writeln!(buf, " 15m: {:>31.2} MH/s                             Ignored:  {:8} ", rate15m / 1_000_000.0, 0).ok();
    writeln!(buf, "[{}]----------------------------------------------------------[ver. {}]", humantime::format_duration(now.elapsed()), env!("CARGO_PKG_VERSION")).ok();

    println!("{}", buf);

    Ok(())
}
