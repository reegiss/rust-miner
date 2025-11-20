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
use gpu::{detect_gpus, select_gpus, GpuDevice};
use mining::MiningStats;
use stratum::{StratumClient, StratumConfig, StratumJob};
use std::collections::VecDeque;
use std::process::Command;
use std::time::Duration;
use tokio::sync::mpsc;

// CPU fallback removed by design. This miner requires a compatible GPU.

/// Statistics for a single GPU
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub gpu_index: usize,
    pub hashes: u64,
    pub shares_found: u64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub current_hashrate: f64,
    pub temperature: Option<i32>,
    pub power_usage: Option<f64>,
    pub utilization: Option<i32>,
}

impl GpuStats {
    pub fn new(gpu_index: usize) -> Self {
        Self {
            gpu_index,
            hashes: 0,
            shares_found: 0,
            shares_accepted: 0,
            shares_rejected: 0,
            current_hashrate: 0.0,
            temperature: None,
            power_usage: None,
            utilization: None,
        }
    }
}

/// Global mining statistics
#[derive(Debug, Clone)]
pub struct GlobalStats {
    pub hashes: u64,
    pub shares_found: u64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
}

impl GlobalStats {
    pub fn new() -> Self {
        Self {
            hashes: 0,
            shares_found: 0,
            shares_accepted: 0,
            shares_rejected: 0,
        }
    }
}

/// Result from GPU mining task
#[derive(Debug)]
enum GpuMiningResult {
    StatsUpdate { gpu_index: usize, hashes: u64, kernel_ms: u64 },
    ShareFound { gpu_index: usize, nonce: u32, hash: [u8; 32], extranonce2: Vec<u8> },
    Error { gpu_index: usize, error: String },
}

/// GPU Miner instance with dedicated backend and stats
#[derive(Clone)]
struct GpuMiner {
    pub device: GpuDevice,
    pub backend: std::sync::Arc<tokio::sync::Mutex<Box<dyn MiningBackend>>>,
    pub device_info: backend::GpuInfo,
    pub stats: GpuStats,
}

impl GpuMiner {
    fn new(device: GpuDevice, backend: std::sync::Arc<tokio::sync::Mutex<Box<dyn MiningBackend>>>, device_info: backend::GpuInfo) -> Self {
        let stats = GpuStats::new(device.id);
        Self {
            device,
            backend,
            device_info,
            stats,
        }
    }
}

/// GPU mining task - runs in dedicated tokio::spawn for each GPU
async fn gpu_mining_task(
    gpu_miner: GpuMiner,
    stats_tx: mpsc::Sender<GpuMiningResult>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stratum_client: std::sync::Arc<tokio::sync::Mutex<StratumClient>>,
) {
    let gpu_index = gpu_miner.device.id;
    let chunk_size = 50_000_000u32; // 50M nonces per batch
    let mut current_nonce = (gpu_index as u32) * 1_000_000_000; // Unique nonce space per GPU
    let mut extranonce2_counter: u32 = gpu_index as u32; // Unique extranonce2 per GPU
    
    tracing::info!("GPU #{} mining task started with nonce offset: {}", gpu_index, current_nonce);
    
    loop {
        // Check for shutdown
        if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            tracing::info!("GPU #{} shutting down", gpu_index);
            break;
        }
        
        // Wait for job
        let job = loop {
            let stratum_lock = stratum_client.lock().await;
            match stratum_lock.get_job().await {
                Some(job) => break job,
                None => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        };
        
        // Get extranonce1 and create unique extranonce2 for this GPU
        let (extranonce1, extranonce2) = {
            let stratum_lock = stratum_client.lock().await;
            let en1 = stratum_lock.get_extranonce1().await.unwrap_or_else(|| "".to_string());
            let en2 = stratum_lock.create_extranonce2(extranonce2_counter).await;
            (en1, en2)
        };
        extranonce2_counter = extranonce2_counter.wrapping_add(1); // Simple increment for now
        
        // Mine on GPU
        let nonce_start = current_nonce;
        let backend_clone = gpu_miner.backend.clone();
        let extranonce2_clone = extranonce2.clone();
        
        let mining_result = tokio::task::spawn_blocking(move || {
            let backend_ref = backend_clone.blocking_lock();
            backend_ref.mine_job(
                &job,
                &extranonce1,
                &extranonce2_clone,
                nonce_start,
                chunk_size,
            )
        }).await;
        
        match mining_result {
            Ok(Ok(mining_result)) => {
                // Send stats update
                let kernel_ms = gpu_miner.backend.lock().await.last_kernel_ms();
                let _ = stats_tx.send(GpuMiningResult::StatsUpdate {
                    gpu_index,
                    hashes: mining_result.hashes_computed,
                    kernel_ms,
                }).await;
                
                if mining_result.found_share {
                    if let (Some(nonce), Some(hash)) = (mining_result.nonce, mining_result.hash) {
                        let _ = stats_tx.send(GpuMiningResult::ShareFound {
                            gpu_index,
                            nonce,
                            hash: *hash,
                            extranonce2,
                        }).await;
                    }
                }
            }
            Ok(Err(e)) => {
                let _ = stats_tx.send(GpuMiningResult::Error {
                    gpu_index,
                    error: format!("Mining error: {}", e),
                }).await;
            }
            Err(e) => {
                let _ = stats_tx.send(GpuMiningResult::Error {
                    gpu_index,
                    error: format!("Task join error: {}", e),
                }).await;
            }
        }
        
        current_nonce = current_nonce.wrapping_add(chunk_size);
    }
}

/// Create mining backend for specific device (synchronous wrapper)
fn create_backend_for_device_sync(algo: &str, device_index: usize) -> Result<(std::sync::Arc<tokio::sync::Mutex<Box<dyn MiningBackend>>>, backend::GpuInfo)> {
    match algo {
        "qhash" => {
            let backend = cuda::QHashCudaBackend::new(device_index)?;
            let device_info = backend.device_info()?;
            let boxed: Box<dyn MiningBackend> = Box::new(backend);
            Ok((std::sync::Arc::new(tokio::sync::Mutex::new(boxed)), device_info))
        }
        _ => {
            anyhow::bail!("Unsupported algorithm: {}", algo);
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

    // Initialize mining backends (one per selected GPU)
    tracing::info!("Initializing mining backends for {} GPU(s)...", selected_devices.len());
    
    let mut gpu_miners = Vec::new();
    for device in &selected_devices {
        match create_backend_for_device_sync(&args.algo, device.id) {
            Ok((backend, device_info)) => {
                let gpu_miner = GpuMiner::new(device.clone(), backend, device_info);
                gpu_miners.push(gpu_miner);
                println!("   {} GPU #{}: {} initialized", "✓".green(), device.id, device.name);
            }
            Err(e) => {
                eprintln!("   {} Failed to initialize GPU #{} ({}): {}", "❌".red(), device.id, device.name, e);
                eprintln!("   {} Continuing with remaining GPUs...", "⚠️".yellow());
            }
        }
    }
    
    if gpu_miners.is_empty() {
        eprintln!("\n{}", "❌ No GPUs successfully initialized!".red().bold());
        eprintln!("{}", "Cannot continue mining.".red());
        std::process::exit(1);
    }
    
    println!("\n{}", "=== GPU Mining Initialized ===".green().bold());
    for gpu_miner in &gpu_miners {
        println!("   GPU #{}: {} ({})", 
            gpu_miner.device.id,
            gpu_miner.device_info.name,
            {
                let (maj, min) = gpu_miner.device_info.compute_capability;
                format!("{}.{}", maj, min)
            }
        );
    }

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
    
    // Channel for GPU workers to send statistics and shares to main thread
    let (stats_tx, mut stats_rx) = mpsc::channel::<GpuMiningResult>(32);
    
    // Global mining statistics (aggregated across all GPUs)
    let mut global_stats = MiningStats::new();
    let job_count = 0;
    
    // Long-term windows for 1h/6h/24h views (sample per chunk)
    let mut long_samples: VecDeque<(std::time::Instant, u64)> = VecDeque::new();
    
    // Ctrl+C handler with atomic flag
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                println!("\n{}", "⚠️  Interrupt signal received, shutting down...".yellow().bold());
                shutdown_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Err(err) => {
                eprintln!("Unable to listen for shutdown signal: {}", err);
            }
        }
    });
    
    // Start GPU mining tasks (one per GPU)
    let stratum_client = std::sync::Arc::new(tokio::sync::Mutex::new(stratum_client));
    
    // Collect all GPU mining tasks
    let mut mining_tasks = Vec::new();
    for gpu_miner in gpu_miners {
        let stats_tx = stats_tx.clone();
        let shutdown = shutdown.clone();
        let stratum_client = stratum_client.clone();
        
        let task = tokio::spawn(async move {
            gpu_mining_task(gpu_miner, stats_tx, shutdown, stratum_client).await;
        });
        mining_tasks.push(task);
    }
    
    // Create a dummy gpu_miners vec for stats display (will be empty but that's ok for now)
    let gpu_miners: Vec<GpuMiner> = vec![];
    
    // Main mining loop - manage GPU results and UI
    loop {
        // Check for shutdown (strong ordering)
        if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            println!("\n{}", "Shutting down miner...".yellow());
            
            // Final statistics
            println!("\n{}", "=== Final Statistics ===".cyan().bold());
            println!("   {} {}", "Total Hashes:".green(), global_stats.hashes);
            println!("   {} {}", "Shares Found:".green(), global_stats.shares_found);
            println!("   {} {}", "Shares Accepted:".green(), global_stats.shares_accepted);
            println!("   {} {}", "Shares Rejected:".green(), global_stats.shares_rejected);
            
            break;
        }
        
        tokio::select! {
            // Handle GPU results
            Some(gpu_result) = stats_rx.recv() => {
                match gpu_result {
                    GpuMiningResult::StatsUpdate { gpu_index, hashes, kernel_ms } => {
                        // Update global stats
                        global_stats.hashes += hashes;
                        
                        // Log per-GPU stats
                        tracing::trace!("GPU #{}: {} hashes, {} ms kernel", gpu_index, hashes, kernel_ms);
                    }
                    GpuMiningResult::ShareFound { gpu_index, nonce, hash, extranonce2 } => {
                        global_stats.shares_found += 1;
                        
                        // Submit share to pool
                        let extranonce2_hex = hex::encode(&extranonce2);
                        let nonce_hex = format!("{:08x}", nonce);

                        tracing::info!(
                            "GPU #{} found share: job_id=? extranonce2={} ntime=? nonce={}",
                            gpu_index, extranonce2_hex, nonce_hex,
                        );

                        println!("\n{}", "🎉 SHARE FOUND!".green().bold());
                        println!("   {} GPU #{}", "Device:".green(), gpu_index);
                        println!("   {} 0x{:08x}", "Nonce:".green(), nonce);
                        println!("   {} {}", "Hash:".green(), hex::encode(hash));

                        println!("   {} Submitting share...", "📤".yellow());
                        
                        // Get current job for submission (simplified - would need to store current job)
                        // TODO: Store current job in GpuMiner for share submission
                        let current_job = StratumJob {
                            job_id: "unknown".to_string(),
                            prevhash: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                            coinb1: "".to_string(),
                            coinb2: "".to_string(),
                            merkle_branch: vec![],
                            version: "00000000".to_string(),
                            nbits: "00000000".to_string(),
                            ntime: "00000000".to_string(),
                            clean_jobs: false,
                            difficulty: 1.0,
                        };
                        
                        let stratum_lock = stratum_client.lock().await;
                        match stratum_lock.submit_share(
                            &current_job.job_id,
                            &extranonce2_hex,
                            &current_job.ntime,
                            &nonce_hex
                        ).await {
                                Ok(true) => {
                                    global_stats.shares_accepted += 1;
                                    // WildRig-style share accepted log
                                    println!("[{}] accepted ({}/{})       diff {:.2}G    GPU#{}   ({:.0} ms)",
                                        chrono::Local::now().format("%H:%M:%S"),
                                        global_stats.shares_accepted,
                                        global_stats.shares_rejected,
                                        current_job.difficulty / 1_000_000_000.0, // Convert to G
                                        gpu_index, // GPU index
                                        0 // TODO: track actual time
                                    );
                                }
                                Ok(false) => {
                                    global_stats.shares_rejected += 1;
                                    println!("   {} {}", "❌ Share rejected".red().bold(),
                                        format!("({} rejected)", global_stats.shares_rejected).dimmed());
                                }
                                Err(e) => {
                                    global_stats.shares_rejected += 1;
                                    eprintln!("   {} Submit error: {}", "❌".red(), e);
                                }
                        }
                    }
                    GpuMiningResult::Error { gpu_index, error } => {
                        eprintln!("   {} GPU #{} error: {}", "❌".red(), gpu_index, error);
                        // Continue with other GPUs
                    }
                }
            }
            
            // Periodic UI update
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                // Print WildRig-style stats for all GPUs
                print_wildrig_stats_multi_gpu(&gpu_miners, &global_stats).await?;
            }
        }
        
        // Display statistics every 5 jobs
        if job_count % 5 == 0 && job_count > 0 {
            println!("\n{}", "=== Statistics ===".cyan());
            println!("   {} {}", "Total Hashes:".green(), global_stats.hashes);
            println!("   {} {}", "Shares Found:".green(), global_stats.shares_found);
            // Print WildRig-like statistics block (best-effort)
            print_wildrig_stats_multi_gpu(&gpu_miners, &global_stats).await?;
            // Add a long-sample for the last processed chunk to support 1h/6h/24h views
            let now = std::time::Instant::now();
            long_samples.push_back((now, global_stats.hashes));
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

// Print stats in WildRig-like format (multi-GPU support)
async fn print_wildrig_stats_multi_gpu(
    gpu_miners: &[GpuMiner],
    global_stats: &MiningStats,
) -> anyhow::Result<()> {
    use std::fmt::Write as _;

    let mut buf = String::new();
    writeln!(buf, "--------------------------------------[Statistics]--------------------------------------").ok();
    writeln!(buf, " ID Name                         Hashrate Temp  Fan  Power   Eff CClk MClk     A   R   I").ok();
    writeln!(buf, "----------------------------------------------------------------------------------------").ok();

    let mut total_rate10 = 0.0;
    let mut total_power = 0.0;

    for (idx, gpu) in gpu_miners.iter().enumerate() {
        // Calculate 10s rate for this GPU (simplified - would need per-GPU history)
        let rate10 = gpu.stats.current_hashrate; // Use current hashrate as approximation
        total_rate10 += rate10;

        // Get GPU stats (placeholder - would need actual monitoring)
        let gstat = get_gpu_stats(idx).unwrap_or((None, None, None, None, None));

        // Calculate efficiency (MH/s per Watt)
        let efficiency = if let Some(power) = gstat.2 {
            if power > 0.0 {
                (rate10 / 1_000_000.0) / power
            } else {
                0.0
            }
        } else {
            0.0
        };

        total_power += gstat.2.unwrap_or(0.0);

        // Per-device row - WildRig style
        writeln!(buf, " #{} {:26} {:9.2} MH/s  {}C  {}%  {:.1}W {:.3} {} {}     {}   {}   {}",
            idx,
            gpu.device_info.name,
            rate10 / 1_000_000.0,
            gstat.0.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string()),
            gstat.1.map(|f| f.to_string()).unwrap_or_else(|| "-".to_string()),
            gstat.2.unwrap_or(0.0),
            efficiency,
            gstat.3.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string()),
            gstat.4.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string()),
            global_stats.shares_accepted,
            if global_stats.shares_rejected > 0 { global_stats.shares_rejected.to_string() } else { "-".to_string() },
            "-"  // Ignored shares
        ).ok();
    }

    writeln!(buf, "----------------------------------------------------------------------------------------").ok();

    // Calculate rates for different time windows (simplified - would need proper history)
    let rate60 = total_rate10 * 0.9; // Approximation
    let rate15m = total_rate10 * 0.8; // Approximation

    writeln!(buf, " 10s: {:>28.2} MH/s Power: {:>7.1}W            Accepted: {:>8}",
             total_rate10 / 1_000_000.0, total_power, global_stats.shares_accepted).ok();
    writeln!(buf, " 60s: {:>28.2} MH/s                             Rejected: {:>8}",
             rate60 / 1_000_000.0, if global_stats.shares_rejected > 0 { global_stats.shares_rejected.to_string() } else { "-".to_string() }).ok();
    writeln!(buf, " 15m: {:>28.2} MH/s                             Ignored: {:>9}",
             if rate15m > 0.0 { rate15m / 1_000_000.0 } else { 0.0 }, "-").ok();
    writeln!(buf, "[{}]----------------------------------------------------------[ver. {}]",
             humantime::format_duration(std::time::Instant::now().elapsed()), env!("CARGO_PKG_VERSION")).ok();

    println!("{}", buf);

    Ok(())
}
