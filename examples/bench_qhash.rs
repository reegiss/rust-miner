/// Simple benchmark for QHash kernel performance
/// This directly measures GPU kernel execution time without pool overhead

use std::time::Instant;

// Import from rust-miner
use rust_miner::cuda::CudaMiner;

fn main() -> anyhow::Result<()> {
    println!("═════════════════════════════════════════════");
    println!("  QHash Kernel Performance Benchmark");
    println!("═════════════════════════════════════════════");
    println!();

    // Initialize CUDA miner
    println!("Initializing CUDA miner...");
    // Use first CUDA device (index 0) for the benchmark
    let miner = CudaMiner::new(0)?;
    
    // Get device info
    let device_name = miner.device_name()?;
    let (cc_major, cc_minor) = miner.compute_capability()?;
    println!("Device: {}", device_name);
    println!("Compute Capability: {}.{}", cc_major, cc_minor);
    println!();

    // Test parameters
    let test_nonces: Vec<u32> = vec![
        10_000_000,   // 10M
        25_000_000,   // 25M
        50_000_000,   // 50M (current default)
        100_000_000,  // 100M
    ];

    // Dummy block header (76 bytes)
    let block_header = [0u8; 76];
    let ntime = 1234567890u32;
    let target = [0xFFu8; 32]; // Easy target (accepts all hashes)
    
    println!("Benchmark Configuration:");
    println!("  Block header: 76 bytes (zeros)");
    println!("  nTime: {}", ntime);
    println!("  Target: all 0xFF (all hashes valid)");
    println!();

    // Run benchmarks
    println!("Running kernel benchmarks...");
    println!();
    println!("{:<15} {:<15} {:<15} {:<15}", "Nonces", "Time (ms)", "MH/s", "Notes");
    println!("{:-<15} {:-<15} {:-<15} {:-<15}", "", "", "", "");

    let mut measurements = Vec::new();

    for num_nonces in test_nonces {
        // Warm-up run
        let _ = miner.mine_job(
            &block_header,
            ntime,
            &target,
            0,
            1_000_000, // Small warmup
        );

        // Actual benchmark runs (3x)
        let mut times = Vec::new();
        for run in 0..3 {
            let start = Instant::now();
            let _result = miner.mine_job(
                &block_header,
                ntime,
                &target,
                (run as u32) * num_nonces,
                num_nonces,
            );
            let elapsed_ms = start.elapsed().as_millis() as u32;
            times.push(elapsed_ms);
        }

        let avg_ms = times.iter().sum::<u32>() / times.len() as u32;
        let hashrate_mhs = (num_nonces as f64 / 1_000_000.0) / (avg_ms as f64 / 1000.0);
        
        measurements.push((num_nonces, avg_ms, hashrate_mhs));
        
        let variance = if times.len() > 1 {
            format!("(runs: {:?}ms)", times)
        } else {
            String::new()
        };
        
        println!(
            "{:<15} {:<15} {:<15.1} {}",
            format!("{}", num_nonces / 1_000_000),
            format!("{}ms", avg_ms),
            hashrate_mhs,
            variance
        );
    }

    println!();
    println!("═════════════════════════════════════════════");
    println!("Analysis:");
    println!("═════════════════════════════════════════════");
    
    // Check if hashrate scales with nonce count
    if measurements.len() > 1 {
        let first_rate = measurements[0].2;
        let last_rate = measurements[measurements.len() - 1].2;
        
        println!("First measurement:  {:.1} MH/s", first_rate);
        println!("Last measurement:   {:.1} MH/s", last_rate);
        println!("Variance:           {:.1}%", ((last_rate - first_rate) / first_rate) * 100.0);
        println!();
        
        if (last_rate - first_rate).abs() / first_rate < 0.1 {
            println!("✅ Hashrate is stable across batch sizes (good!)");
        } else {
            println!("⚠️  Hashrate varies significantly");
            println!("   → May indicate register spilling or occupancy issues");
        }
    }

    println!();
    println!("Interpretation:");
    println!("  - 37 MH/s  = slow (occupancy issue?)");
    println!("  - 100+ MH/s = good (matches WildRig potential)");
    println!("  - 500 MH/s = exceptional (full optimization)");
    println!();

    // Final measurement: sustained rate
    println!("Sustained performance test (10 × 50M nonces)...");
    let start = Instant::now();
    for i in 0..10 {
        let _ = miner.mine_job(
            &block_header,
            ntime,
            &target,
            i * 50_000_000,
            50_000_000,
        );
    }
    let total_elapsed = start.elapsed();
    let sustained_hashrate = (10.0 * 50_000_000.0 / 1_000_000.0) / total_elapsed.as_secs_f64();
    
    println!("Processed: 500M nonces in {:.1}s", total_elapsed.as_secs_f64());
    println!("Sustained hashrate: {:.1} MH/s", sustained_hashrate);
    println!();

    Ok(())
}
