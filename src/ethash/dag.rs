use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Minimal Etchash/Ethash DAG information
#[derive(Debug, Clone)]
pub struct DagInfo {
    pub epoch: u64,
    pub seed_hash: [u8; 32],
    pub dataset_bytes: u64,
    pub cache_bytes: u64,
    pub dataset_path: PathBuf,
    pub cache_path: PathBuf,
}

/// PoW algorithms supported for DAG sizing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowAlgo {
    Ethash,
    Etchash,
}

/// ETHASH epoch length in blocks (Ethereum-like)
pub const ETHASH_EPOCH_LENGTH: u64 = 30_000;
/// ETCHASH epoch length in blocks (ECIP-1099)
pub const ETCHASH_EPOCH_LENGTH: u64 = 60_000;

/// Compute epoch from block height
pub fn epoch_from_height(height: u64) -> u64 { height / ETHASH_EPOCH_LENGTH }

/// Compute epoch from block height based on algorithm
pub fn epoch_from_height_for_algo(height: u64, algo: PowAlgo) -> u64 {
    match algo {
        PowAlgo::Ethash => height / ETHASH_EPOCH_LENGTH,
        PowAlgo::Etchash => height / ETCHASH_EPOCH_LENGTH,
    }
}

/// Default DAG directory: ~/.cache/rust-miner/dag
pub fn default_dag_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        Path::new(&home).join(".cache").join("rust-miner").join("dag")
    } else {
        PathBuf::from("dag")
    }
}

fn parse_seed_hash(hex: &str) -> Result<[u8; 32]> {
    let h = hex.trim_start_matches("0x");
    let bytes = hex::decode(h)?;
    if bytes.len() != 32 { return Err(anyhow!("Invalid seed hash length: {}", bytes.len())); }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Simple primality test for Ethash size rounding
fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n.is_multiple_of(2) { return n == 2; }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) { return false; }
        d += 2;
    }
    true
}

// Constants from Ethash specification (sizes in bytes)
const HASH_BYTES: u64 = 64;               // Keccak-512
const MIX_BYTES: u64 = 128;               // Mix size
const CACHE_BYTES_INIT: u64 = 16 * 1024 * 1024;  // 16 MiB
const CACHE_BYTES_GROWTH: u64 = 131_072;        // 128 KiB per epoch
const DATASET_BYTES_INIT: u64 = 1 << 30;        // 1 GiB
const DATASET_BYTES_GROWTH: u64 = 1 << 23;      // 8 MiB per epoch

/// Ethash cache size for a given epoch
pub fn ethash_cache_size(epoch: u64) -> u64 {
    let mut sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * epoch - HASH_BYTES;
    while !is_prime(sz / HASH_BYTES) {
        sz -= 2 * HASH_BYTES;
    }
    sz
}

/// Ethash dataset size for a given epoch
/// Note: For ETC (Etchash, ECIP-1099), dataset sizes diverge after epoch 390.
/// We'll integrate the Etchash adjustment alongside GPU DAG generation.
pub fn ethash_dataset_size(epoch: u64) -> u64 {
    let mut sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * epoch - MIX_BYTES;
    while !is_prime(sz / MIX_BYTES) {
        sz -= 2 * MIX_BYTES;
    }
    sz
}

fn dataset_filename(epoch: u64) -> String {
    format!("etchash-DAG-epoch-{:05}.bin", epoch)
}

fn cache_filename(epoch: u64) -> String {
    format!("etchash-CACHE-epoch-{:05}.bin", epoch)
}

/// Prepare DAG info from pool job data (seed hash and optional height)
/// If `algo_hint` is Some("etchash"), use Etchash epoch sizing; otherwise default to Ethash.
pub fn prepare_from_pool(seed_hash_hex: &str, height: Option<u64>, algo_hint: Option<&str>) -> Result<DagInfo> {
    let seed_hash = parse_seed_hash(seed_hash_hex)?;
    // Detect algorithm from hint (case-insensitive), default to Ethash
    let pow_algo = match algo_hint.map(|s| s.to_ascii_lowercase()) {
        Some(a) if a.contains("etchash") => PowAlgo::Etchash,
        _ => PowAlgo::Ethash,
    };
    // Prefer height if provided; otherwise default to epoch 0
    let epoch = match height {
        Some(h) => epoch_from_height_for_algo(h, pow_algo),
        None => 0,
    };
    let dataset_bytes = ethash_dataset_size(epoch);
    let cache_bytes = ethash_cache_size(epoch);

    let dir = default_dag_dir();
    fs::create_dir_all(&dir).ok();

    let dataset_path = dir.join(dataset_filename(epoch));
    let cache_path = dir.join(cache_filename(epoch));

    Ok(DagInfo { epoch, seed_hash, dataset_bytes, cache_bytes, dataset_path, cache_path })
}

/// Check whether dataset file exists and meets minimal size
pub fn dataset_on_disk(info: &DagInfo) -> bool {
    if let Ok(meta) = fs::metadata(&info.dataset_path) {
        meta.is_file() && meta.len() >= info.dataset_bytes / 2 // tolerant lower bound for now
    } else { false }
}
