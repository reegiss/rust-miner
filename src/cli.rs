use clap::Parser;
use colored::*;

/// rust-miner - High-performance GPU cryptocurrency miner
#[derive(Parser, Debug)]
#[command(name = "rust-miner")]
#[command(author = "Rust Miner Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "High-performance GPU cryptocurrency miner written in Rust", long_about = None)]
pub struct Args {
    /// Mining algorithm (e.g., qhash, ethash, kawpow)
    #[arg(short, long, value_name = "ALGORITHM")]
    pub algo: String,

    /// Mining pool URL (e.g., pool.example.com:3333)
    #[arg(short = 'o', long, value_name = "URL")]
    pub url: String,

    /// Wallet address and optional worker name (e.g., wallet.worker)
    #[arg(short, long, value_name = "WALLET")]
    pub user: String,

    /// Pool password (default: "x")
    #[arg(short, long, default_value = "x", value_name = "PASSWORD")]
    pub pass: String,

    /// GPU device IDs to use (comma-separated, e.g., 0,1,2)
    #[arg(short, long, value_name = "DEVICES")]
    pub gpu: Option<String>,

    /// Use OpenCL backend instead of CUDA
    #[arg(long)]
    pub opencl: bool,

    /// Number of threads for work distribution (default: auto)
    #[arg(short, long, value_name = "THREADS")]
    pub threads: Option<usize>,

    /// Enable debug logging
    #[arg(short, long)]
    pub debug: bool,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

pub fn display_banner() {
    let banner = format!(
        r#"
{}
{}  {}
{}  {}
{}
{}  {}
{}  {}
{}
"#,
        "╔════════════════════════════════════════════════════════════╗".bright_cyan(),
        "║".bright_cyan(),
        "🦀 rust-miner - GPU Cryptocurrency Miner".bright_white().bold(),
        "║".bright_cyan(),
        format!("   Version {} | CUDA + OpenCL Support", env!("CARGO_PKG_VERSION")).bright_green(),
        "╠════════════════════════════════════════════════════════════╣".bright_cyan(),
        "║".bright_cyan(),
        "⚡ High Performance | Cross-Platform | GPU Required".yellow(),
        "║".bright_cyan(),
        "🔗 Built with Rust - Maximum Safety & Speed".bright_magenta(),
        "╚════════════════════════════════════════════════════════════╝".bright_cyan(),
    );

    println!("{}", banner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // This would normally use clap's testing utilities
        // Just a placeholder for now
    }
}
