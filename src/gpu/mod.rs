pub mod cuda;

use std::fmt;

/// GPU backend type (CUDA-only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA")
    }
}

/// Information about a detected GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID (0, 1, 2, etc.)
    pub id: usize,
    
    /// Device name (e.g., "GeForce GTX 1660 SUPER")
    pub name: String,
    
    /// Backend type (CUDA or OpenCL)
    pub backend: GpuBackend,
    
    /// Total memory in bytes
    pub memory_total: u64,
    
    /// Number of compute units (CUDA cores, Stream Processors, etc.)
    pub compute_units: u32,
    
    /// Clock speed in MHz
    pub clock_mhz: u32,
    
    /// PCI Bus ID (for multi-GPU systems)
    pub pci_bus_id: Option<String>,
    
    /// Compute capability (CUDA) or OpenCL version
    pub compute_capability: String,
}

impl fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {} - {} MB VRAM, {} compute units @ {} MHz (CC: {}, PCI: {})",
            self.id,
            self.backend,
            self.name,
            self.memory_total / (1024 * 1024),
            self.compute_units,
            self.clock_mhz,
            self.compute_capability,
            self.pci_bus_id.as_ref().unwrap_or(&"N/A".to_string())
        )
    }
}

/// Result type for GPU detection
pub type GpuResult<T> = Result<T, GpuError>;

/// Errors that can occur during GPU detection
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No compatible NVIDIA GPU detected")]
    NoGpuFound,
    
    #[error("CUDA runtime error: {0}")]
    CudaError(String),
    
    #[error("GPU {0} not found")]
    DeviceNotFound(usize),
    
    #[error("Invalid GPU device ID: {0}")]
    InvalidDeviceId(String),
}

/// Detect all available GPUs (CUDA-only)
pub fn detect_gpus() -> GpuResult<Vec<GpuDevice>> {
    // CUDA is the only supported backend
    match cuda::detect_cuda_devices() {
        Ok(devices) => {
            tracing::info!("Found {} CUDA device(s)", devices.len());
            Ok(devices)
        }
        Err(e) => {
            tracing::error!("CUDA detection failed: {}", e);
            Err(GpuError::NoGpuFound)
        }
    }
}

/// Select GPUs by ID from command-line argument (e.g., "0,1,2")
pub fn select_gpus(gpu_arg: Option<String>, all_devices: &[GpuDevice]) -> GpuResult<Vec<GpuDevice>> {
    match gpu_arg {
        Some(ids_str) => {
            let mut selected = Vec::new();
            
            for id_str in ids_str.split(',') {
                let id = id_str.trim().parse::<usize>()
                    .map_err(|_| GpuError::InvalidDeviceId(id_str.to_string()))?;
                
                let device = all_devices.iter()
                    .find(|d| d.id == id)
                    .ok_or(GpuError::DeviceNotFound(id))?;
                
                selected.push(device.clone());
            }
            
            Ok(selected)
        }
        None => {
            // Use all available devices
            Ok(all_devices.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(format!("{}", GpuBackend::Cuda), "CUDA");
    }

    #[test]
    fn test_select_gpus_all() {
        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Test GPU 1".to_string(),
                backend: GpuBackend::Cuda,
                memory_total: 6_000_000_000,
                compute_units: 1408,
                clock_mhz: 1785,
                pci_bus_id: None,
                compute_capability: "7.5".to_string(),
            },
            GpuDevice {
                id: 1,
                name: "Test GPU 2".to_string(),
                backend: GpuBackend::Cuda,
                memory_total: 8_000_000_000,
                compute_units: 2048,
                clock_mhz: 1800,
                pci_bus_id: None,
                compute_capability: "8.6".to_string(),
            },
        ];
        
        let selected = select_gpus(None, &devices).unwrap();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_gpus_specific() {
        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Test GPU 1".to_string(),
                backend: GpuBackend::Cuda,
                memory_total: 6_000_000_000,
                compute_units: 1408,
                clock_mhz: 1785,
                pci_bus_id: None,
                compute_capability: "7.5".to_string(),
            },
            GpuDevice {
                id: 1,
                name: "Test GPU 2".to_string(),
                backend: GpuBackend::Cuda,
                memory_total: 8_000_000_000,
                compute_units: 2048,
                clock_mhz: 1800,
                pci_bus_id: None,
                compute_capability: "8.6".to_string(),
            },
        ];
        
        let selected = select_gpus(Some("0".to_string()), &devices).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, 0);
    }
}
