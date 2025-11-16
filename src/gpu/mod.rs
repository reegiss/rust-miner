pub mod cuda;
pub mod opencl;

use std::fmt;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
        }
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
    #[error("No compatible GPU detected")]
    NoGpuFound,
    
    #[error("CUDA runtime error: {0}")]
    CudaError(String),
    
    #[error("OpenCL error: {0}")]
    OpenClError(String),
    
    #[error("GPU {0} not found")]
    DeviceNotFound(usize),
    
    #[error("Invalid GPU device ID: {0}")]
    InvalidDeviceId(String),
}

/// Detect all available GPUs
pub fn detect_gpus() -> GpuResult<Vec<GpuDevice>> {
    let mut devices = Vec::new();
    
    // Try CUDA first (priority)
    match cuda::detect_cuda_devices() {
        Ok(mut cuda_devices) => {
            tracing::info!("Found {} CUDA device(s)", cuda_devices.len());
            devices.append(&mut cuda_devices);
        }
        Err(e) => {
            tracing::debug!("CUDA detection failed: {}", e);
        }
    }
    
    // Try OpenCL as fallback
    match opencl::detect_opencl_devices() {
        Ok(mut opencl_devices) => {
            tracing::info!("Found {} OpenCL device(s)", opencl_devices.len());
            
            // Filter out duplicates (same GPU detected via both CUDA and OpenCL)
            opencl_devices.retain(|ocl_dev| {
                !devices.iter().any(|cuda_dev| {
                    // Simple heuristic: if names are similar and memory matches, likely same device
                    cuda_dev.name.contains(&ocl_dev.name[..10.min(ocl_dev.name.len())])
                        && cuda_dev.memory_total == ocl_dev.memory_total
                })
            });
            
            devices.append(&mut opencl_devices);
        }
        Err(e) => {
            tracing::debug!("OpenCL detection failed: {}", e);
        }
    }
    
    if devices.is_empty() {
        return Err(GpuError::NoGpuFound);
    }
    
    // Re-index devices sequentially
    for (idx, device) in devices.iter_mut().enumerate() {
        device.id = idx;
    }
    
    Ok(devices)
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
        assert_eq!(format!("{}", GpuBackend::OpenCL), "OpenCL");
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
                backend: GpuBackend::OpenCL,
                memory_total: 8_000_000_000,
                compute_units: 2048,
                clock_mhz: 1800,
                pci_bus_id: None,
                compute_capability: "3.0".to_string(),
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
                backend: GpuBackend::OpenCL,
                memory_total: 8_000_000_000,
                compute_units: 2048,
                clock_mhz: 1800,
                pci_bus_id: None,
                compute_capability: "3.0".to_string(),
            },
        ];
        
        let selected = select_gpus(Some("0".to_string()), &devices).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, 0);
    }
}
