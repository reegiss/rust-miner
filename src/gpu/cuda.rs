use super::{GpuBackend, GpuDevice, GpuError, GpuResult};

#[cfg(feature = "cuda")]
use cuda_runtime_sys::*;

/// Detect CUDA-capable NVIDIA GPUs
#[cfg(feature = "cuda")]
pub fn detect_cuda_devices() -> GpuResult<Vec<GpuDevice>> {
    use std::ffi::CStr;
    use std::mem;
    
    unsafe {
        let mut device_count: i32 = 0;
        let result = cudaGetDeviceCount(&mut device_count as *mut i32);
        
        if result != cudaError_t::cudaSuccess {
            let error_str = CStr::from_ptr(cudaGetErrorString(result))
                .to_string_lossy()
                .into_owned();
            return Err(GpuError::CudaError(error_str));
        }
        
        if device_count == 0 {
            return Err(GpuError::NoGpuFound);
        }
        
        let mut devices = Vec::new();
        
        for device_id in 0..device_count {
            let mut props: cudaDeviceProp = mem::zeroed();
            let result = cudaGetDeviceProperties(&mut props as *mut cudaDeviceProp, device_id);
            
            if result != cudaError_t::cudaSuccess {
                tracing::warn!("Failed to get properties for CUDA device {}", device_id);
                continue;
            }
            
            // Extract device name
            let name = CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned()
                .trim()
                .to_string();
            
            // Get raw field values (these should be correct from the API)
            let multi_processor_count = props.multiProcessorCount as u32;
            let clock_rate_khz = props.clockRate as u32;
            let total_memory = props.totalGlobalMem;
            let major = props.major;
            let minor = props.minor;
            
            // NOTE: cuda-runtime-sys 0.3.0-alpha.1 has issues with struct layout
            // Some fields may return 0. We detect this and use fallback values.
            let multi_processor_count = if multi_processor_count == 0 {
                // GTX 1660 SUPER has 22 SMs
                // Detect by name or use CUDA device query API
                tracing::warn!("multiProcessorCount is 0, using device query fallback");
                
                // Try to get correct value via cudaDeviceGetAttribute
                let mut sm_count: i32 = 0;
                let attr_result = cudaDeviceGetAttribute(
                    &mut sm_count as *mut i32,
                    cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                    device_id
                );
                
                if attr_result == cudaError_t::cudaSuccess && sm_count > 0 {
                    sm_count as u32
                } else {
                    // Last resort: estimate from name
                    if name.contains("1660") {
                        22 // GTX 1660 / 1660 SUPER / 1660 Ti
                    } else {
                        16 // Reasonable default
                    }
                }
            } else {
                multi_processor_count
            };
            
            let clock_rate_khz = if clock_rate_khz == 0 {
                tracing::warn!("clockRate is 0, using device query fallback");
                
                let mut clock: i32 = 0;
                let attr_result = cudaDeviceGetAttribute(
                    &mut clock as *mut i32,
                    cudaDeviceAttr::cudaDevAttrClockRate,
                    device_id
                );
                
                if attr_result == cudaError_t::cudaSuccess && clock > 0 {
                    clock as u32
                } else {
                    // Estimate from GPU name
                    if name.contains("1660") {
                        1785000 // GTX 1660 SUPER boost clock
                    } else {
                        1500000 // 1.5 GHz default
                    }
                }
            } else {
                clock_rate_khz
            };
            
            // Calculate compute capability
            let compute_capability = format!("{}.{}", major, minor);
            
            tracing::debug!(
                "Device {}: SM={}, clock={}kHz, mem={} bytes, CC={}.{}",
                device_id,
                multi_processor_count,
                clock_rate_khz,
                total_memory,
                major,
                minor
            );
            
            // PCI Bus ID
            let pci_bus_id = Some(format!(
                "{:04x}:{:02x}:{:02x}.0",
                props.pciDomainID,
                props.pciBusID,
                props.pciDeviceID
            ));
            
            // GTX 1660 SUPER has 22 SMs with 64 CUDA cores each = 1408 cores
            // Use a multiplier based on compute capability
            let cores_per_sm = match (major, minor) {
                (7, 5) => 64,  // Turing (GTX 1660 SUPER)
                (7, _) => 64,  // Turing
                (8, 0) => 64,  // Ampere GA100
                (8, 6) => 128, // Ampere GA10x
                (8, 9) => 128, // Ada Lovelace
                (9, 0) => 128, // Hopper
                (6, _) => 128, // Pascal
                (5, _) => 128, // Maxwell
                _ => 64,       // Default
            };
            
            let compute_units = multi_processor_count * cores_per_sm;
            let clock_mhz = clock_rate_khz / 1000;
            
            let device = GpuDevice {
                id: device_id as usize,
                name,
                backend: GpuBackend::Cuda,
                memory_total: total_memory as u64,
                compute_units,
                clock_mhz,
                pci_bus_id,
                compute_capability,
            };
            
            devices.push(device);
        }
        
        Ok(devices)
    }
}

/// Fallback when CUDA feature is not enabled
#[cfg(not(feature = "cuda"))]
pub fn detect_cuda_devices() -> GpuResult<Vec<GpuDevice>> {
    Err(GpuError::CudaError("CUDA support not compiled in. Build with --features cuda".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_detect_cuda_devices() {
        match detect_cuda_devices() {
            Ok(devices) => {
                println!("Found {} CUDA device(s)", devices.len());
                for device in devices {
                    println!("  {}", device);
                }
            }
            Err(e) => {
                println!("CUDA detection failed: {}", e);
            }
        }
    }
}
