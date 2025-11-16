use super::{GpuDevice, GpuError, GpuResult};

#[cfg(feature = "opencl")]
use super::GpuBackend;

#[cfg(feature = "opencl")]
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
#[cfg(feature = "opencl")]
use opencl3::platform::get_platforms;
#[cfg(feature = "opencl")]
use opencl3::types::{
    CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS,
    CL_DEVICE_MAX_CLOCK_FREQUENCY, CL_DEVICE_NAME, CL_DEVICE_VERSION,
};

/// Detect OpenCL-capable GPUs
#[cfg(feature = "opencl")]
pub fn detect_opencl_devices() -> GpuResult<Vec<GpuDevice>> {
    let platforms = get_platforms()
        .map_err(|e| GpuError::OpenClError(format!("Failed to get platforms: {:?}", e)))?;
    
    if platforms.is_empty() {
        return Err(GpuError::OpenClError("No OpenCL platforms found".to_string()));
    }
    
    let mut all_devices = Vec::new();
    let mut device_id = 0;
    
    for platform in platforms {
        let devices = platform
            .get_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| GpuError::OpenClError(format!("Failed to get devices: {:?}", e)))?;
        
        for device in devices {
            match extract_device_info(device, device_id) {
                Ok(gpu_device) => {
                    all_devices.push(gpu_device);
                    device_id += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to extract OpenCL device info: {}", e);
                }
            }
        }
    }
    
    if all_devices.is_empty() {
        return Err(GpuError::NoGpuFound);
    }
    
    Ok(all_devices)
}

#[cfg(feature = "opencl")]
fn extract_device_info(device: Device, id: usize) -> GpuResult<GpuDevice> {
    // Get device name
    let name = device
        .name()
        .map_err(|e| GpuError::OpenClError(format!("Failed to get device name: {:?}", e)))?;
    
    // Get total memory
    let memory_total = device
        .get_info(CL_DEVICE_GLOBAL_MEM_SIZE)
        .map_err(|e| GpuError::OpenClError(format!("Failed to get memory size: {:?}", e)))?
        .to_ulong();
    
    // Get compute units
    let compute_units = device
        .get_info(CL_DEVICE_MAX_COMPUTE_UNITS)
        .map_err(|e| GpuError::OpenClError(format!("Failed to get compute units: {:?}", e)))?
        .to_uint();
    
    // Get clock speed
    let clock_mhz = device
        .get_info(CL_DEVICE_MAX_CLOCK_FREQUENCY)
        .map_err(|e| GpuError::OpenClError(format!("Failed to get clock frequency: {:?}", e)))?
        .to_uint();
    
    // Get OpenCL version
    let version = device
        .version()
        .map_err(|e| GpuError::OpenClError(format!("Failed to get OpenCL version: {:?}", e)))?;
    
    Ok(GpuDevice {
        id,
        name,
        backend: GpuBackend::OpenCL,
        memory_total,
        compute_units,
        clock_mhz,
        pci_bus_id: None, // OpenCL doesn't easily provide PCI bus ID
        compute_capability: version,
    })
}

/// Fallback when OpenCL feature is not enabled
#[cfg(not(feature = "opencl"))]
pub fn detect_opencl_devices() -> GpuResult<Vec<GpuDevice>> {
    Err(GpuError::OpenClError("OpenCL support not compiled in. Build with --features opencl".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "opencl")]
    fn test_detect_opencl_devices() {
        match detect_opencl_devices() {
            Ok(devices) => {
                println!("Found {} OpenCL device(s)", devices.len());
                for device in devices {
                    println!("  {}", device);
                }
                assert!(!devices.is_empty());
            }
            Err(e) => {
                println!("OpenCL detection failed: {}", e);
            }
        }
    }
}
