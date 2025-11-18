use super::{GpuBackend, GpuDevice, GpuError, GpuResult};
use cudarc::driver::CudaDevice;

/// Detect CUDA-capable NVIDIA GPUs
pub fn detect_cuda_devices() -> GpuResult<Vec<GpuDevice>> {
    use cudarc::driver::sys::CUdevice_attribute;
    
    // Get device count via cudarc
    let device_count = match CudaDevice::count() {
        Ok(count) => count,
        Err(e) => {
            return Err(GpuError::CudaError(format!("Failed to get CUDA device count: {:?}", e)));
        }
    };
    
    if device_count == 0 {
        return Err(GpuError::NoGpuFound);
    }
    
    let mut devices = Vec::new();
    
    for device_id in 0..device_count {
        // Initialize device to query properties
        let device = match CudaDevice::new(device_id as usize) {
            Ok(dev) => dev,
            Err(e) => {
                tracing::warn!("Failed to initialize CUDA device {}: {:?}", device_id, e);
                continue;
            }
        };
        
        // Get device name
        let name = match device.name() {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("Failed to get name for device {}: {:?}", device_id, e);
                format!("CUDA Device {}", device_id)
            }
        };
        
        // Get compute capability
        let major = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(7);
        let minor = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(5);
        let compute_capability = format!("{}.{}", major, minor);
        
        // Get multiprocessor count (SMs)
        let multi_processor_count = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(22) as u32;
        
        // Get clock rate (kHz)
        let clock_rate_khz = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
            .unwrap_or(1785000) as u32;
        let clock_mhz = clock_rate_khz / 1000;
        
        // Get total memory (cudarc doesn't expose total_mem directly, get from ordinal)
        let total_memory = match cudarc::driver::result::mem_get_info() {
            Ok((_free, total)) => total,
            Err(_) => 6_000_000_000, // Fallback
        };
        
        // Get PCI info
        let pci_bus = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
            .unwrap_or(0);
        let pci_device = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
            .unwrap_or(0);
        let pci_domain = device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
            .unwrap_or(0);
        
        let pci_bus_id = Some(format!(
            "{:04x}:{:02x}:{:02x}.0",
            pci_domain, pci_bus, pci_device
        ));
        
        // Calculate CUDA cores from SMs
        // Turing (7.5) = 64 cores per SM
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
        
        let gpu_device = GpuDevice {
            id: device_id as usize,
            name,
            backend: GpuBackend::Cuda,
            memory_total: total_memory as u64,
            compute_units,
            clock_mhz,
            pci_bus_id,
            compute_capability,
        };
        
        devices.push(gpu_device);
    }
    
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
