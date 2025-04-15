use pyo3::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use sysinfo::{ProcessExt, System, SystemExt};
use chrono;
use serde_json::json;
use rand;

mod memory;
use memory::MemoryManager;

/// Python module for memory monitoring and management
#[pymodule]
fn memory_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_monitoring, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_memory_info, m)?)?;
    m.add_function(wrap_pyfunction!(allocate_memory, m)?)?;
    m.add_function(wrap_pyfunction!(deallocate_memory, m)?)?;
    m.add_function(wrap_pyfunction!(defragment_memory, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_memory_usage, m)?)?;
    Ok(())
}

/// Start the memory monitoring system
#[pyfunction]
fn start_monitoring(log_path: Option<String>) -> PyResult<()> {
    println!("Starting Real-Time Memory Manager from Python...");
    
    // Use provided path or default
    let jsonl_path = log_path.unwrap_or_else(|| 
        "d:\\clg\\COA\\2\\Self_healing_memory\\data\\memory_events.jsonl".to_string());
    println!("Using log path: {}", jsonl_path);
    
    let total_memory = get_system_memory();
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    // Initialize memory
    manager.init_memory();
    
    // Start status monitoring thread
    let manager_clone = Arc::clone(&manager);
    let _monitor_thread = thread::spawn(move || {
        loop {
            println!("\x1B[2J\x1B[1;1H");
            println!("Real-Time Memory Status:");
            println!("Total System Memory: {} GB", total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
            let status = MemoryManager::get_memory_status(&manager_clone);
            println!("{}", status);
            thread::sleep(Duration::from_millis(1000));
        }
    });
    
    // Start monitoring thread for RAG with custom path
    let jsonl_path_clone = jsonl_path.clone();
    monitor_memory_for_rag(Arc::clone(&manager), jsonl_path_clone);
    
    // Example operations from main.rs
    if let Some(addr1) = manager.allocate(1024 * 1024 * 100) {
        println!("Allocated 100MB at address {}", addr1);
    }
    
    thread::sleep(Duration::from_millis(2000));
    
    if let Some(addr2) = manager.allocate(1024 * 1024 * 200) {
        println!("Allocated 200MB at address {}", addr2);
    }
    
    // Main loop from main.rs
    let manager_clone = Arc::clone(&manager);
    thread::spawn(move || {
        loop {
            // Calculate health scores
            manager_clone.calculate_health();
            
            // Optional: Simulate usage
            if rand::random::<f32>() < 0.3 {
                simulate_usage(&manager_clone);
            }
            
            // Defragment periodically
            if rand::random::<f32>() < 0.1 {
                manager_clone.defragment();
            }
            
            thread::sleep(Duration::from_secs(1));
        }
    });
    
    println!("Memory monitoring started successfully!");
    Ok(())
}

/// Get system memory information
#[pyfunction]
fn get_system_memory_info() -> PyResult<String> {
    let total_memory = get_system_memory();
    let total_gb = total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
    Ok(format!("Total System Memory: {:.2} GB", total_gb))
}

/// Allocate memory of specified size
#[pyfunction]
fn allocate_memory(size_mb: usize) -> PyResult<Option<usize>> {
    let total_memory = get_system_memory();
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    let size_bytes = size_mb * 1024 * 1024;
    let addr = manager.allocate(size_bytes);
    
    Ok(addr)
}

/// Deallocate memory at specified address
#[pyfunction]
fn deallocate_memory(address: usize) -> PyResult<bool> {
    let total_memory = get_system_memory();
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    let result = manager.deallocate(address);
    Ok(result)
}

/// Defragment memory
#[pyfunction]
fn defragment_memory() -> PyResult<()> {
    let total_memory = get_system_memory();
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    manager.defragment();
    Ok(())
}

/// Simulate memory usage with random allocations and deallocations
#[pyfunction]
fn simulate_memory_usage() -> PyResult<()> {
    let total_memory = get_system_memory();
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    simulate_usage(&manager);
    Ok(())
}

// Function to get system memory
fn get_system_memory() -> usize {
    let sys = System::new_all();
    let total_memory = sys.total_memory();
    println!("Total system memory: {} GB", total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    total_memory as usize
}

// Function to print memory status
#[allow(dead_code)]
fn print_status(manager: &Arc<MemoryManager>) {
    println!("Memory Status:");
    println!("{}", MemoryManager::get_memory_status(manager));
}

// Function to simulate memory usage
fn simulate_usage(manager: &Arc<MemoryManager>) {
    // Simulate random allocations and deallocations
    for _ in 0..10 {
        let size = rand::random::<usize>() % (1024 * 1024 * 500) + 1;
        if let Some(addr) = manager.allocate(size) {
            thread::sleep(Duration::from_millis(100));
            manager.deallocate(addr);
        }
    }
}

// Function to monitor memory for RAG
fn monitor_memory_for_rag(manager: Arc<MemoryManager>, jsonl_path: String) {
    thread::spawn(move || {
        // Create data directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&jsonl_path).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Failed to create data directory: {}", e);
                return;
            }
        }
        
        // Main monitoring loop - run forever
        loop {
            // Get process information
            let mut sys = System::new_all();
            sys.refresh_processes();
            
            // Calculate system metrics
            let system_load = sys.load_average().one;
            let free_memory_percent = (sys.free_memory() as f64 / sys.total_memory() as f64) * 100.0;
            
            // Calculate fragmentation index
            let blocks = manager.get_blocks();
            let fragmentation_index = calculate_fragmentation_index(&blocks);
            
            // Get timestamp
            let timestamp = chrono::Local::now().to_rfc3339();
            
            // Create JSONL entry for RAG system
            let memory_status = MemoryManager::get_memory_status(&manager);
            
            // Collect top processes by memory usage
            let top_processes: Vec<_> = sys.processes()
                .iter()
                .map(|(pid, process)| {
                    json!({
                        "pid": pid.to_string(),
                        "name": process.name(),
                        "memory_kb": process.memory() / 1024,
                        "cpu_usage": process.cpu_usage()
                    })
                })
                .collect();
            
            // Create comprehensive JSON log entry
            let log_entry = json!({
                "timestamp": timestamp,
                "system_metrics": {
                    "load": system_load,
                    "free_memory_percent": free_memory_percent,
                    "fragmentation_index": fragmentation_index
                },
                "memory_status": memory_status,
                "processes": top_processes,
                "memory_blocks": blocks.iter().map(|block| {
                    json!({
                        "start_address": block.start_address,
                        "size": block.size,
                        "is_allocated": block.is_allocated,
                        "health_score": block.health_score
                    })
                }).collect::<Vec<_>>()
            });
            
            // Replace the manual file writing with the method call
            if let Err(e) = manager.write_memory_log(log_entry, &jsonl_path) {
                eprintln!("Failed to write memory log: {}", e);
            }

            thread::sleep(Duration::from_millis(100));
        }
    });
}

// Helper function to calculate fragmentation index
fn calculate_fragmentation_index(blocks: &Vec<memory::MemoryBlock>) -> f64 {
    if blocks.is_empty() {
        return 0.0;
    }
    
    let free_blocks = blocks.iter().filter(|b| !b.is_allocated).count();
    let total_blocks = blocks.len();
    
    // Simple fragmentation index: ratio of free blocks to total blocks
    (free_blocks as f64) / (total_blocks as f64)
}  // End of calculate_fragmentation_index function