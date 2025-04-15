use std::sync::Arc;
use std::thread;
use std::time::Duration;
use sysinfo::{ProcessExt, System, SystemExt};
use std::io::Write;
use chrono;
use serde_json::json;
use rand;
use std::fs::OpenOptions;

// Import the memory module
mod memory;
use memory::MemoryManager;

// Function to read healing actions from Python
fn read_healing_actions() -> Vec<serde_json::Value> {
    let action_path = "d:\\clg\\COA\\2\\Self_healing_memory\\data\\healing_actions.jsonl";
    let mut actions = Vec::new();
    
    // Check if file exists
    if !std::path::Path::new(action_path).exists() {
        return actions;
    }
    
    // Read the file line by line
    if let Ok(file) = std::fs::File::open(action_path) {
        let reader = std::io::BufReader::new(file);
        
        // Use BufRead trait from std::io
        use std::io::BufRead;
        
        for line_result in reader.lines() {
            if let Ok(line) = line_result {
                if let Ok(action) = serde_json::from_str::<serde_json::Value>(&line) {
                    actions.push(action);
                }
            }
        }
    }
    
    // Clear the file after reading
    if let Ok(_) = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(action_path) {
        // File truncated
    }
    
    actions
}

// Function to process healing actions
fn process_healing_actions(manager: &Arc<MemoryManager>, actions: Vec<serde_json::Value>) {
    for action in actions {
        let action_type = action["action_type"].as_str().unwrap_or("");
        let reason = action["reason"].as_str().unwrap_or("Unknown reason");
        
        match action_type {
            "defragment_memory" => {
                println!("Executing defragmentation: {}", reason);
                manager.defragment();
            },
            "compact_memory" => {
                println!("Executing memory compaction: {}", reason);
                manager.defragment(); // Use defragment as compact
            },
            "reallocate_memory" => {
                println!("Executing memory reallocation: {}", reason);
                // Reallocate memory blocks with low health scores
                manager.calculate_health();
                // Additional reallocation logic could be added here
            },
            "terminate_process" => {
                if let Some(target) = action["target"].as_str() {
                    println!("Would terminate process: {} ({})", target, reason);
                    // In a real system, you'd use OS-specific APIs to terminate processes
                }
            },
            _ => println!("Unknown action type: {}", action_type),
        }
    }
}

// Function to monitor memory for RAG
fn monitor_memory_for_rag(manager: Arc<MemoryManager>) {
    thread::spawn(move || {
        // Create data directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all("d:\\clg\\COA\\2\\Self_healing_memory\\data") {
            eprintln!("Failed to create data directory: {}", e);
            return;
        }
        
        // Set up JSONL file for RAG ingestion
        let jsonl_path = "d:\\clg\\COA\\2\\Self_healing_memory\\data\\memory_events.jsonl";
        
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
            
            // Get timestamp in format that Python expects (YYYY-MM-DD HH:MM:SS)
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            
            // Create JSONL entry for RAG system
            let memory_status = MemoryManager::get_memory_status(&manager);
            
            // Collect top processes by memory usage
            let mut processes = sys.processes()
                .iter()
                .map(|(pid, process)| {
                    json!({
                        "pid": pid.to_string(),
                        "name": process.name(),
                        "memory_kb": process.memory() / 1024,
                        "cpu_usage": process.cpu_usage()
                    })
                })
                .collect::<Vec<_>>();
            
            // Sort processes by memory usage
            processes.sort_by(|a, b| {
                let a_mem = a["memory_kb"].as_u64().unwrap_or(0);
                let b_mem = b["memory_kb"].as_u64().unwrap_or(0);
                b_mem.cmp(&a_mem) // Sort in descending order
            });
            
            // Take top 10 processes
            let top_processes = if processes.len() > 10 {
                processes[0..10].to_vec()
            } else {
                processes
            };
            
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
            
            // Write to JSONL file
            match std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(jsonl_path) {
                    Ok(mut jsonl_file) => {
                        if let Err(e) = writeln!(jsonl_file, "{}", log_entry.to_string()) {
                            eprintln!("Error writing to JSONL: {}", e);
                        } else {
                            // Flush to ensure data is written immediately
                            if let Err(e) = jsonl_file.flush() {
                                eprintln!("Error flushing JSONL file: {}", e);
                            }
                            println!("Memory data written to memory_events.jsonl for RAG ingestion");
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to open JSONL file: {}", e);
                    }
                };
            
            // Sleep for a short time
            thread::sleep(Duration::from_millis(1000)); // Update once per second
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
}

fn main() {
    println!("Starting Self-Healing Memory System...");
    
    // Get total system memory
    let sys = System::new_all();
    let total_memory = sys.total_memory() as usize;
    println!("Total system memory: {} GB", total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    
    // Create memory manager
    let manager = Arc::new(MemoryManager::new(total_memory));
    
    // Initialize memory
    manager.init_memory();
    
    // Start memory monitoring for RAG
    monitor_memory_for_rag(Arc::clone(&manager));
    
    // Start healing action processing thread
    let manager_clone = Arc::clone(&manager);
    thread::spawn(move || {
        loop {
            // Read healing actions from Python
            let actions = read_healing_actions();
            
            // Process healing actions
            if !actions.is_empty() {
                println!("Processing {} healing actions from Python", actions.len());
                process_healing_actions(&manager_clone, actions);
            }
            
            // Sleep for a short time
            thread::sleep(Duration::from_secs(5));
        }
    });
    
    // Simulate memory allocations
    let manager_clone = Arc::clone(&manager);
    thread::spawn(move || {
        let mut allocations = Vec::new();
        
        loop {
            // Allocate memory
            if rand::random::<f32>() < 0.3 {
                let size = rand::random::<usize>() % (1024 * 1024 * 10) + 1024; // 1KB to 10MB
                if let Some(addr) = manager_clone.allocate(size) {
                    allocations.push(addr);
                    println!("Allocated {} bytes at address {}", size, addr);
                }
            }
            
            // Deallocate memory
            if rand::random::<f32>() < 0.2 && !allocations.is_empty() {
                let index = rand::random::<usize>() % allocations.len();
                let addr = allocations.swap_remove(index);
                if manager_clone.deallocate(addr) {
                    println!("Deallocated memory at address {}", addr);
                }
            }
            
            // Calculate health scores
            manager_clone.calculate_health();
            
            // Sleep for a short time
            thread::sleep(Duration::from_millis(500));
        }
    });
    
    // Main loop - keep the program running
    loop {
        // Print memory status every 10 seconds
        println!("\nMemory Status:");
        println!("{}", MemoryManager::get_memory_status(&manager));
        
        thread::sleep(Duration::from_secs(10));
    }
}
