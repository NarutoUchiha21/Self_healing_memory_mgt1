use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize)]  // Remove Deserialize
pub struct MemoryBlock {
    pub start_address: usize,
    pub size: usize,
    pub is_allocated: bool,
    #[serde(skip)]
    pub last_accessed: Instant,
    pub health_score: f64,
}

// Add a separate struct for deserialization
#[derive(Deserialize)]
pub struct MemoryBlockDeserialize {
    pub start_address: usize,
    pub size: usize,
    pub is_allocated: bool,
    pub health_score: f64,
}

// Implement conversion from deserialized to regular MemoryBlock
impl From<MemoryBlockDeserialize> for MemoryBlock {
    fn from(block: MemoryBlockDeserialize) -> Self {
        MemoryBlock {
            start_address: block.start_address,
            size: block.size,
            is_allocated: block.is_allocated,
            last_accessed: Instant::now(),
            health_score: block.health_score,
        }
    }
}

#[derive(Debug)]
pub struct MemoryManager {
    blocks: Arc<Mutex<Vec<MemoryBlock>>>,
    allocations: Arc<Mutex<HashMap<usize, usize>>>,
    pub total_memory: usize,
    health_threshold: f64,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ProcessMemoryInfo {
    pub name: String,
    pub memory_usage: u64,
    pub is_essential: bool,
}

impl MemoryManager {
    pub fn new(total_memory: usize) -> Self {
        let initial_block = MemoryBlock {
            start_address: 0,
            size: total_memory,
            is_allocated: false,
            last_accessed: Instant::now(),
            health_score: 1.0,
        };

        MemoryManager {
            blocks: Arc::new(Mutex::new(vec![initial_block])),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_memory,
            health_threshold: 0.7,
        }
    }

    pub fn allocate(&self, size: usize) -> Option<usize> {
        let mut blocks = self.blocks.lock().unwrap();
        let mut found_block = None;
        let mut block_index = 0;

        for (i, block) in blocks.iter().enumerate() {
            if !block.is_allocated && block.size >= size {
                found_block = Some(block.clone());
                block_index = i;
                break;
            }
        }

        if found_block.is_none() {
            drop(blocks);
            if self.heal_memory() {
                return self.allocate(size);
            }
            return None;
        }

        let block = found_block.unwrap();
        let start_address = block.start_address;

        if block.size > size {
            blocks[block_index].size = size;
            blocks[block_index].is_allocated = true;
            
            let new_block = MemoryBlock {
                start_address: block.start_address + size,
                size: block.size - size,
                is_allocated: false,
                last_accessed: Instant::now(),
                health_score: 1.0,
            };
            blocks.push(new_block);
        } else {
            blocks[block_index].is_allocated = true;
        }

        let mut allocations = self.allocations.lock().unwrap();
        allocations.insert(start_address, size);

        Some(start_address)
    }

    pub fn deallocate(&self, address: usize) -> bool {
        let mut blocks = self.blocks.lock().unwrap();
        let mut allocations = self.allocations.lock().unwrap();
    
        for i in 0..blocks.len() {
            if blocks[i].start_address == address {
                blocks[i].is_allocated = false;
                allocations.remove(&address);
                
                self.merge_free_blocks(&mut blocks);
                return true;
            }
        }
    
        false
    }

    fn merge_free_blocks(&self, blocks: &mut Vec<MemoryBlock>) {
        let mut i = 0;
        while i < blocks.len() - 1 {
            if !blocks[i].is_allocated && !blocks[i + 1].is_allocated {
                blocks[i].size += blocks[i + 1].size;
                blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    // Change from private to public
    pub fn heal_memory(&self) -> bool {
        let mut blocks = self.blocks.lock().unwrap();
        let mut healed = false;

        for block in blocks.iter_mut() {
            if block.health_score < self.health_threshold {
                if self.analyze_and_heal_block(block) {
                    healed = true;
                }
            }
        }

        if healed {
            self.compact_memory(&mut blocks);
        }

        healed
    }

    fn analyze_and_heal_block(&self, block: &mut MemoryBlock) -> bool {
        let age = block.last_accessed.elapsed();
        let health_factor = if age > Duration::from_secs(60) {
            0.8
        } else {
            1.0
        };

        block.health_score = health_factor;

        if block.health_score < self.health_threshold {
            block.last_accessed = Instant::now();
            block.health_score = 1.0;
            true
        } else {
            false
        }
    }

    fn compact_memory(&self, blocks: &mut Vec<MemoryBlock>) {
        blocks.sort_by(|a, b| a.start_address.cmp(&b.start_address));
        self.merge_free_blocks(blocks);

        let mut new_blocks = Vec::new();
        let mut current_address = 0;

        for block in blocks.iter() {
            if !block.is_allocated {
                continue;
            }

            if block.start_address != current_address {
                let mut new_block = block.clone();
                new_block.start_address = current_address;
                new_blocks.push(new_block);
            } else {
                new_blocks.push(block.clone());
            }

            current_address += block.size;
        }

        if current_address < self.total_memory {
            new_blocks.push(MemoryBlock {
                start_address: current_address,
                size: self.total_memory - current_address,
                is_allocated: false,
                last_accessed: Instant::now(),
                health_score: 1.0,
            });
        }

        *blocks = new_blocks;
    }

    pub fn get_memory_status(manager: &Arc<Self>) -> String {
        let blocks = manager.blocks.lock().unwrap();
        let mut status = String::new();

        for block in blocks.iter() {
            status.push_str(&format!(
                "Block: {} bytes at 0x{:X} - {} - Health: {:.2}\n",
                block.size,
                block.start_address,
                if block.is_allocated { "Allocated" } else { "Free" },
                block.health_score
            ));
        }

        status
    }

    pub fn init_memory(&self) {
        let mut blocks = self.blocks.lock().unwrap();
        *blocks = vec![MemoryBlock {
            start_address: 0,
            size: self.total_memory,
            is_allocated: false,
            last_accessed: Instant::now(),
            health_score: 1.0,
        }];
    }

    #[allow(dead_code)]
    pub fn free_block(&self, index: usize) -> bool {
        let mut blocks = self.blocks.lock().unwrap();
        if index < blocks.len() {
            blocks[index].is_allocated = false;
            blocks[index].last_accessed = Instant::now();
            true
        } else {
            false
        }
    }

    pub fn defragment(&self) {
        let mut blocks = self.blocks.lock().unwrap();
        self.merge_free_blocks(&mut blocks);
        self.compact_memory(&mut blocks);
    }

    pub fn get_blocks(&self) -> Vec<MemoryBlock> {
        let blocks = self.blocks.lock().unwrap();
        blocks.clone()
    }

    pub fn calculate_health(&self) {
        let mut blocks = self.blocks.lock().unwrap();
        for block in blocks.iter_mut() {
            let age = block.last_accessed.elapsed().as_secs_f64();
            let size_factor = (block.size as f64) / (self.total_memory as f64);
            block.health_score = 0.7 * size_factor + 0.3 * (1.0 - (age / 3600.0).min(1.0));
        }
    }

    #[allow(dead_code)]
    pub fn log_memory_metrics(&self, process_info: &[ProcessMemoryInfo], system_load: f64) -> serde_json::Value {
        let blocks = self.blocks.lock().unwrap();
        let timestamp = chrono::Local::now().to_rfc3339();
        
        // Calculate fragmentation index
        let free_blocks = blocks.iter().filter(|b| !b.is_allocated).count();
        let total_blocks = blocks.len();
        let fragmentation_index = if total_blocks > 0 {
            free_blocks as f64 / total_blocks as f64
        } else {
            0.0
        };
        
        // Calculate overall memory health
        let avg_health = if blocks.is_empty() {
            1.0
        } else {
            blocks.iter().map(|b| b.health_score).sum::<f64>() / blocks.len() as f64
        };
        
        // Create process metrics
        let processes = process_info.iter().map(|proc| {
            serde_json::json!({
                "name": proc.name,
                "pid": proc.name.split_whitespace().last().unwrap_or("unknown"),
                "memory_usage_kb": proc.memory_usage / 1024,
                "is_essential": proc.is_essential,
                "priority": if proc.is_essential { "high" } else { "normal" }
            })
        }).collect::<Vec<_>>();
        
        // Create the comprehensive log entry
        serde_json::json!({
            "timestamp": timestamp,
            "system_metrics": {
                "total_memory": self.total_memory,
                "used_memory": blocks.iter().filter(|b| b.is_allocated).map(|b| b.size).sum::<usize>(),
                "free_memory": blocks.iter().filter(|b| !b.is_allocated).map(|b| b.size).sum::<usize>(),
                "system_load": system_load,
                "fragmentation_index": fragmentation_index,
                "memory_health_score": avg_health,
                "log_sequence": self.get_log_sequence_number()
            },
            "memory_blocks": blocks.iter().map(|block| {
                serde_json::json!({
                    "address": format!("0x{:X}", block.start_address),
                    "size_bytes": block.size,
                    "status": if block.is_allocated { "allocated" } else { "free" },
                    "health_score": block.health_score
                })
            }).collect::<Vec<_>>(),
            "processes": processes
        })
    }
    
    // Add a method to get and increment a log sequence number
    #[allow(dead_code)]
    fn get_log_sequence_number(&self) -> u64 {
        static mut LOG_SEQUENCE: u64 = 0;
        static mut LOG_FILE_COUNT: u64 = 1;
        
        unsafe {
            LOG_SEQUENCE += 1;
            
            // If we reach 60,000 logs, increment file count and reset sequence
            if LOG_SEQUENCE >= 60000 {
                LOG_FILE_COUNT += 1;
                LOG_SEQUENCE = 1;
            }
            
            (LOG_FILE_COUNT << 32) | LOG_SEQUENCE
        }
    }
    
    // Add a method to write logs with rotation
    pub fn write_memory_log(&self, log_data: serde_json::Value, base_path: &str) -> std::io::Result<()> {
        use std::fs::{self, OpenOptions};
        use std::io::Write;
        
        // Create data directory if it doesn't exist
        fs::create_dir_all("d:\\clg\\COA\\2\\Self_healing_memory\\data")?;
        
        // Get the sequence number to determine which file to use
        let sequence = log_data["system_metrics"]["log_sequence"].as_u64().unwrap_or(0);
        let file_num = (sequence >> 32) as u64;
        
        // Create filename with rotation
        let file_path = format!("{}_{}.jsonl", base_path, file_num);
        
        // Open file for appending
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;
            
        // Write the log entry
        writeln!(file, "{}", log_data.to_string())?;
        
        Ok(())
    }
}

