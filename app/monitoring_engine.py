import os
import time
import threading
import json
from pathlib import Path
# import Self_healing_memory

# This will import your Rust module after building with maturin
try:
    import memory_core
except ImportError:
    print("Memory core module not found. Please build it with 'maturin develop' first.")
    print("Install maturin with: pip install maturin")
    exit(1)

# Path to the memory events log file
MEMORY_EVENTS_PATH = Path("D:/clg/COA/Self_healing_memory/data/memory_events.jsonl")
HEALING_ACTIONS_PATH = Path("D:/clg/COA/Self_healing_memory/data/healing_actions.jsonl")

# Ensure directories exist
os.makedirs(os.path.dirname(MEMORY_EVENTS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HEALING_ACTIONS_PATH), exist_ok=True)

from datetime import datetime
import psutil

def get_memory_stats():
    """
    Returns memory statistics for monitoring.
    Uses psutil to get system memory information.
    
    Returns:
        dict: Dictionary containing memory statistics
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "free_memory_percent": memory.percent,
                "fragmentation_index": 0.0,  # Placeholder for fragmentation index
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free
            }
        }
    except ImportError:
        print("Warning: psutil not installed. Install with: pip install psutil")
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "free_memory_percent": 0,
                "fragmentation_index": 0
            },
            "note": "No memory data available - psutil not installed"
        }
    except Exception as e:
        print(f"Error getting memory stats: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "free_memory_percent": 0,
                "fragmentation_index": 0
            },
            "error": str(e)
        }

class LogWatcher:
    def __init__(self, log_path, actions_path):
        self.log_path = log_path
        self.actions_path = actions_path
        self.last_position = 0
        self.running = False
    
    def start(self):
        """Start watching the log file for new entries"""
        self.running = True
        self.watch_thread = threading.Thread(target=self._watch_logs)
        self.watch_thread.daemon = True
        self.watch_thread.start()
        
    def _watch_logs(self):
        """Watch the log file and process new entries"""
        print(f"Watching log file: {self.log_path}")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Create the file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                pass
        
        while self.running:
            try:
                with open(self.log_path, 'r') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()
                    
                    if new_lines:
                        for line in new_lines:
                            self.process_log_entry(line.strip())
                        
                        self.last_position = f.tell()
                
                time.sleep(0.1)  # Check for new logs every 100ms
            except Exception as e:
                print(f"Error watching logs: {e}")
                time.sleep(1)  # Wait a bit longer if there's an error
    
    def process_log_entry(self, line):
        """Process a new log entry and generate healing actions if needed"""
        try:
            data = json.loads(line)
            
            # Extract key metrics
            timestamp = data.get('timestamp', '')
            free_memory = data.get('system_metrics', {}).get('free_memory_percent', 0)
            fragmentation = data.get('system_metrics', {}).get('fragmentation_index', 0)
            
            # Get top memory consumers
            processes = data.get('processes', [])
            top_processes = sorted(processes, key=lambda p: p.get('memory_kb', 0), reverse=True)[:5]
            
            # Print summary
            print(f"\n--- Memory Update ({timestamp}) ---")
            print(f"Free Memory: {free_memory:.2f}%")
            print(f"Fragmentation Index: {fragmentation:.2f}")
            
            if top_processes:
                print("\nTop Memory Consumers:")
                for i, proc in enumerate(top_processes, 1):
                    print(f"{i}. {proc.get('name')} - {proc.get('memory_kb')/1024:.2f} MB")
            
            # Generate healing actions based on memory state
            healing_actions = []
            
            # Check for low memory
            if free_memory < 20:
                # Find top memory consumers
                for proc in top_processes[:3]:  # Top 3 memory consumers
                    healing_actions.append({
                        "action_type": "terminate_process",
                        "target": proc.get('name'),
                        "reason": f"High memory usage ({proc.get('memory_kb', 0)/1024:.2f} MB)",
                        "priority": "high",
                        "timestamp": timestamp
                    })
                
                # Add memory reallocation action
                healing_actions.append({
                    "action_type": "reallocate_memory",
                    "reason": f"Low memory health score ({free_memory/100:.2f})",
                    "priority": "medium",
                    "timestamp": timestamp
                })
            
            # Check for high fragmentation
            if fragmentation > 0.7:
                healing_actions.append({
                    "action_type": "defragment_memory",
                    "reason": f"High memory fragmentation ({fragmentation:.2f})",
                    "priority": "medium",
                    "timestamp": timestamp
                })
            
            # Write healing actions to file
            if healing_actions:
                print(f"\nGenerated {len(healing_actions)} healing actions")
                try:
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(self.actions_path), exist_ok=True)
                    
                    # Write with explicit encoding and flush
                    with open(self.actions_path, 'a', encoding='utf-8') as f:
                        for action in healing_actions:
                            f.write(json.dumps(action) + '\n')
                            f.flush()  # Force write to disk
                    
                    print(f"Healing actions written to file: {self.actions_path}")
                except Exception as e:
                    print(f"Error writing healing actions to file: {e}")
            
        except json.JSONDecodeError:
            print(f"Invalid JSON in log: {line[:50]}...")
        except Exception as e:
            print(f"Error processing log entry: {e}")

def generate_sample_log():
    """Generate a sample log entry for testing"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    log_entry = {
        "timestamp": timestamp,
        "system_metrics": {
            "free_memory_percent": 15.5,
            "fragmentation_index": 0.75,
            "load": 3.2
        },
        "processes": [
            {"name": "chrome.exe", "memory_kb": 1500000, "cpu_usage": 5.2},
            {"name": "firefox.exe", "memory_kb": 800000, "cpu_usage": 3.1},
            {"name": "python.exe", "memory_kb": 400000, "cpu_usage": 10.5},
            {"name": "vscode.exe", "memory_kb": 350000, "cpu_usage": 2.0},
            {"name": "explorer.exe", "memory_kb": 200000, "cpu_usage": 0.5}
        ],
        "memory_blocks": [
            {"start_address": 1000, "size": 4096, "is_allocated": True, "health_score": 0.8},
            {"start_address": 5096, "size": 8192, "is_allocated": False, "health_score": 1.0},
            {"start_address": 13288, "size": 2048, "is_allocated": True, "health_score": 0.5}
        ]
    }
    
    return log_entry

def write_sample_logs():
    """Write sample logs to the file for testing"""
    print("Writing sample logs to file...")
    
    with open(MEMORY_EVENTS_PATH, 'a') as f:
        # Write 5 sample logs
        for _ in range(5):
            log_entry = generate_sample_log()
            f.write(json.dumps(log_entry) + '\n')
            time.sleep(0.5)  # Small delay between logs
    
    print(f"Sample logs written to {MEMORY_EVENTS_PATH}")

# Move the import and class definition before they're used
from ingestion import MemoryLogIngestion

# Move this class definition up in the file, before test_log_writing and main functions
class MemoryMonitoringEngine:
    def __init__(self, log_path=None):
        if log_path is None:
            log_path = "D:/clg/COA/Self_healing_memory/data/memory_events.jsonl"
        self.log_path = Path(log_path)
        
        # Ensure directory and file exist
        os.makedirs(self.log_path.parent, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w', encoding='utf-8') as f:
                pass
            print(f"Created new log file at: {self.log_path}")
            
        # Initialize the log ingestion system
        try:
            self.ingestion = MemoryLogIngestion(str(self.log_path))
            # Start background ingestion thread
            self.ingestion_thread = self.ingestion.start_streaming_in_background()
        except Exception as e:
            print(f"Warning: Could not initialize log ingestion: {e}")
            self.ingestion = None
            self.ingestion_thread = None
    
    def log_memory_state(self, memory_state):
        """
        Log the current memory state to a file and ingest it into the vector database.
        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),  # Use ISO format for consistency
            "memory_blocks": memory_state.get("memory_blocks", []),
            "memory_status": memory_state.get("memory_status", ""),
            "processes": memory_state.get("processes", []),
            "system_metrics": {
                "free_memory_percent": memory_state.get("free_memory_percent", 0),
                "fragmentation_index": memory_state.get("fragmentation_index", 0),
                "load": memory_state.get("system_load", 0)
            }
        }
        
        # Write the log entry to the file
        try:
            # Convert Path object to string to avoid any issues
            log_path_str = str(self.log_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_path_str), exist_ok=True)
            
            # Open file with explicit encoding and write mode
            with open(log_path_str, 'a', encoding='utf-8') as f:
                json_str = json.dumps(log_entry)
                f.write(json_str + '\n')
                f.flush()  # Force write to disk
                
                print(f"Successfully logged memory state at {log_entry['timestamp']}")
                print(f"Log file location: {log_path_str}")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
            # Create the directory and try again if it failed
            try:
                os.makedirs(os.path.dirname(log_path_str), exist_ok=True)
                with open(log_path_str, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    print(f"Retry successful: logged memory state")
            except Exception as retry_e:
                print(f"Retry failed: {str(retry_e)}")

def test_log_writing():
    """Test function to verify log writing works"""
    engine = MemoryMonitoringEngine()
    test_data = {
        "free_memory_percent": 75.5,
        "fragmentation_index": 0.25,
        "system_load": 2.1,
        "processes": [
            {"name": "test.exe", "memory_kb": 50000, "cpu_usage": 1.5}
        ],
        "memory_blocks": [
            {"start_address": 1000, "size": 4096, "is_allocated": True, "health_score": 0.9}
        ]
    }
    engine.log_memory_state(test_data)
    print(f"Test complete. Check file at {engine.log_path}")

def main():
    print("🚀 Starting Self-Healing Memory System")
    
    # Create the log files if they don't exist
    for path in [MEMORY_EVENTS_PATH, HEALING_ACTIONS_PATH]:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass
            print(f"Created log file: {path}")
    
    # Test log writing directly to verify it works
    test_log_writing()
    
    # Generate sample logs immediately to ensure we have data
    write_sample_logs()
    
    # Start the memory monitoring in Rust
    print("\n🔧 Initializing memory monitoring...")
    try:
        # Check if the directory exists and is writable
        log_dir = os.path.dirname(MEMORY_EVENTS_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created directory: {log_dir}")
        
        # Test if we can write to the log file
        with open(MEMORY_EVENTS_PATH, 'a') as test_file:
            test_file.write("")
        print(f"Successfully verified write access to: {MEMORY_EVENTS_PATH}")
        
        # Start the memory monitoring with explicit path
        try:
            # Try with path parameter first
            memory_core.start_monitoring(str(MEMORY_EVENTS_PATH))
        except TypeError:
            # If that fails, try without parameters (in case the Rust function doesn't accept parameters)
            print("Rust function doesn't accept path parameter, using default path")
            memory_core.start_monitoring()
        
        print("Memory monitoring started successfully")
    except Exception as e:
        print(f"Error starting memory monitoring: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Will continue with sample logs only")
        write_sample_logs()
    
    # Start the log watcher
    print("\n📊 Starting log watcher...")
    watcher = LogWatcher(MEMORY_EVENTS_PATH, HEALING_ACTIONS_PATH)
    watcher.start()
    
    # Start a thread to periodically check log files and generate more sample logs if needed
    def check_and_generate_logs():
        while True:
            time.sleep(10)  # Check every 10 seconds
            try:
                # Check memory events file
                if os.path.exists(MEMORY_EVENTS_PATH):
                    size = os.path.getsize(MEMORY_EVENTS_PATH)
                    print(f"Memory events file size: {size} bytes")
                    
                    # If file is empty or very small, generate more sample logs
                    if size < 100:
                        print("Log file is too small, generating more sample logs...")
                        write_sample_logs()
                else:
                    print("Memory events file does not exist! Creating it...")
                    with open(MEMORY_EVENTS_PATH, 'w') as f:
                        pass
                    write_sample_logs()
                
                # Check healing actions file
                if os.path.exists(HEALING_ACTIONS_PATH):
                    size = os.path.getsize(HEALING_ACTIONS_PATH)
                    print(f"Healing actions file size: {size} bytes")
                else:
                    print("Healing actions file does not exist! Creating it...")
                    with open(HEALING_ACTIONS_PATH, 'w') as f:
                        pass
            except Exception as e:
                print(f"Error checking log files: {e}")
    
    check_thread = threading.Thread(target=check_and_generate_logs)
    check_thread.daemon = True
    check_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⚠️ Shutting down...")

if __name__ == "__main__":
    main()