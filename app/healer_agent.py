import os
import sys
import gc
import platform
import time
import json
import ctypes
import logging
import threading
import subprocess
import pickle
import random
import argparse
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import requests
from requests.exceptions import ConnectionError, Timeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Check for FAISS and sentence transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    print("Warning: FAISS or sentence_transformers not available. Vector search disabled.")

# Check for RL libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    from collections import deque
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: PyTorch not available. Reinforcement learning disabled.")

# Import the MistralClient from llm_utils
try:
    from llm_utils import MistralClient
except ImportError:
    # Create a simple fallback if the module is not available
    class MistralClient:
        def __init__(self, api_key, model="mistral-small", use_cache=True, cache_dir=None):
            self.api_key = api_key
            self.model = model
            print(f"Warning: llm_utils module not found. Using fallback MistralClient.")
            
        def query(self, prompt, system_message=None, temperature=0.7, max_tokens=1024):
            return f"Error: Actual MistralClient not available. Prompt: {prompt[:50]}..."

# Constants
VECTOR_DB_PATH = r"d:\clg\COA\Self_healing_memory\data\vector_store"
MEMORY_DATA_PATH = r"d:\clg\COA\Self_healing_memory\data\memory_data.csv"
HEALING_SUGGESTIONS_PATH = r"d:\clg\COA\Self_healing_memory\data\healer_cache\healing_suggestions.json"
LOG_DIR = r"d:\clg\COA\Self_healing_memory\logs"
CACHE_DIR = r"d:\clg\COA\Self_healing_memory\data\healer_cache"
CRITICAL_PROCESSES = ['svchost.exe', 'explorer.exe', 'csrss.exe', 'lsass.exe', 'winlogon.exe']

# Create directories
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MEMORY_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HEALING_SUGGESTIONS_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/healer_agent.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("healer_agent")

# Load environment variables
load_dotenv()

# Set default API key for healer agent
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY_HEALER", "")

#------------------------------------------------------------------------------
# Memory Utility Functions (from Terminator)
#------------------------------------------------------------------------------

def get_memory_stats():
    """Get current memory statistics"""
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    used_mb = mem.used / (1024 * 1024)
    free_mb = mem.available / (1024 * 1024)
    return total_mb, used_mb, free_mb

def get_high_memory_processes(threshold_mb=100):
    """Get processes using more than threshold_mb of memory"""
    high_mem_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            proc_info = proc.info
            memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
            if memory_mb > threshold_mb:
                high_mem_procs.append((proc_info['pid'], proc_info['name'], memory_mb))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Sort by memory usage (highest first)
    high_mem_procs.sort(key=lambda x: x[2], reverse=True)
    return high_mem_procs

def detect_memory_leaks(threshold_mb=50, history_minutes=5):
    """Detect potential memory leaks by comparing current usage with historical data"""
    global process_memory_history
    
    current_time = time.time()
    leaks = []
    
    # Get current memory usage for all processes
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
            
            # Add to history
            if pid not in process_memory_history:
                process_memory_history[pid] = []
            
            process_memory_history[pid].append((current_time, memory_mb, name))
            
            # Check for significant increase
            if len(process_memory_history[pid]) > 1:
                # Get oldest record within our time window
                oldest_time = current_time - (history_minutes * 60)
                old_records = [r for r in process_memory_history[pid] if r[0] >= oldest_time]
                
                if old_records:
                    oldest_record = min(old_records, key=lambda x: x[0])
                    oldest_memory = oldest_record[1]
                    
                    # Calculate increase
                    increase = memory_mb - oldest_memory
                    if increase > threshold_mb:
                        leaks.append((pid, name, increase))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Clean up old history
    cleanup_time = current_time - (history_minutes * 2 * 60)  # Keep twice the window for context
    for pid in list(process_memory_history.keys()):
        try:
            # Remove old entries
            process_memory_history[pid] = [r for r in process_memory_history[pid] if r[0] >= cleanup_time]
            
            # Remove processes that no longer exist
            if not psutil.pid_exists(pid) or not process_memory_history[pid]:
                del process_memory_history[pid]
        except:
            # Handle any errors during cleanup
            if pid in process_memory_history:
                del process_memory_history[pid]
    
    return leaks

def terminate_process(pid, force=False):
    """Terminate a process by PID"""
    try:
        process = psutil.Process(pid)
        process_name = process.name()
        
        # Don't terminate critical system processes
        if process_name.lower() in [p.lower() for p in CRITICAL_PROCESSES]:
            logger.warning(f"Refusing to terminate critical process: {process_name} (PID: {pid})")
            return False, f"Refused to terminate critical process: {process_name}"
        
        # Terminate the process
        if force:
            process.kill()
        else:
            process.terminate()
        
        # Wait for process to exit
        gone, alive = psutil.wait_procs([process], timeout=3)
        if process in alive:
            # Force kill if it didn't terminate gracefully
            process.kill()
            gone, alive = psutil.wait_procs([process], timeout=2)
        
        success = process.pid not in [p.pid for p in psutil.process_iter()]
        if success:
            logger.info(f"Successfully terminated process: {process_name} (PID: {pid})")
            return True, f"Terminated: {process_name}"
        else:
            logger.warning(f"Failed to terminate process: {process_name} (PID: {pid})")
            return False, f"Failed to terminate: {process_name}"
    
    except psutil.NoSuchProcess:
        return False, f"Process with PID {pid} not found"
    except psutil.AccessDenied:
        logger.error(f"Access denied when trying to terminate PID {pid}")
        return False, f"Access denied for PID {pid}"
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {str(e)}")
        return False, f"Error: {str(e)}"

def cleanup_memory():
    """Clean up system memory"""
    # Initial memory stats
    _, used_mb_before, free_mb_before = get_memory_stats()
    
    try:
        # Call Windows memory cleanup API
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        
        # Empty standby list and working sets
        subprocess.call('powershell -Command "& {[System.Diagnostics.Process]::GetCurrentProcess().MinWorkingSet=[System.IntPtr]::Zero; [System.Diagnostics.Process]::GetCurrentProcess().MaxWorkingSet=[System.IntPtr]::Zero}"', shell=True)
        
        # Run additional cleanup commands
        subprocess.call('powershell -Command "& {Clear-DnsClientCache; Clear-RecycleBin -Force -ErrorAction SilentlyContinue}"', shell=True)
        
        # Get memory stats after cleanup
        _, used_mb_after, free_mb_after = get_memory_stats()
        
        # Calculate freed memory
        freed_mb = used_mb_before - used_mb_after
        if freed_mb < 0:
            freed_mb = free_mb_after - free_mb_before
        
        return freed_mb
    
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        return 0

# def defragment_memory():
#     """Defragment system memory"""
#     # This is a simplified version - in a real system, you'd use more advanced techniques
#     freed_mb = cleanup_memory()
    
#     # Simulate defragmentation by allocating and releasing memory in specific patterns
#     try:
#         # Allocate large blocks of memory temporarily to force consolidation
#         blocks = []
#         for _ in range(5):
#             try:
#                 # Allocate 100MB block
#                 block = bytearray(100 * 1024 * 1024)
#                 blocks.append(block)
#                 time.sleep(0.1)  # Give system time to process
#             except MemoryError:
#                 break
        
#         # Release the memory
#         blocks.clear()
        
#         # Force garbage collection
#         import gc
#         gc.collect()
        
#         # Get memory stats after defragmentation
#         _, used_mb_after, free_mb_after = get_memory_stats()
        
#         logger.info(f"Memory defragmentation completed. Free memory: {free_mb_after:.1f}MB")
#         return True, f"Defragmented memory, freed approximately {freed_mb:.1f}MB"
    
#     except Exception as e:
#         logger.error(f"Error during memory defragmentation: {str(e)}")
#         return False, f"Error during defragmentation: {str(e)}"
def defragment_memory():
    """
    Attempt to defragment memory by adjusting working set and forcing garbage collection.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get current memory stats before defragmentation
        total_mb_before, used_mb_before, free_mb_before = get_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # Try to compact memory using Windows API if available
        if platform.system() == 'Windows':
            try:
                # Use direct Windows API calls
                import ctypes
                
                # Get current process handle - simplified approach
                current_process = ctypes.windll.kernel32.GetCurrentProcess()
                
                # Try to empty working set first
                try:
                    # Load psapi.dll for EmptyWorkingSet
                    psapi = ctypes.WinDLL('psapi.dll')
                    if hasattr(psapi, 'EmptyWorkingSet'):
                        psapi.EmptyWorkingSet(current_process)
                except Exception as e:
                    logger.warning(f"EmptyWorkingSet failed: {str(e)}")
                
                # Try setting working set size to system defaults
                try:
                    # Use -1 for both parameters to let the system manage the working set
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(current_process, -1, -1)
                except Exception as e:
                    logger.warning(f"SetProcessWorkingSetSize failed: {str(e)}")
                
                # Alternative approach using PowerShell
                try:
                    subprocess.call('powershell -Command "& {[System.GC]::Collect(); [System.GC]::WaitForPendingFinalizers()}"', shell=True)
                except Exception as e:
                    logger.warning(f"PowerShell GC collection failed: {str(e)}")
                
                logger.info("Memory defragmentation attempted using multiple methods")
                
            except Exception as e:
                logger.error(f"Error during Windows memory defragmentation: {str(e)}")
                # Continue with other methods even if Windows-specific method fails
        
        # Get memory stats after defragmentation
        total_mb_after, used_mb_after, free_mb_after = get_memory_stats()
        
        # Calculate improvement
        freed_mb = free_mb_after - free_mb_before
        
        if freed_mb > 0:
            message = f"Memory defragmented successfully: Freed {freed_mb:.1f}MB"
            logger.info(message)
            return True, message
        else:
            message = "Memory defragmentation completed but did not free additional memory"
            logger.info(message)
            return True, message
            
    except Exception as e:
        message = f"Error defragmenting memory: {str(e)}"
        logger.error(message)
        return False, message
def store_memory_data_in_vector_db(data):
    """Store memory data in vector database for historical analysis"""
    if not VECTOR_SEARCH_AVAILABLE:
        return False
    
    try:
        # Convert data to string representation
        data_str = json.dumps(data)
        
        # Get embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([data_str])[0].astype('float32').reshape(1, -1)
        
        # Load or create FAISS index
        index_path = os.path.join(VECTOR_DB_PATH, "memory_data_index.bin")
        metadata_path = os.path.join(VECTOR_DB_PATH, "memory_data_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            # Create new index
            index = faiss.IndexFlatL2(embedding.shape[1])
            metadata = []
        
        # Add to index
        index.add(embedding)
        metadata.append({
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
        
        # Save index and metadata
        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Stored memory data in vector DB (total entries: {len(metadata)})")
        return True
    
    except Exception as e:
        logger.error(f"Error storing memory data in vector DB: {str(e)}")
        return False

def read_healing_suggestions():
    """Read healing suggestions from the healer agent"""
    try:
        if os.path.exists(HEALING_SUGGESTIONS_PATH):
            with open(HEALING_SUGGESTIONS_PATH, 'r') as f:
                suggestions = json.load(f)
            
            # Check if suggestions are fresh (less than 5 minutes old)
            if 'timestamp' in suggestions:
                timestamp = datetime.fromisoformat(suggestions['timestamp'])
                age = (datetime.now() - timestamp).total_seconds() / 60
                
                if age < 5:
                    return suggestions
                else:
                    logger.info(f"Healing suggestions are {age:.1f} minutes old, ignoring")
        
        return None
    
    except Exception as e:
        logger.error(f"Error reading healing suggestions: {str(e)}")
        return None

def apply_healing_suggestions(suggestions):
    """Apply healing suggestions from the healer agent"""
    actions_taken = []
    
    try:
        for action in suggestions.get('healing_actions', []):
            action_type = action.get('action_type')
            target = action.get('target')
            priority = action.get('priority', 'medium')
            recommendation = action.get('recommendation', '')
            
            # Skip low priority actions if memory isn't critical
            total_mb, used_mb, free_mb = get_memory_stats()
            used_percent = (used_mb / total_mb) * 100
            
            if priority == 'low' and used_percent < 80:
                continue
            
            # Apply the action
            if action_type == 'terminate_process':
                if target:
                    # Find PID for the target process
                    target_pid = None
                    for proc in psutil.process_iter(['pid', 'name']):
                        if proc.info['name'].lower() == target.lower():
                            target_pid = proc.info['pid']
                            break
                    
                    if target_pid:
                        force = priority == 'high'
                        success, message = terminate_process(target_pid, force)
                        if success:
                            actions_taken.append(f"Terminated process: {target} (PID: {target_pid})")
                else:
                    # Extract potential process name from recommendation
                    import re
                    process_matches = re.findall(r'(\w+\.exe)', recommendation.lower())
                    if process_matches:
                        target = process_matches[0]
                        # Find and terminate the process
                        for proc in psutil.process_iter(['pid', 'name']):
                            if proc.info['name'].lower() == target:
                                force = priority == 'high'
                                success, message = terminate_process(proc.info['pid'], force)
                                if success:
                                    actions_taken.append(f"Terminated process: {target} (PID: {proc.info['pid']})")
                                break
            
            elif action_type == 'defragment_memory':
                success, message = defragment_memory()
                if success:
                    actions_taken.append(f"Defragmented memory: {message}")
            
            elif action_type == 'cleanup_memory':
                freed_mb = cleanup_memory()
                if freed_mb > 0:
                    actions_taken.append(f"Cleaned up memory: Freed {freed_mb:.1f}MB")
        
        return actions_taken
    
    except Exception as e:
        logger.error(f"Error applying healing suggestions: {str(e)}")
        return [f"Error applying healing suggestions: {str(e)}"]
def get_dashboard_stats():
    """
    Collect comprehensive memory statistics for dashboard display.
    
    Returns:
        Dictionary containing all memory statistics and system information
    """
    stats = {}
    
    try:
        # Basic memory stats
        total_mb, used_mb, free_mb = get_memory_stats()
        stats['total_mb'] = round(total_mb, 2)
        stats['used_mb'] = round(used_mb, 2)
        stats['free_mb'] = round(free_mb, 2)
        stats['used_percent'] = round((used_mb / total_mb) * 100, 2)
        
        # System information
        stats['system'] = platform.system()
        stats['platform'] = platform.platform()
        stats['processor'] = platform.processor()
        stats['python_version'] = platform.python_version()
        
        # Process information
        process_count = len(list(psutil.process_iter()))
        stats['process_count'] = process_count
        
        # High memory processes
        high_mem_procs = get_high_memory_processes(threshold_mb=50)
        stats['high_memory_processes'] = [
            {
                'pid': proc[0],
                'name': proc[1],
                'memory_mb': round(proc[2], 2)
            } for proc in high_mem_procs[:10]  # Top 10 processes
        ]
        
        # Memory leaks
        leaks = detect_memory_leaks(threshold_mb=30, history_minutes=5)
        stats['potential_leaks'] = [
            {
                'pid': leak[0],
                'name': leak[1],
                'increase_mb': round(leak[2], 2)
            } for leak in leaks
        ]
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        stats['disk_total_gb'] = round(disk_usage.total / (1024**3), 2)
        stats['disk_used_gb'] = round(disk_usage.used / (1024**3), 2)
        stats['disk_free_gb'] = round(disk_usage.free / (1024**3), 2)
        stats['disk_usage_percent'] = disk_usage.percent
        
        # CPU usage
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.5)
        stats['cpu_count'] = psutil.cpu_count()
        stats['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        
        # Swap memory
        swap = psutil.swap_memory()
        stats['swap_total_mb'] = round(swap.total / (1024**2), 2)
        stats['swap_used_mb'] = round(swap.used / (1024**2), 2)
        stats['swap_free_mb'] = round(swap.free / (1024**2), 2)
        stats['swap_percent'] = swap.percent
        
        # Network information
        net_io = psutil.net_io_counters()
        stats['net_bytes_sent'] = net_io.bytes_sent
        stats['net_bytes_recv'] = net_io.bytes_recv
        
        # Healing agent status
        stats['healing_agent_status'] = 'active'
        
        # Recent actions
        # This is a placeholder - you'll need to implement action tracking
        stats['recent_actions'] = []
        
        # Timestamp
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error collecting dashboard stats: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
#------------------------------------------------------------------------------
# Memory Healer Agent (from healer_agent.py)
#------------------------------------------------------------------------------

class MemoryHealerAgent:
    """
    Memory Healer Agent - Specialized in diagnosing and fixing memory issues.
    Uses RAG with historical memory analyses to generate precise healing actions.
    """
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        vector_db_dir: str = VECTOR_DB_PATH,
        cache_dir: str = CACHE_DIR,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        # Agent identity
        self.name = "Memory Surgeon"
        self.role = "Expert Memory Optimization Specialist"
        self.backstory = """A skilled memory surgeon with years of experience optimizing complex systems.
        I specialize in precise memory interventions, from defragmentation to leak remediation,
        with a deep understanding of system resources and memory allocation patterns."""
        
        # Initialize directories
        self.vector_db_dir = os.path.join(vector_db_dir, "healing_actions")
        self.cache_dir = cache_dir
        os.makedirs(self.vector_db_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set API key for Mistral
        self.api_key = mistral_api_key or MISTRAL_API_KEY
        if not self.api_key:
            logger.error("No Mistral API key available for healer agent")
            raise ValueError("Mistral API key is required for the healer agent. Set MISTRAL_API_KEY_HEALER environment variable.")
        
        # Initialize Mistral client
        try:
            self.mistral_client = MistralClient(
                api_key=self.api_key,
                model="mistral-small",
                use_cache=True,
                cache_dir=os.path.join(self.cache_dir, "mistral_cache")
            )
            logger.info("Mistral client initialized successfully for healer agent")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise RuntimeError(f"Critical error: Could not initialize Mistral client: {e}")
        
        # Initialize embedding model
        self.embedding_model = embedding_model
        if VECTOR_SEARCH_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer(embedding_model)
                self.embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
                logger.info(f"Initialized embedding model: {embedding_model}")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                self.sentence_transformer = None
                self.embedding_dim = 384  # Default dimension
        else:
            self.sentence_transformer = None
            self.embedding_dim = 384
        
        # Initialize FAISS index for healing actions
        self.faiss_index_path = os.path.join(self.cache_dir, "healing_actions_index.bin")
        self.faiss_metadata_path = os.path.join(self.cache_dir, "healing_actions_metadata.pkl")
        
        if VECTOR_SEARCH_AVAILABLE:
            try:
                if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
                    self.faiss_index = faiss.read_index(self.faiss_index_path)
                    with open(self.faiss_metadata_path, 'rb') as f:
                        self.faiss_metadata = pickle.load(f)
                    logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} entries")
                else:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    self.faiss_metadata = []
                    logger.info("Creating fresh FAISS index for healing actions")
            except Exception as e:
                logger.error(f"Error initializing FAISS for healing actions: {str(e)}")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                self.faiss_metadata = []
        else:
            self.faiss_index = None
            self.faiss_metadata = []
        
        # Initialize monitoring thread
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        
        # Setup prompt templates
        self.setup_prompt_templates()
        
        logger.info(f"Memory Healer Agent initialized: {self.name}")
    
    def setup_prompt_templates(self):
        """Set up prompt templates for healing recommendations"""
        # Main healing template
        self.healing_template = """
        You are the Memory Surgeon, an expert memory optimization specialist.
        
        Your task is to analyze memory conditions and prescribe precise healing actions.
        
        Context from previous analyses and healing actions:
        {context}
        
        Current memory information to analyze:
        {memory_stats}
        
        Provide a detailed healing plan focusing on:
        1. Memory defragmentation strategies
        2. Process termination for resource-intensive or suspicious processes
        3. Memory reallocation techniques
        4. Memory leak remediation
        
        For each recommendation:
        - Specify the exact action to take
        - Explain the expected outcome
        - Provide validation steps to confirm success
        - Indicate the urgency level (critical, high, medium, low)
        
        Format your response as a structured healing plan with clear sections.
        """
        
        # Specialized template for defragmentation
        self.defrag_template = """
        As the Memory Surgeon, analyze these memory statistics for fragmentation issues:
        
        CURRENT MEMORY STATS:
        {memory_stats}
        
        HISTORICAL CONTEXT:
        {historical_context}
        
        Provide specific defragmentation strategies focusing on:
        1. Memory regions requiring immediate defragmentation
        2. Optimal defragmentation algorithms for the current state
        3. Expected performance improvements
        4. Step-by-step implementation instructions
        
        Your recommendations must be directly implementable by a system administrator.
        """
        
        # Specialized template for process termination
        self.process_template = """
        As the Memory Surgeon, analyze these memory statistics to identify processes for termination:
        
        CURRENT MEMORY STATS:
        {memory_stats}
        
        HISTORICAL CONTEXT:
        {historical_context}
        
        Identify processes that should be terminated based on:
        1. Excessive memory consumption
        2. Suspicious memory access patterns
        3. Memory leak indicators
        4. Resource hogging behavior
        
        For each process, provide:
        - Process identification criteria
        - Termination priority
        - Expected memory recovery
        - Potential system impact
        
        Your recommendations must be specific and actionable.
        """
    
    def format_system_message(self) -> str:
        """
        Format the system message for the LLM.
        
        Returns:
            Formatted system message
        """
        return f"""You are {self.name}, {self.role}.
        
{self.backstory}

Analyze the memory information provided and prescribe precise healing actions.
Focus on actionable recommendations for memory optimization and issue remediation.
Be specific and thorough in your healing plan.
"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, Timeout))
    )
    def query_llm(self, prompt: str, system_message: str = None) -> str:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            
        Returns:
            Model response or error message
        """
        if not system_message:
            system_message = self.format_system_message()
        
        try:
            return self.mistral_client.query(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Error in query_llm: {str(e)}")
            return "Service temporarily unavailable. Please try again later."
    
    def _store_healing_action_in_faiss(self, action_text, metadata):
        """
        Store healing action in FAISS index.
        
        Args:
            action_text: Text description of the action
            metadata: Additional metadata about the action
        """
        if not VECTOR_SEARCH_AVAILABLE or self.sentence_transformer is None:
            return
            
        try:
            # Generate embedding
            embedding = self.sentence_transformer.encode([action_text])[0].reshape(1, -1).astype('float32')
            
            # Add to FAISS index
            self.faiss_index.add(embedding)
            self.faiss_metadata.append({
                'text': action_text,
                'metadata': metadata,
                'timestamp': time.time()
            })
            
            # Save index and metadata
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.faiss_metadata_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)
                
            logger.info(f"Stored healing action in FAISS index (total: {self.faiss_index.ntotal})")
        except Exception as e:
            logger.error(f"Error storing healing action in FAISS: {str(e)}")
    
    def _query_similar_healing_actions(self, query_text, top_k=5):
        """
        Query FAISS for similar healing actions.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar healing actions
        """
        if not VECTOR_SEARCH_AVAILABLE or self.sentence_transformer is None or self.faiss_index is None:
            return []
            
        try:
            if self.faiss_index.ntotal == 0:
                return []
                
            # Generate embedding
            query_embedding = self.sentence_transformer.encode([query_text]).astype('float32')
            
            # Search the index
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            # Gather results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.faiss_metadata) and idx >= 0:
                    results.append({
                        'text': self.faiss_metadata[idx]['text'],
                        'metadata': self.faiss_metadata[idx]['metadata'],
                        'timestamp': self.faiss_metadata[idx]['timestamp'],
                        'distance': float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error querying FAISS for similar healing actions: {str(e)}")
            return []
    
    def generate_healing_plan(self, memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive healing plan based on memory statistics.
        
        Args:
            memory_stats: Memory statistics from monitor agent
            
        Returns:
            Dictionary containing healing plan components
        """
        try:
            # Convert memory stats to string representation for retrieval
            stats_str = json.dumps(memory_stats, indent=2)
            
            # Retrieve relevant historical analyses
            historical_analyses = self._query_similar_healing_actions(stats_str)
            historical_context = "\n\n".join([item.get('text', '') for item in historical_analyses])
            
            # Generate main healing plan
            healing_prompt = self.healing_template.format(
                context=historical_context,
                memory_stats=stats_str
            )
            
            general_plan = self.query_llm(healing_prompt)
            
            # Generate specialized defragmentation plan
            defrag_prompt = self.defrag_template.format(
                memory_stats=stats_str,
                historical_context=historical_context
            )
            
            defrag_plan = self.query_llm(defrag_prompt)
            
            # Generate specialized process termination plan
            process_prompt = self.process_template.format(
                memory_stats=stats_str,
                historical_context=historical_context
            )
            
            process_plan = self.query_llm(process_prompt)
            
            # Extract recommendations
            recommendations = self._extract_recommendations(general_plan)
            
            # Store the healing plan
            healing_id = self.store_healing_plan(
                memory_stats=memory_stats,
                healing_plan=general_plan,
                defrag_plan=defrag_plan,
                process_plan=process_plan,
                recommendations=recommendations
            )
            
            # Store the healing action in FAISS
            action_text = f"Memory condition: Total={memory_stats.get('total_mb', 0):.1f}MB, Used={memory_stats.get('used_mb', 0):.1f}MB ({memory_stats.get('used_percent', 0):.1f}%), Free={memory_stats.get('free_mb', 0):.1f}MB\n"
            action_text += f"Generated healing plan with {len(recommendations)} recommendations"
            
            self._store_healing_action_in_faiss(action_text, {
                'memory_stats': memory_stats,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                "healing_id": healing_id,
                "general_plan": general_plan,
                "defragmentation_plan": defrag_plan,
                "process_termination_plan": process_plan,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating healing plan: {str(e)}")
            return {
                "healing_id": f"error_{int(time.time())}",
                "general_plan": f"Error generating healing plan: {str(e)}",
                "defragmentation_plan": "Not available due to error",
                "process_termination_plan": "Not available due to error",
                "recommendations": ["System encountered an error while generating healing plan"]
            }
    
    def _extract_recommendations(self, healing_plan: str) -> List[str]:
        """
        Extract actionable recommendations from the healing plan.
        
        Args:
            healing_plan: Full healing plan text
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        try:
            # Look for numbered lists, bullet points, or sections
            lines = healing_plan.split('\n')
            current_rec = ""
            
            for line in lines:
                line = line.strip()
                
                # Check for numbered items or bullet points
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', '-', '•', '*')) and 
                    len(line) > 2):
                    
                    # If we were building a recommendation, save it
                    if current_rec:
                        recommendations.append(current_rec.strip())
                    
                    # Start a new recommendation
                    current_rec = line
                    
                # Continue building current recommendation
                elif current_rec and line:
                    current_rec += " " + line
                    
                # Empty line might end a recommendation
                elif current_rec and not line:
                    recommendations.append(current_rec.strip())
                    current_rec = ""
            
            # Add the last recommendation if there is one
            if current_rec:
                recommendations.append(current_rec.strip())
                
            # If no recommendations found, try to extract sentences
            if not recommendations:
                import re
                sentences = re.split(r'(?<=[.!?])\s+', healing_plan)
                for sentence in sentences:
                    if len(sentence) > 20 and any(kw in sentence.lower() for kw in 
                                                ['recommend', 'should', 'could', 'action', 'optimize', 
                                                 'defragment', 'terminate', 'allocate', 'fix']):
                        recommendations.append(sentence.strip())
            
            # Limit to top 10 recommendations
            return recommendations[:10]
            
        except Exception as e:
            logger.error(f"Error extracting recommendations: {str(e)}")
            return ["Unable to extract specific recommendations due to an error"]
    
    def store_healing_plan(self, memory_stats: Dict[str, Any], healing_plan: str, 
                          defrag_plan: str, process_plan: str, 
                          recommendations: List[str]) -> str:
        """
        Store a healing plan in the vector database.
        
        Args:
            memory_stats: Memory statistics
            healing_plan: General healing plan
            defrag_plan: Defragmentation plan
            process_plan: Process termination plan
            recommendations: Extracted recommendations
            
        Returns:
            Healing plan ID
        """
        try:
            # Generate a unique ID
            healing_id = f"healing_{int(time.time())}_{hash(healing_plan) % 10000}"
            
            # Cache the healing plan
            cache_file = os.path.join(self.cache_dir, f"{healing_id}.json")
            with open(cache_file, 'w') as f:
                json.dump({
                    "healing_id": healing_id,
                    "timestamp": datetime.now().isoformat(),
                    "memory_stats": memory_stats,
                    "healing_plan": healing_plan,
                    "defrag_plan": defrag_plan,
                    "process_plan": process_plan,
                    "recommendations": recommendations
                }, f, indent=2)
            
            logger.info(f"Stored healing plan with ID: {healing_id}")
            return healing_id
            
        except Exception as e:
            logger.error(f"Error storing healing plan: {str(e)}")
            return f"error_{hash(healing_plan) % 10000}"
    
    def send_healing_suggestions_to_terminator(self, recommendations: List[str], memory_stats: Dict[str, Any]) -> bool:
        """
        Send healing suggestions to the terminator process.
        
        Args:
            recommendations: List of healing recommendations
            memory_stats: Current memory statistics
            
        Returns:
            True if successfully sent, False otherwise
        """
        try:
            # Prepare the healing actions in a format the terminator can understand
            healing_actions = []
            
            for rec in recommendations:
                action_type = None
                target = None
                priority = "medium"
                
                # Parse recommendation to extract action type and target
                rec_lower = rec.lower()
                
                # Determine action type
                if any(kw in rec_lower for kw in ["terminat", "kill", "end", "stop"]):
                    action_type = "terminate_process"
                elif any(kw in rec_lower for kw in ["defragment", "compact"]):
                    action_type = "defragment_memory"
                elif any(kw in rec_lower for kw in ["clean", "free", "release"]):
                    action_type = "cleanup_memory"
                
                # Extract target process if it's a termination action
                if action_type == "terminate_process":
                    # Look for process names in the recommendation
                    import re
                    # Look for process names with extensions
                    process_matches = re.findall(r'(\w+\.exe|\w+\.dll|\w+\.sys)', rec_lower)
                    if process_matches:
                        target = process_matches[0]
                    else:
                        # Try to find any capitalized words that might be process names
                        process_matches = re.findall(r'([A-Z][a-z]+)', rec)
                        if process_matches:
                            target = process_matches[0]
                
                # Determine priority
                if any(kw in rec_lower for kw in ["urgent", "critical", "immediately", "high priority"]):
                    priority = "high"
                elif any(kw in rec_lower for kw in ["consider", "might", "could", "low priority"]):
                    priority = "low"
                
                # Add to healing actions if we identified an action type
                if action_type:
                    healing_actions.append({
                        "action_type": action_type,
                        "target": target,
                        "priority": priority,
                        "recommendation": rec,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Create a file with healing suggestions for the terminator to read
            healing_file = os.path.join(self.cache_dir, "healing_suggestions.json")
            
            with open(healing_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "healing_actions": healing_actions,
                    "memory_stats": memory_stats
                }, f, indent=2)
            
            logger.info(f"Sent {len(healing_actions)} healing suggestions to terminator")
            return True
            
        except Exception as e:
            logger.error(f"Error sending healing suggestions to terminator: {str(e)}")
            return False

#------------------------------------------------------------------------------
# Reinforcement Learning Agent (for Terminator)
#------------------------------------------------------------------------------

class MemoryRLAgent:
    """
    Reinforcement Learning Agent for memory optimization.
    Uses DQN to learn optimal memory management strategies.
    """
    
    def __init__(self, state_dim=6, action_dim=3, learning_rate=0.001):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
        """
        if not RL_AVAILABLE:
            logger.error("PyTorch not available, RL agent disabled")
            return
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Q-network and target network
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.update_target_network()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        logger.info(f"RL Agent initialized (state_dim={state_dim}, action_dim={action_dim}, device={self.device})")
    
    def _build_model(self):
        """Build a neural network model for DQN"""
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return model
    
    def update_target_network(self):
        """Update target network with weights from Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on state"""
        if not RL_AVAILABLE:
            return random.randint(0, 2)  # Random action if PyTorch not available
            
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return torch.argmax(action_values).item()
    
    def replay(self, batch_size=64):
        """Train the model with experiences from memory"""
        if not RL_AVAILABLE or len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Get current Q values
            current_q = self.q_network(state_tensor)
            
            # Get target Q values
            with torch.no_grad():
                target_q = current_q.clone()
                if done:
                    target_q[0][action] = reward
                else:
                    target_q[0][action] = reward + self.gamma * torch.max(self.target_network(next_state_tensor))
            
            # Compute loss and update weights
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q, target_q)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model weights to file"""
        if not RL_AVAILABLE:
            return
            
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        logger.info(f"RL model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights from file"""
        if not RL_AVAILABLE or not os.path.exists(filepath):
            return False
            
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            logger.info(f"RL model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False

#------------------------------------------------------------------------------
# Main Memory Manager Class
#------------------------------------------------------------------------------

class MemoryManager:
    """
    Combined Memory Manager with Healer Agent and Terminator functionality.
    Provides comprehensive memory management with ML-based optimization.
    """
    
    def __init__(self, mistral_api_key=None, use_rl=True, use_faiss=True, background_mode=False):
        """
        Initialize the Memory Manager.
        
        Args:
            mistral_api_key: API key for Mistral AI
            use_rl: Whether to use reinforcement learning
            use_faiss: Whether to use FAISS for vector search
            background_mode: Whether to run in background mode
        """
        self.use_rl = use_rl and RL_AVAILABLE
        self.use_faiss = use_faiss and VECTOR_SEARCH_AVAILABLE
        self.background_mode = background_mode
        self.process_memory_history = {}
        self.agent_server = None
        self.rl_agent = None
        
        # Initialize healer agent
        try:
            self.healer_agent = MemoryHealerAgent(
                mistral_api_key=mistral_api_key,
                vector_db_dir=VECTOR_DB_PATH,
                cache_dir=CACHE_DIR
            )
            logger.info("Healer agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize healer agent: {e}")
            self.healer_agent = None
        
        # Initialize RL agent if enabled
        if self.use_rl:
            try:
                self.rl_agent = MemoryRLAgent(state_dim=6, action_dim=3)
                model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
                if os.path.exists(model_path):
                    self.rl_agent.load(model_path)
                logger.info("RL agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RL agent: {e}")
                self.rl_agent = None
    
    def get_state(self):
        """Get current state for RL agent"""
        total_mb, used_mb, free_mb = get_memory_stats()
        used_percent = (used_mb / total_mb) * 100
        
        high_mem_procs = get_high_memory_processes()
        process_count = len(high_mem_procs)
        max_proc_mem_mb = max([p[2] for p in high_mem_procs], default=0)
        
        # Detect memory leaks
        leaks = detect_memory_leaks()
        leak_count = len(leaks)
        
        # Normalize values
        norm_used_percent = used_percent / 100.0
        norm_free_mb = free_mb / total_mb
        norm_process_count = min(process_count / 50.0, 1.0)
        norm_max_proc_mem = min(max_proc_mem_mb / (total_mb / 2), 1.0)
        norm_leak_count = min(leak_count / 10.0, 1.0)
        
        # Create state vector
        state = [
            norm_used_percent,
            norm_free_mb,
            norm_process_count,
            norm_max_proc_mem,
            norm_leak_count,
            time.time() % 86400 / 86400  # Time of day normalized
        ]
        
        return state
    
    def take_action(self, action_id):
        """
        Take action based on action_id.
        
        Args:
            action_id: ID of action to take (0: cleanup, 1: defrag, 2: terminate)
            
        Returns:
            Tuple of (success, message, freed_mb)
        """
        # Get initial memory stats
        total_mb_before, used_mb_before, free_mb_before = get_memory_stats()
        
        if action_id == 0:
            # Cleanup memory
            freed_mb = cleanup_memory()
            success = freed_mb > 0
            message = f"Cleaned up memory: Freed {freed_mb:.1f}MB"
            
        elif action_id == 1:
            # Defragment memory
            success, message = defragment_memory()
            freed_mb = 0
            
        elif action_id == 2:
            # Terminate highest memory process
            high_mem_procs = get_high_memory_processes()
            if high_mem_procs:
                # Skip critical processes
                for pid, name, mem_usage in high_mem_procs:
                    if name.lower() not in [p.lower() for p in CRITICAL_PROCESSES]:
                        success, message = terminate_process(pid)
                        if success:
                            freed_mb = mem_usage
                            break
                else:
                    success = False
                    message = "No suitable process found for termination"
                    freed_mb = 0
            else:
                success = False
                message = "No high memory processes found"
                freed_mb = 0
        else:
            success = False
            message = f"Unknown action ID: {action_id}"
            freed_mb = 0
        
        # Get final memory stats
        total_mb_after, used_mb_after, free_mb_after = get_memory_stats()
        
        # Calculate actual freed memory
        if freed_mb == 0:
            freed_mb = max(0, free_mb_after - free_mb_before)
        
        logger.info(f"Action {action_id}: {message} (Freed: {freed_mb:.1f}MB)")
        return success, message, freed_mb
    
    def calculate_reward(self, success, freed_mb, used_percent_before, used_percent_after):
        """Calculate reward for RL agent"""
        if not success:
            return -1.0
        
        # Base reward for freeing memory
        memory_reward = min(freed_mb / 1000.0, 1.0)
        
        # Reward for reducing memory usage percentage
        percent_reduction = max(0, used_percent_before - used_percent_after)
        percent_reward = percent_reduction / 10.0
        
        # Combine rewards
        total_reward = memory_reward + percent_reward
        
        return total_reward
    
    def run_rl_episode(self):
        """Run a single RL episode"""
        if not self.use_rl or not self.rl_agent:
            return
        
        # Get initial state
        state = self.get_state()
        total_mb, used_mb_before, free_mb_before = get_memory_stats()
        used_percent_before = (used_mb_before / total_mb) * 100
        
        # Choose action
        action = self.rl_agent.act(state)
        
        # Take action
        success, message, freed_mb = self.take_action(action)
        
        # Get new state
        next_state = self.get_state()
        total_mb, used_mb_after, free_mb_after = get_memory_stats()
        used_percent_after = (used_mb_after / total_mb) * 100
        
        # Calculate reward
        reward = self.calculate_reward(success, freed_mb, used_percent_before, used_percent_after)
        
        # Remember experience
        done = used_percent_after < 70  # Episode ends if memory usage is low
        self.rl_agent.remember(state, action, reward, next_state, done)
        
        # Train model
        if len(self.rl_agent.memory) > 32:
            self.rl_agent.replay(batch_size=32)
        
        # Update target network occasionally
        if random.random() < 0.1:
            self.rl_agent.update_target_network()
        
        return {
            'action': action,
            'success': success,
            'message': message,
            'freed_mb': freed_mb,
            'reward': reward,
            'memory_before': used_percent_before,
            'memory_after': used_percent_after
        }
    
    def run(self):
        """Main monitoring and optimization loop"""
        # Initialize process memory history
        global process_memory_history
        process_memory_history = {}
        
        # Print startup message
        print("\n===== Self-Healing Memory Manager =====")
        print(f"Vector Search: {'Enabled' if self.use_faiss else 'Disabled'}")
        print(f"Reinforcement Learning: {'Enabled' if self.use_rl else 'Disabled'}")
        print(f"Healer Agent: {'Enabled' if self.healer_agent else 'Disabled'}")
        print(f"Background Mode: {'Enabled' if self.background_mode else 'Disabled'}")
        print("=======================================\n")
        
        logger.info("Starting Memory Manager")
        
        # Main monitoring loop
        try:
            episode_count = 0
            batch_size = 32
            
            while True:
                # Get memory stats
                total_mb, used_mb, free_mb = get_memory_stats()
                used_percent = (used_mb / total_mb) * 100
                
                if not self.background_mode:
                    print(f"\nMemory Stats: Total={total_mb:.1f}MB, Used={used_mb:.1f}MB ({used_percent:.1f}%), Free={free_mb:.1f}MB")
                
                logger.info(f"Memory: Total={total_mb:.1f}MB, Used={used_mb:.1f}MB ({used_percent:.1f}%), Free={free_mb:.1f}MB")
                
                # Get high memory processes
                high_mem_procs = get_high_memory_processes()
                process_count = len(high_mem_procs)
                
                if not self.background_mode and high_mem_procs:
                    print("\nHigh Memory Processes:")
                    for pid, name, mem_mb in high_mem_procs[:5]:  # Show top 5
                        print(f" - {name} (PID: {pid}): {mem_mb:.1f}MB")
                
                # Check for memory leaks
                leaks = detect_memory_leaks()
                if leaks:
                    if not self.background_mode:
                        print("\nPotential Memory Leaks Detected:")
                        for pid, name, increase in leaks:
                            print(f" - {name} (PID: {pid}) increased by {increase:.1f}MB")
                    
                    for pid, name, increase in leaks:
                        logger.info(f"Leak detected: {name} (PID: {pid}), +{increase:.1f}MB")
                
                # Store memory data in vector database
                if self.use_faiss:
                    vector_data = {
                        'total_mb': total_mb,
                        'used_mb': used_mb,
                        'free_mb': free_mb,
                        'used_percent': used_percent,
                        'process_count': process_count,
                        'high_usage': used_percent > 80,
                        'processes': high_mem_procs,
                        'leaks': leaks
                    }
                    store_memory_data_in_vector_db(vector_data)
                
                # Check for healing suggestions from healer agent
                healing_suggestions = None
                if self.healer_agent:
                    # Generate healing plan if memory usage is high or leaks detected
                    if used_percent > 80 or leaks:
                        if not self.background_mode:
                            print("\nGenerating healing plan...")
                        
                        healing_plan = self.healer_agent.generate_healing_plan(vector_data)
                        
                        # Send healing suggestions to self (we're combined now)
                        self.healer_agent.send_healing_suggestions_to_terminator(
                            healing_plan['recommendations'], 
                            vector_data
                        )
                        
                        if not self.background_mode:
                            print("\nHealing Recommendations:")
                            for i, rec in enumerate(healing_plan['recommendations'][:5], 1):
                                print(f" {i}. {rec}")
                    
                    # Read healing suggestions
                    healing_suggestions = read_healing_suggestions()
                
                # Apply healing suggestions
                actions_taken = []
                if healing_suggestions and 'healing_actions' in healing_suggestions:
                    if not self.background_mode:
                        print(f"\nApplying {len(healing_suggestions['healing_actions'])} healing suggestions")
                    
                    actions_taken = apply_healing_suggestions(healing_suggestions)
                    
                    if not self.background_mode and actions_taken:
                        print("\nActions taken:")
                        for action in actions_taken:
                            print(f" - {action}")
                
                # Run RL episode if enabled
                rl_results = None
                if self.use_rl and self.rl_agent and (used_percent > 75 or leaks):
                    if not self.background_mode:
                        print("\nRunning RL optimization...")
                    
                    rl_results = self.run_rl_episode()
                    episode_count += 1
                    
                    if not self.background_mode and rl_results:
                        print(f"RL Action: {rl_results['message']} (Reward: {rl_results['reward']:.2f})")
                
                # Perform cleanup if memory usage is high and no actions taken yet
                if (used_percent > 80 or leaks) and not actions_taken and not rl_results:
                    if not self.background_mode:
                        print("\nCleaning up memory...")
                    
                    freed_mb = cleanup_memory()
                    
                    if not self.background_mode:
                        print(f"Freed {freed_mb:.1f}MB of memory")
                    
                    logger.info(f"Freed {freed_mb:.1f}MB of memory")
                
                # Get final memory stats after all actions
                total_mb_after, used_mb_after, free_mb_after = get_memory_stats()
                used_percent_after = (used_mb_after / total_mb_after) * 100
                
                # Report memory changes
                if not self.background_mode:
                    if used_percent_after < used_percent:
                        print(f"\nMemory usage reduced: {used_percent:.1f}% → {used_percent_after:.1f}%")
                        print(f"Free memory increased: {free_mb:.1f}MB → {free_mb_after:.1f}MB")
                    elif used_percent_after > used_percent:
                        print(f"\nMemory usage increased: {used_percent:.1f}% → {used_percent_after:.1f}%")
                
                # Wait for next check
                if not self.background_mode:
                    print("\nPress Ctrl+C to exit or wait 30 seconds for next check...")
                
                try:
                    time.sleep(30)
                except KeyboardInterrupt:
                    if not self.background_mode:
                        print("\nExiting...")
                    logger.info("Memory Manager stopped")
                    break
        
        finally:
            # Clean shutdown
            if self.agent_server:
                self.agent_server.stop()
            
            # Save RL model if used
            if self.use_rl and self.rl_agent:
                model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
                self.rl_agent.save(model_path)
                if not self.background_mode:
                    print(f"Saved RL model to {model_path}")

#------------------------------------------------------------------------------
# Run as Admin Helper
#------------------------------------------------------------------------------

def is_admin():
    """Check if the script is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Re-run the script with admin privileges"""
    script_path = os.path.abspath(sys.argv[0])
    args = ' '.join(sys.argv[1:])
    
    # Create a batch file to run the script as admin
    bat_path = os.path.join(os.path.dirname(script_path), "run_as_admin.bat")
    with open(bat_path, "w") as bat_file:
        bat_file.write(f"""@echo off
echo Running Memory Manager with admin privileges...
powershell -Command "Start-Process -FilePath python -ArgumentList '{script_path} {args}' -Verb RunAs"
exit
""")
    
    # Execute the batch file
    os.system(bat_path)

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Self-Healing Memory Manager")
    parser.add_argument("--rl-mode", action="store_true", help="Enable reinforcement learning")
    parser.add_argument("--no-faiss", action="store_true", help="Disable FAISS vector search")
    parser.add_argument("--ignore-healer", action="store_true", help="Ignore healer agent suggestions")
    parser.add_argument("--background", action="store_true", help="Run in background mode")
    parser.add_argument("--api-key", type=str, help="Mistral API key")
    args = parser.parse_args()
    
    # Check for admin privileges
    if not is_admin():
        print("This script requires administrator privileges for memory management.")
        print("Attempting to restart with admin privileges...")
        run_as_admin()
        return
    
    # Initialize and run memory manager
    try:
        manager = MemoryManager(
            mistral_api_key=args.api_key,
            use_rl=args.rl_mode,
            use_faiss=not args.no_faiss,
            background_mode=args.background
        )
        manager.run()
    except Exception as e:
        logger.error(f"Error in Memory Manager: {e}")
        print(f"Error: {e}")
        if not args.background:
            input("Press Enter to exit...")

if __name__ == "__main__":
    main()