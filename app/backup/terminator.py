import psutil
import ctypes
import time
import os
import logging
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from sentence_transformers import SentenceTransformer
import pickle
import argparse
import json
import threading
import socket
from collections import deque
import random

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS not available. Vector database functionality will be disabled.")
    FAISS_AVAILABLE = False

# Try to import RL libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    RL_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Reinforcement learning will be disabled.")
    RL_AVAILABLE = False

# Try to import Rust memory manager
try:
    import memory_core
    RUST_AVAILABLE = True
except ImportError:
    print("Rust memory manager not available. Using Python-only implementation.")
    RUST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Setup logging
log_dir = r"d:\clg\COA\Self_healing_memory\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'memory_terminator.log'), 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
VECTOR_DB_PATH = r"d:\clg\COA\Self_healing_memory\data\vector_store"
MEMORY_DATA_PATH = r"d:\clg\COA\Self_healing_memory\data\memory_data.csv"
HEALING_SUGGESTIONS_PATH = r"d:\clg\COA\Self_healing_memory\data\healer_cache\healing_suggestions.json"
CRITICAL_PROCESSES = ['svchost.exe', 'explorer.exe', 'csrss.exe', 'lsass.exe', 'winlogon.exe']

# Create directories
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MEMORY_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HEALING_SUGGESTIONS_PATH), exist_ok=True)
os.makedirs('d:\\clg\\COA\\Self_healing_memory\\logs', exist_ok=True)

# RL Agent implementation
class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MemoryDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)[0]).item()
            
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }, filepath)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint['memory']
            print(f"Loaded RL model from {filepath}")
            return True
        return False

# Memory management functions
def get_memory_stats():
    """Retrieve system memory statistics."""
    mem = psutil.virtual_memory()
    total_mb = mem.total / 1024 / 1024
    used_mb = mem.used / 1024 / 1024
    free_mb = mem.free / 1024 / 1024
    return total_mb, used_mb, free_mb

def get_high_memory_processes(threshold_mb=100):
    """Identify processes using memory above threshold."""
    high_mem_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            mem_usage_mb = proc.memory_info().rss / 1024 / 1024
            if mem_usage_mb > threshold_mb:
                high_mem_processes.append((proc.pid, proc.name(), mem_usage_mb))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return high_mem_processes

def cleanup_memory():
    """Reduce memory usage by clearing working sets of non-critical processes."""
    freed_memory = 0
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if proc.name() not in CRITICAL_PROCESSES:
                mem_before = proc.memory_info().rss
                # Use SetProcessWorkingSetSize to empty the working set
                handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, proc.pid)
                if handle:
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
                    ctypes.windll.kernel32.CloseHandle(handle)
                mem_after = proc.memory_info().rss
                freed_memory += (mem_before - mem_after) / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied, WindowsError):
            continue
    return freed_memory

def detect_memory_leaks(window=5, threshold_increase=50):
    """Detect processes with rapidly increasing memory usage."""
    memory_history = {}
    leaks = []
    for _ in range(window):
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                pid = proc.pid
                mem_mb = proc.memory_info().rss / 1024 / 1024
                if pid not in memory_history:
                    memory_history[pid] = []
                memory_history[pid].append(mem_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(1)

    for pid, mem_values in memory_history.items():
        if len(mem_values) >= 2:
            increase = mem_values[-1] - mem_values[0]
            if increase > threshold_increase:
                try:
                    proc = psutil.Process(pid)
                    leaks.append((pid, proc.name(), increase))
                except psutil.NoSuchProcess:
                    continue
    return leaks

def terminate_process(pid):
    """Safely terminate a process by PID."""
    try:
        proc = psutil.Process(pid)
        if proc.name() not in CRITICAL_PROCESSES:
            proc.terminate()
            return True
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

# FAISS vector database functions
def store_memory_data_in_vector_db(data):
    """Store memory data in FAISS vector database for later retrieval and analysis."""
    if not FAISS_AVAILABLE:
        logging.info("FAISS not available. Skipping vector storage.")
        return False
    
    try:
        # Create directory for vector DB persistence
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Define paths for saving index and metadata
        index_path = os.path.join(VECTOR_DB_PATH, "memory_faiss_index.bin")
        metadata_path = os.path.join(VECTOR_DB_PATH, "memory_metadata.pkl")
        
        # Convert data to text representation
        text_data = f"Memory Stats: Total={data['total_mb']:.1f}MB, Used={data['used_mb']:.1f}MB, Free={data['free_mb']:.1f}MB\n"
        text_data += f"Process Count: {data['process_count']}\n"
        text_data += f"Max Process Memory: {data['max_proc_mem_mb']:.1f}MB\n"
        text_data += f"High Usage: {data['high_usage']}\n"
        
        # Add process information if available
        if 'processes' in data and data['processes']:
            text_data += "Top Processes:\n"
            for proc in data['processes'][:5]:  # Top 5 processes
                text_data += f"- {proc[1]} (PID: {proc[0]}): {proc[2]:.1f}MB\n"
        
        # Add action information if available
        if 'action' in data:
            text_data += f"Action Taken: {data['action']}\n"
            text_data += f"Action Result: {data['action_result']}\n"
        
        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = model.get_sentence_embedding_dimension()
        
        # Generate embedding
        embedding = model.encode([text_data])[0].reshape(1, -1).astype('float32')
        
        # Load existing index if available
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadatas = pickle.load(f)
        else:
            index = faiss.IndexFlatL2(embedding_dim)
            metadatas = []
        
        # Add to FAISS index
        index.add(embedding)
        metadatas.append({
            'text': text_data,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data': data
        })
        
        # Save index and metadata
        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadatas, f)
        
        logging.info(f"Stored memory data in FAISS vector database with {index.ntotal} total entries")
        return True
    except Exception as e:
        logging.error(f"Error storing memory data in vector database: {e}")
        return False

def query_similar_memory_patterns(query_text, top_k=5):
    """Query FAISS for similar memory patterns based on text description."""
    if not FAISS_AVAILABLE:
        logging.info("FAISS not available. Skipping vector query.")
        return []
    
    try:
        # Define paths for index and metadata
        index_path = os.path.join(VECTOR_DB_PATH, "memory_faiss_index.bin")
        metadata_path = os.path.join(VECTOR_DB_PATH, "memory_metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logging.info("No vector database found. Skipping query.")
            return []
        
        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load index and metadata
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadatas = pickle.load(f)
        
        # Generate embedding for query
        query_embedding = model.encode([query_text]).astype('float32')
        
        # Search the index
        distances, indices = index.search(query_embedding, min(top_k, index.ntotal))
        
        # Gather results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadatas) and idx >= 0:
                results.append({
                    'text': metadatas[idx]['text'],
                    'timestamp': metadatas[idx]['timestamp'],
                    'data': metadatas[idx]['data'],
                    'distance': float(distances[0][i])
                })
        
        return results
    except Exception as e:
        logging.error(f"Error querying vector database: {e}")
        return []

# Agent communication server
class AgentCommunicationServer:
    def __init__(self, port=AGENT_SOCKET_PORT):
        self.port = port
        self.server_socket = None
        self.running = False
        self.healing_suggestions = []
        self.lock = threading.Lock()
    
    def start(self):
        """Start the agent communication server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(5)
        self.running = True
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logging.info(f"Agent communication server started on port {self.port}")
    
    def _listen_for_connections(self):
        """Listen for incoming connections from agents."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self._handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                if self.running:
                    logging.error(f"Error accepting connection: {e}")
                    time.sleep(1)
    
    def _handle_client(self, client_socket, addr):
        """Handle communication with a connected agent."""
        try:
            data = b""
            while self.running:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                # Try to parse as JSON
                try:
                    message = json.loads(data.decode('utf-8'))
                    data = b""
                    
                    # Process message
                    if message.get('type') == 'healing_suggestion':
                        with self.lock:
                            self.healing_suggestions.append({
                                'timestamp': time.time(),
                                'agent': message.get('agent', 'unknown'),
                                'suggestion': message.get('suggestion', {}),
                                'confidence': message.get('confidence', 0.0)
                            })
                            logging.info(f"Received healing suggestion from {message.get('agent', 'unknown')}")
                    
                    # Send acknowledgment
                    client_socket.sendall(json.dumps({'status': 'ok'}).encode('utf-8'))
                except json.JSONDecodeError:
                    # Incomplete data, continue receiving
                    continue
        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
    
    def get_healing_suggestions(self, max_age_seconds=60):
        """Get recent healing suggestions from agents."""
        with self.lock:
            current_time = time.time()
            # Filter out old suggestions
            self.healing_suggestions = [
                s for s in self.healing_suggestions 
                if current_time - s['timestamp'] < max_age_seconds
            ]
            return self.healing_suggestions.copy()
    
    def stop(self):
        """Stop the agent communication server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logging.info("Agent communication server stopped")

# RL Environment
class MemoryEnvironment:
    def __init__(self):
        # State: [used_memory_percent, free_memory_mb, process_count, max_proc_memory_mb, leak_count]
        self.state_size = 5
        
        # Actions: 0=do_nothing, 1=cleanup_memory, 2=terminate_top_process, 
        # 3=terminate_leaky_process, 4=defragment
        self.action_size = 5
        
        self.previous_memory_stats = None
        self.previous_action = None
        self.previous_state = None
        
        # Initialize memory stats
        total_mb, used_mb, free_mb = get_memory_stats()
        self.previous_memory_stats = {
            'total_mb': total_mb,
            'used_mb': used_mb,
            'free_mb': free_mb,
            'used_percent': (used_mb / total_mb) * 100
        }
    
    def get_state(self):
        """Get the current state of the memory environment."""
        total_mb, used_mb, free_mb = get_memory_stats()
        used_percent = (used_mb / total_mb) * 100
        
        high_mem_procs = get_high_memory_processes()
        process_count = len(high_mem_procs)
        max_proc_memory_mb = max([p[2] for p in high_mem_procs], default=0)
        
        leaks = detect_memory_leaks(window=3, threshold_increase=30)
        leak_count = len(leaks)
        
        return [used_percent, free_mb, process_count, max_proc_memory_mb, leak_count]
    
    def take_action(self, action):
        """Take an action in the environment and return reward."""
        self.previous_state = self.get_state()
        self.previous_action = action
        
        # Get initial memory stats
        total_mb_before, used_mb_before, free_mb_before = get_memory_stats()
        used_percent_before = (used_mb_before / total_mb_before) * 100
        
        action_name = "unknown"
        action_result = "no_effect"
        
        # Execute the selected action
        if action == 0:  # Do nothing
            action_name = "do_nothing"
            time.sleep(1)  # Just wait a bit
            
        elif action == 1:  # Cleanup memory
            action_name = "cleanup_memory"
            freed_mb = cleanup_memory()
            action_result = f"freed_{freed_mb:.1f}MB"
            logging.info(f"Cleaned up memory, freed {freed_mb:.1f}MB")
            
        elif action == 2:  # Terminate top memory process
            action_name = "terminate_top_process"
            high_mem_procs = get_high_memory_processes()
            if high_mem_procs:
                # Sort by memory usage (highest first)
                high_mem_procs.sort(key=lambda x: x[2], reverse=True)
                
                # Find the first non-critical process
                for pid, name, mem_usage in high_mem_procs:
                    if name not in CRITICAL_PROCESSES:
                        if terminate_process(pid):
                            action_result = f"terminated_{name}_{mem_usage:.1f}MB"
                            logging.info(f"Terminated top process {name} (PID: {pid}) using {mem_usage:.1f}MB")
                            break
            
        elif action == 3:  # Terminate leaky process
            action_name = "terminate_leaky_process"
            leaks = detect_memory_leaks()
            if leaks:
                # Sort by leak size (highest first)
                leaks.sort(key=lambda x: x[2], reverse=True)
                
                # Find the first non-critical leaky process
                for pid, name, increase in leaks:
                    if name not in CRITICAL_PROCESSES:
                        if terminate_process(pid):
                            action_result = f"terminated_leaky_{name}_{increase:.1f}MB"
                            logging.info(f"Terminated leaky process {name} (PID: {pid}) with increase of {increase:.1f}MB")
                            break
            
        elif action == 4:  # Defragment memory
            action_name = "defragment_memory"
            if RUST_AVAILABLE:
                # Use Rust memory manager for defragmentation
                try:
                    result = memory_core.defragment_memory()
                    action_result = f"defragmented_{result}"
                    logging.info(f"Defragmented memory using Rust memory manager: {result}")
                except Exception as e:
                    logging.error(f"Error defragmenting memory: {e}")
            else:
                # Fallback to Python implementation
                freed_mb = cleanup_memory()
                action_result = f"python_defrag_{freed_mb:.1f}MB"
                logging.info(f"Python defragmentation freed {freed_mb:.1f}MB")
        
        # Get memory stats after action
        total_mb_after, used_mb_after, free_mb_after = get_memory_stats()
        used_percent_after = (used_mb_after / total_mb_after) * 100
        
        # Calculate reward
        # Reward for freeing memory
        memory_freed_mb = free_mb_after - free_mb_before
        memory_percent_improved = used_percent_before - used_percent_after
        
        # Base reward on memory improvement
        reward = memory_percent_improved * 2 + memory_freed_mb / 100
        
        # Penalize for unnecessary actions
        if action > 0 and memory_freed_mb <= 0:
            reward -= 1
        
        # Extra reward for fixing leaks
        if action == 3 and "terminated_leaky" in action_result:
            reward += 3
        
        # Store action results in FAISS
        self._store_action_result(
            total_mb_after, used_mb_after, free_mb_after, 
            action_name, action_result, reward
        )
        
        # Update previous memory stats
        self.previous_memory_stats = {
            'total_mb': total_mb_after,
            'used_mb': used_mb_after,
            'free_mb': free_mb_after,
            'used_percent': used_percent_after
        }
        
        # Get new state
        new_state = self.get_state()
        
        # Check if episode is done (always False in continuous monitoring)
        done = False
        
        return new_state, reward, done
    
    def _store_action_result(self, total_mb, used_mb, free_mb, action_name, action_result, reward):
        """Store action results in FAISS for future reference."""
        high_mem_procs = get_high_memory_processes()
        process_count = len(high_mem_procs)
        max_proc_mem_mb = max([p[2] for p in high_mem_procs], default=0)
        
        data = {
            'total_mb': total_mb,
            'used_mb': used_mb,
            'free_mb': free_mb,
            'process_count': process_count,
            'max_proc_mem_mb': max_proc_mem_mb,
            'high_usage': (used_mb / total_mb) > 0.8,
            'processes': high_mem_procs,
            'action': action_name,
            'action_result': action_result,
            'reward': reward
        }
        
        store_memory_data_in_vector_db(data)

# Main function with RL integration
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Memory Terminator with RL capabilities')
    parser.add_argument('--rl-mode', action='store_true', help='Enable reinforcement learning mode')
    parser.add_argument('--use-faiss', action='store_true', help='Use FAISS for vector storage')
    parser.add_argument('--online-training', action='store_true', help='Train RL model online')
    parser.add_argument('--reward-memory-free', action='store_true', help='Reward for freeing memory')
    parser.add_argument('--reward-defrag', action='store_true', help='Reward for defragmentation')
    parser.add_argument('--ignore-healer', action='store_true', help='Ignore healer suggestions')
    args = parser.parse_args()
    
    print("=== Windows Memory Terminator with RL ===")
    logging.info("Starting memory terminator with RL capabilities")
    
    # Initialize agent communication server
    agent_server = AgentCommunicationServer()
    agent_server.start()
    
    # Initialize RL components if available and requested
    rl_agent = None
    env = None
    
    if args.rl_mode and RL_AVAILABLE:
        print("Initializing RL environment and agent...")
        env = MemoryEnvironment()
        rl_agent = MemoryDQNAgent(env.state_size, env.action_size)
        
        # Try to load existing model
        model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
        rl_agent.load(model_path)
    
    # Main monitoring loop
    try:
        episode_count = 0
        batch_size = 32
        
        while True:
            # Get memory stats
            total_mb, used_mb, free_mb = get_memory_stats()
            used_percent = (used_mb / total_mb) * 100
            
            print(f"\nMemory Stats: Total={total_mb:.1f}MB, Used={used_mb:.1f}MB ({used_percent:.1f}%), Free={free_mb:.1f}MB")
            logging.info(f"Memory: Total={total_mb:.1f}MB, Used={used_mb:.1f}MB ({used_percent:.1f}%), Free={free_mb:.1f}MB")
            
            # Get high memory processes
            high_mem_procs = get_high_memory_processes()
            process_count = len(high_mem_procs)
            max_proc_mem_mb = max([p[2] for p in high_mem_procs], default=0)
            
            # Check for memory leaks
            leaks = detect_memory_leaks()
            if leaks:
                print("Potential Memory Leaks Detected:")
                for pid, name, increase in leaks:
                    print(f" - {name} (PID: {pid}) increased by {increase:.1f}MB")
                    logging.info(f"Leak detected: {name} (PID: {pid}), +{increase:.1f}MB")
            
            # Store memory data in FAISS
            if args.use_faiss:
                vector_data = {
                    'total_mb': total_mb,
                    'used_mb': used_mb,
                    'free_mb': free_mb,
                    'process_count': process_count,
                    'max_proc_mem_mb': max_proc_mem_mb,
                    'high_usage': used_percent > 80,
                    'processes': high_mem_procs
                }
                store_memory_data_in_vector_db(vector_data)
            
            # Get healing suggestions from agents
            healing_suggestions = agent_server.get_healing_suggestions()
            if healing_suggestions:
                print(f"\nReceived {len(healing_suggestions)} healing suggestions from agents:")
                for suggestion in healing_suggestions:
                    agent_name = suggestion.get('agent', 'unknown')
                    confidence = suggestion.get('confidence', 0)
                    action = suggestion.get('suggestion', {}).get('action', 'unknown')
                    target = suggestion.get('suggestion', {}).get('target', 'unknown')
                    print(f" - {agent_name} suggests {action} on {target} (confidence: {confidence:.2f})")
            
            # Use RL agent if available
            if args.rl_mode and RL_AVAILABLE and rl_agent and env:
                # Get current state
                state = env.get_state()
                
                # Choose action
                action = rl_agent.act(state)
                
                # Take action and observe result
                next_state, reward, done = env.take_action(action)
                
                # Print action and reward
                action_names = ["do_nothing", "cleanup_memory", "terminate_top_process", 
                               "terminate_leaky_process", "defragment_memory"]
                print(f"\nRL Agent Action: {action_names[action]}, Reward: {reward:.2f}")
                
                # Remember experience
                rl_agent.remember(state, action, reward, next_state, done)
                
                # Train the model if online training is enabled
                if args.online_training and len(rl_agent.memory) > batch_size:
                    rl_agent.replay(batch_size)
                
                # Update target model periodically
                if episode_count % 10 == 0:
                    rl_agent.update_target_model()
                
                # Save model periodically
                if episode_count % 50 == 0:
                    model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
                    rl_agent.save(model_path)
                    print(f"Saved RL model to {model_path}")
                
                episode_count += 1
            else:
                # Fall back to traditional approach if RL is not available
                # Detect memory leaks and terminate them
                for pid, name, increase in leaks:
                    try:
                        proc = psutil.Process(pid)
                        if proc.name() not in CRITICAL_PROCESSES:
                            proc.terminate()
                            print(f"Terminated {name} to prevent issues")
                            logging.info(f"Terminated {name} (PID: {pid})")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Get termination candidates using FAISS-based analysis
                if used_percent > 85 or leaks:
                    termination_candidates = get_process_termination_candidates(high_mem_procs)
                    if termination_candidates:
                        print("\nSmart Process Termination Candidates:")
                        for i, candidate in enumerate(termination_candidates[:3], 1):
                            print(f"{i}. {candidate['name']} (PID: {candidate['pid']})")
                            print(f"   Memory: {candidate['memory_mb']:.1f}MB")
                            print(f"   Termination Score: {candidate['termination_score']:.2f}")
                            
                            # Terminate highest-scoring non-critical processes
                            if (candidate['termination_score'] > 3.0 and 
                                candidate['name'] not in CRITICAL_PROCESSES):
                                try:
                                    proc = psutil.Process(candidate['pid'])
                                    proc.terminate()
                                    print(f"   → Terminated based on smart analysis")
                                    logging.info(f"Smart termination: {candidate['name']} (PID: {candidate['pid']})")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    print(f"   → Could not terminate (access denied)")
                
                # Perform cleanup if memory usage is high
                if used_percent > 80 or leaks:
                    print("Cleaning up memory...")
                    freed_mb = cleanup_memory()
                    print(f"Freed {freed_mb:.1f}MB of memory")
                    logging.info(f"Freed {freed_mb:.1f}MB of memory")
            
            # Wait for next check
            print("Press Ctrl+C to exit or wait 30 seconds for next check...")
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                print("Exiting...")
                logging.info("Utility stopped")
                break
    
    finally:
        # Clean shutdown
        if agent_server:
            agent_server.stop()
        
        # Save RL model if used
        if args.rl_mode and RL_AVAILABLE and rl_agent:
            model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
            rl_agent.save(model_path)
            print(f"Saved RL model to {model_path}")

if __name__ == "__main__":
    # Ensure script runs with admin privileges for memory cleanup
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("Please run this script as Administrator for full functionality.")
        exit(1)
    
    # Import additional modules needed for RL and agent communication
    import argparse
    import json
    import threading
    import socket
    from collections import deque
    import random
    
    def read_healing_suggestions() -> Dict[str, Any]:
        """
        Read healing suggestions from the healer agent.
        
        Returns:
            Dictionary containing healing suggestions or empty dict if none available
        """
        try:
            if not os.path.exists(HEALING_SUGGESTIONS_PATH):
                return {}
            
            # Check if the file was modified in the last 5 minutes
            file_mod_time = os.path.getmtime(HEALING_SUGGESTIONS_PATH)
            if time.time() - file_mod_time > 300:  # 5 minutes
                return {}  # File is too old
            
            with open(HEALING_SUGGESTIONS_PATH, 'r') as f:
                suggestions = json.load(f)
            
            return suggestions
        except Exception as e:
            logging.error(f"Error reading healing suggestions: {e}")
            return {}
    
    def apply_healing_suggestions(suggestions: Dict[str, Any]) -> List[str]:
        """
        Apply healing suggestions from the healer agent.
        
        Args:
            suggestions: Dictionary containing healing suggestions
            
        Returns:
            List of actions taken
        """
        actions_taken = []
        
        if not suggestions or 'healing_actions' not in suggestions:
            return actions_taken
        
        for action in suggestions['healing_actions']:
            action_type = action.get('action_type')
            target = action.get('target')
            priority = action.get('priority', 'medium')
            recommendation = action.get('recommendation', '')
            
            # Skip low priority actions if memory isn't critical
            if priority == 'low':
                total_mb, used_mb, free_mb = get_memory_stats()
                used_percent = (used_mb / total_mb) * 100
                if used_percent < 80:
                    continue
            
            if action_type == 'terminate_process' and target:
                # Find processes matching the target
                matching_processes = []
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if target.lower() in proc.info['name'].lower():
                            matching_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Sort by memory usage (highest first)
                matching_processes.sort(
                    key=lambda p: p.memory_info().rss if hasattr(p, 'memory_info') else 0, 
                    reverse=True
                )
                
                # Terminate the highest memory process that matches
                for proc in matching_processes:
                    if proc.info['name'] not in CRITICAL_PROCESSES:
                        try:
                            proc_name = proc.info['name']
                            proc_pid = proc.info['pid']
                            proc.terminate()
                            actions_taken.append(f"Terminated process {proc_name} (PID: {proc_pid}) based on healer recommendation")
                            logging.info(f"Terminated process {proc_name} (PID: {proc_pid}) based on healer recommendation: {recommendation}")
                            break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            
            elif action_type == 'defragment_memory':
                # Perform memory defragmentation
                if RUST_AVAILABLE:
                    try:
                        result = memory_core.defragment_memory()
                        actions_taken.append(f"Defragmented memory using Rust memory manager: {result}")
                        logging.info(f"Defragmented memory based on healer recommendation: {recommendation}")
                    except Exception as e:
                        logging.error(f"Error defragmenting memory: {e}")
                else:
                    # Fallback to Python implementation
                    freed_mb = cleanup_memory()
                    actions_taken.append(f"Python defragmentation freed {freed_mb:.1f}MB")
                    logging.info(f"Python defragmentation freed {freed_mb:.1f}MB based on healer recommendation: {recommendation}")
            
            elif action_type == 'cleanup_memory':
                # Perform memory cleanup
                freed_mb = cleanup_memory()
                actions_taken.append(f"Cleaned up memory, freed {freed_mb:.1f}MB")
                logging.info(f"Cleaned up memory, freed {freed_mb:.1f}MB based on healer recommendation: {recommendation}")
        
        # If we took any actions, rename the suggestions file to prevent reprocessing
        if actions_taken and os.path.exists(HEALING_SUGGESTIONS_PATH):
            try:
                processed_path = HEALING_SUGGESTIONS_PATH.replace('.json', f'_processed_{int(time.time())}.json')
                os.rename(HEALING_SUGGESTIONS_PATH, processed_path)
            except Exception as e:
                logging.error(f"Error renaming processed suggestions file: {e}")
        
        return actions_taken

    # Use RL agent if available
    if args.rl_mode and RL_AVAILABLE and rl_agent and env:
        # Get current state
        state = env.get_state()
        
        # Choose action
        action = rl_agent.act(state)
        
        # Take action and observe result
        next_state, reward, done = env.take_action(action)
        
        # Print action and reward
        action_names = ["do_nothing", "cleanup_memory", "terminate_top_process", 
                       "terminate_leaky_process", "defragment_memory"]
        print(f"\nRL Agent Action: {action_names[action]}, Reward: {reward:.2f}")
        
        # Remember experience
        rl_agent.remember(state, action, reward, next_state, done)
        
        # Train the model if online training is enabled
        if args.online_training and len(rl_agent.memory) > batch_size:
            rl_agent.replay(batch_size)
        
        # Update target model periodically
        if episode_count % 10 == 0:
            rl_agent.update_target_model()
        
        # Save model periodically
        if episode_count % 50 == 0:
            model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
            rl_agent.save(model_path)
            print(f"Saved RL model to {model_path}")
        
        episode_count += 1
    else:
        # Fall back to traditional approach if RL is not available
        # Detect memory leaks and terminate them
        for pid, name, increase in leaks:
            try:
                proc = psutil.Process(pid)
                if proc.name() not in CRITICAL_PROCESSES:
                    proc.terminate()
                    print(f"Terminated {name} to prevent issues")
                    logging.info(f"Terminated {name} (PID: {pid})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            
            # Get termination candidates using FAISS-based analysis
            if used_percent > 85 or leaks:
                termination_candidates = get_process_termination_candidates(high_mem_procs)
                if termination_candidates:
                    print("\nSmart Process Termination Candidates:")
                    for i, candidate in enumerate(termination_candidates[:3], 1):
                        print(f"{i}. {candidate['name']} (PID: {candidate['pid']})")
                        print(f"   Memory: {candidate['memory_mb']:.1f}MB")
                        print(f"   Termination Score: {candidate['termination_score']:.2f}")
                        
                        # Terminate highest-scoring non-critical processes
                        if (candidate['termination_score'] > 3.0 and 
                            candidate['name'] not in CRITICAL_PROCESSES):
                            try:
                                proc = psutil.Process(candidate['pid'])
                                proc.terminate()
                                print(f"   → Terminated based on smart analysis")
                                logging.info(f"Smart termination: {candidate['name']} (PID: {candidate['pid']})")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                print(f"   → Could not terminate (access denied)")
                
            # Perform cleanup if memory usage is high
            if used_percent > 80 or leaks:
                print("Cleaning up memory...")
                freed_mb = cleanup_memory()
                print(f"Freed {freed_mb:.1f}MB of memory")
                logging.info(f"Freed {freed_mb:.1f}MB of memory")
        
            # Wait for next check
            print("Press Ctrl+C to exit or wait 30 seconds for next check...")
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                print("Exiting...")
                logging.info("Utility stopped")
                break
    
            finally:
        # Clean shutdown
                if agent_server:
                    agent_server.stop()
        
        # Save RL model if used
                if args.rl_mode and RL_AVAILABLE and rl_agent:
                    model_path = os.path.join(VECTOR_DB_PATH, "rl_model.pth")
                    rl_agent.save(model_path)
                    print(f"Saved RL model to {model_path}")

if __name__ == "__main__":
    # Ensure script runs with admin privileges for memory cleanup
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("Please run this script as Administrator for full functionality.")
        exit(1)
    
    # Import additional modules needed for RL and agent communication
    import argparse
    import json
    import threading
    import socket
    from collections import deque
    import random
    
    main()