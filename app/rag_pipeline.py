import os
import time
import threading
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
# Remove ChromaDB imports
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import faiss  # Add FAISS import
import pickle  # For saving/loading the index

# Import our ingestion module
from ingestion import MemoryLogIngestion

class MemoryRAGPipeline:
    """
    Retrieval Augmented Generation pipeline for memory logs.
    Handles embedding generation and vector database operations.
    """
    
    def __init__(self, collection_name: str = "memory_logs", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 log_path: str = "d:\\clg\\COA\\Self_healing_memory\\data\\memory_events.jsonl_0.jsonl",
                 batch_size: int = 30,
                 max_logs: int = 10000):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name of the vector database collection
            embedding_model: Name of the sentence transformer model to use
            log_path: Path to the memory log file
            batch_size: Batch size for processing logs
            max_logs: Maximum number of logs to keep in the vector database
        """
        # Set up the log ingestion system
        self.ingestion = MemoryLogIngestion(log_path)
        
        # Set up threading for real-time processing
        self.stop_event = threading.Event()
        self.batch_thread = None
        self.batch_size = batch_size
        self.max_logs = max_logs
        
        self.seen_hashes = set()
        self.embedding_queue = []
        self.embedding_lock = threading.Lock()
        
        # Set up the embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Set up FAISS instead of ChromaDB
        print("Initializing FAISS vector database...")
        # Create directory for vector DB persistence
        self.vector_db_dir = Path("d:/clg/COA/Self_healing_memory/data/vector_store")
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        # Collection-specific directory
        self.collection_dir = self.vector_db_dir / collection_name
        os.makedirs(self.collection_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.index_path = self.collection_dir / "faiss_index.bin"
        self.metadata_path = self.collection_dir / "metadata.pkl"
        self.ids_path = self.collection_dir / "ids.pkl"
        
        # Initialize or load index
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadatas = pickle.load(f)
            with open(self.ids_path, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            print("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance for similarity
            self.metadatas = []
            self.ids = []
        
        print(f"RAG pipeline initialized with collection: {collection_name}")
        
        # Start background thread for batch processing
        self.stop_event = threading.Event()
        self.batch_thread = threading.Thread(target=self._process_embedding_queue, daemon=True)
        self.batch_thread.start()
    
    def _hash_log(self, log_data: Dict[str, Any]) -> str:
        """
        Create a hash of a log entry to detect duplicates.
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            Hash string
        """
        # Use timestamp as a simple hash key
        timestamp = log_data.get('timestamp', '')
        if timestamp:
            return timestamp
        
        # If no timestamp, use a more complex hash
        log_str = json.dumps(log_data, sort_keys=True)
        return hashlib.md5(log_str.encode()).hexdigest()
    
    def is_new_log(self, log_data: Dict[str, Any]) -> bool:
        """
        Check if a log entry is new (not seen before).
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            True if the log is new, False otherwise
        """
        log_hash = self._hash_log(log_data)
        if log_hash in self.seen_hashes:
            return False
        
        self.seen_hashes.add(log_hash)
        return True
    
    def create_text_representation(self, log_data: Dict[str, Any]) -> str:
        """
        Create a text representation of a log entry for embedding.
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            Text representation of the log
        """
        text = f"Timestamp: {log_data.get('timestamp', '')}\n"
        
        # Add system metrics - ensure metrics is a dictionary
        metrics = log_data.get('system_metrics', {})
        if not isinstance(metrics, dict):
            metrics = {}  # If metrics is not a dict, create an empty one
        
        text += f"Free Memory: {metrics.get('free_memory_percent', 0):.2f}%\n"
        text += f"Fragmentation Index: {metrics.get('fragmentation_index', 0):.2f}\n"
        text += f"System Load: {metrics.get('load', 0):.2f}\n"
        
        # Add memory blocks summary - ensure blocks is a list
        blocks = log_data.get('memory_blocks', [])
        if not isinstance(blocks, list):
            blocks = []  # If blocks is not a list, create an empty one
        
        if blocks:
            allocated_blocks = sum(1 for b in blocks if isinstance(b, dict) and b.get('is_allocated', False))
            free_blocks = len(blocks) - allocated_blocks
            text += f"Memory Blocks: {len(blocks)} total, {allocated_blocks} allocated, {free_blocks} free\n"
            
            # Add health score information
            health_scores = [b.get('health_score', 0) for b in blocks if isinstance(b, dict)]
            avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
            text += f"Average Health Score: {avg_health:.2f}\n"
        
        # Add top processes - ensure processes is a list
        processes = log_data.get('processes', [])
        if not isinstance(processes, list):
            processes = []  # If processes is not a list, create an empty one
        
        if processes:
            # Filter out non-dict processes and sort
            valid_processes = [p for p in processes if isinstance(p, dict) and 'memory_kb' in p]
            top_processes = sorted(valid_processes, key=lambda p: p.get('memory_kb', 0), reverse=True)[:5]
            
            if top_processes:
                text += "Top Memory Consumers:\n"
                for proc in top_processes:
                    text += f"- {proc.get('name', 'unknown')}: {proc.get('memory_kb', 0)/1024:.2f} MB\n"
        
        return text
    
    def create_flattened_metadata(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a flattened version of the metadata that ChromaDB can handle.
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            Flattened metadata dictionary
        """
        # Create a flattened version of the metadata that ChromaDB can handle
        flattened_metadata = {
            'timestamp': log_data.get('timestamp', ''),
        }
        
        # Add system metrics - ensure metrics is a dictionary
        metrics = log_data.get('system_metrics', {})
        if not isinstance(metrics, dict):
            metrics = {}  # If metrics is not a dict, create an empty one
        
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)):
                flattened_metadata[f'system_metrics_{key}'] = value
        
        # Add summary of memory blocks - ensure blocks is a list
        blocks = log_data.get('memory_blocks', [])
        if not isinstance(blocks, list):
            blocks = []  # If blocks is not a list, create an empty one
        
        if blocks:
            valid_blocks = [b for b in blocks if isinstance(b, dict)]
            allocated_blocks = sum(1 for b in valid_blocks if b.get('is_allocated', False))
            flattened_metadata['total_blocks'] = len(valid_blocks)
            flattened_metadata['allocated_blocks'] = allocated_blocks
            flattened_metadata['free_blocks'] = len(valid_blocks) - allocated_blocks
            
            # Calculate average health score
            health_scores = [b.get('health_score', 0) for b in valid_blocks]
            flattened_metadata['avg_health_score'] = sum(health_scores) / len(health_scores) if health_scores else 0
        
        # Add summary of processes - ensure processes is a list
        processes = log_data.get('processes', [])
        if not isinstance(processes, list):
            processes = []  # If processes is not a list, create an empty one
        
        if processes:
            valid_processes = [p for p in processes if isinstance(p, dict) and 'memory_kb' in p]
            flattened_metadata['process_count'] = len(valid_processes)
            
            # Add top process info
            top_processes = sorted(valid_processes, key=lambda p: p.get('memory_kb', 0), reverse=True)[:3]
            for i, proc in enumerate(top_processes):
                flattened_metadata[f'top{i+1}_process'] = proc.get('name', 'unknown')
                flattened_metadata[f'top{i+1}_memory_mb'] = proc.get('memory_kb', 0) / 1024
        
        return flattened_metadata
    
    def add_log_to_queue(self, log_data: Dict[str, Any]) -> str:
        """
        Add a log entry to the embedding queue.
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            ID of the log entry
        """
        # Skip if we've seen this log before
        if not self.is_new_log(log_data):
            return None
        
        # Generate a unique ID based on timestamp
        timestamp = log_data.get('timestamp', '')
        log_id = f"log_{timestamp.replace(':', '_').replace('.', '_').replace('-', '_')}"
        
        # Create text representation
        text_repr = self.create_text_representation(log_data)
        
        # Create flattened metadata
        flattened_metadata = self.create_flattened_metadata(log_data)
        
        # Add to queue
        with self.embedding_lock:
            self.embedding_queue.append((log_id, text_repr, flattened_metadata))
        
        return log_id
    
    def _process_embedding_queue(self):
        """
        Background thread to process the embedding queue in batches.
        """
        while not self.stop_event.is_set():
            # Process queue if it has enough items or if it's been waiting too long
            with self.embedding_lock:
                queue_size = len(self.embedding_queue)
                
                if queue_size >= self.batch_size:
                    # Take a batch from the queue
                    batch = self.embedding_queue[:self.batch_size]
                    self.embedding_queue = self.embedding_queue[self.batch_size:]
                elif queue_size > 0 and time.time() % 10 < 1:  # Process partial batches every ~10 seconds
                    batch = self.embedding_queue
                    self.embedding_queue = []
                else:
                    batch = []
            
            # Process the batch
            if batch:
                try:
                    # Extract components
                    ids = [item[0] for item in batch]
                    texts = [item[1] for item in batch]
                    metadatas = [item[2] for item in batch]
                    
                    # Generate embeddings
                    embeddings = self.model.encode(texts)
                    
                    # Add to FAISS index
                    self.index.add(np.array(embeddings).astype('float32'))
                    
                    # Store metadata and IDs
                    self.metadatas.extend(metadatas)
                    self.ids.extend(ids)
                    
                    # Save index and metadata periodically
                    if len(self.ids) % 100 == 0:
                        self._save_index()
                    
                    print(f"Added batch of {len(batch)} log entries to vector database")
                    
                    # Prune old logs if needed
                    self._prune_old_logs()
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Put items back in queue
                    with self.embedding_lock:
                        self.embedding_queue = batch + self.embedding_queue
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadatas, f)
            with open(self.ids_path, 'wb') as f:
                pickle.dump(self.ids, f)
            print(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def _prune_old_logs(self):
        """
        Prune old logs to keep the vector store size under control.
        """
        try:
            if len(self.ids) > self.max_logs:
                # Calculate how many to remove
                to_remove = len(self.ids) - self.max_logs
                
                # Create a new index
                new_index = faiss.IndexFlatL2(self.embedding_dim)
                
                # Keep only the newest logs
                self.ids = self.ids[to_remove:]
                self.metadatas = self.metadatas[to_remove:]
                
                # We need to rebuild the index with the remaining vectors
                # This is inefficient but necessary with FAISS
                # Extract vectors from the remaining logs
                texts = [self.create_text_representation(metadata) for metadata in self.metadatas]
                embeddings = self.model.encode(texts)
                
                # Add to new index
                new_index.add(np.array(embeddings).astype('float32'))
                
                # Replace old index
                self.index = new_index
                
                # Save the pruned index
                self._save_index()
                
                print(f"Pruned {to_remove} old log entries from vector database")
        except Exception as e:
            print(f"Error pruning old logs: {e}")
    
    def add_log_to_vectordb(self, log_data: Dict[str, Any]) -> str:
        """
        Add a log entry directly to the vector database (bypassing the queue).
        
        Args:
            log_data: Log entry dictionary
            
        Returns:
            ID of the added document
        """
        # Skip if we've seen this log before
        if not self.is_new_log(log_data):
            return None
        
        # Create text representation
        text_repr = self.create_text_representation(log_data)
        
        # Generate a unique ID based on timestamp
        timestamp = log_data.get('timestamp', '')
        log_id = f"log_{timestamp.replace(':', '_').replace('.', '_').replace('-', '_')}"
        
        # Create flattened metadata
        flattened_metadata = self.create_flattened_metadata(log_data)
        
        # Generate embedding
        embedding = self.model.encode([text_repr])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Store metadata and ID
        self.metadatas.append(flattened_metadata)
        self.ids.append(log_id)
        
        return log_id
    
    def process_batch_logs(self, max_entries: Optional[int] = None) -> int:
        """
        Process a batch of logs and add them to the vector database.
        
        Args:
            max_entries: Maximum number of entries to process
            
        Returns:
            Number of entries processed
        """
        # Read logs from file directly
        logs = []
        try:
            with open(self.ingestion.log_path, 'r') as f:
                # Seek to the last position if available
                if hasattr(self.ingestion, 'last_position'):
                    f.seek(self.ingestion.last_position)
                
                # Read lines until we hit max_entries or EOF
                count = 0
                for line in f:
                    if max_entries is not None and count >= max_entries:
                        break
                    
                    try:
                        log_data = json.loads(line.strip())
                        logs.append(log_data)
                        count += 1
                    except json.JSONDecodeError:
                        print(f"Error parsing log line: {line[:50]}...")
                
                # Save the current position
                if hasattr(self.ingestion, 'last_position'):
                    self.ingestion.last_position = f.tell()
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        if not logs:
            print("No logs found to process")
            return 0
        
        # Add each log to the queue
        count = 0
        for log in logs:
            log_id = self.add_log_to_queue(log)
            if log_id:  # Only count new logs
                count += 1
            
            # Print progress every 100 entries
            if count % 100 == 0 and count > 0:
                print(f"Queued {count} entries for embedding...")
        
        print(f"Queued {count} log entries for embedding")
        return count
    
    def embed_batch(self, docs: List[str]) -> List[List[float]]:
        """
        Embed a batch of documents.
        
        Args:
            docs: List of documents to embed
            
        Returns:
            List of embeddings
        """
        return self.model.encode(docs).tolist()
    
    def start_streaming_to_vectordb(self):
        """
        Start streaming logs to the vector database in real-time.
        This function runs indefinitely until interrupted.
        """
        print("Starting real-time log streaming to vector database...")
        
        # Instead of using stream_logs, implement a polling approach
        try:
            last_position = 0
            while not self.stop_event.is_set():
                try:
                    # Read new logs from file
                    with open(self.ingestion.log_path, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        
                        if new_lines:
                            for line in new_lines:
                                try:
                                    log_entry = json.loads(line.strip())
                                    log_id = self.add_log_to_vectordb(log_entry)
                                    if log_id:
                                        print(f"Added log entry {log_id} to vector database")
                                except json.JSONDecodeError:
                                    print(f"Error parsing log line: {line[:50]}...")
                                except Exception as e:
                                    print(f"Error processing log entry: {e}")
                                
                                last_position = f.tell()
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error reading log file: {e}")
                    time.sleep(1)  # Wait longer if there's an error
        except KeyboardInterrupt:
            print("Stopped streaming logs to vector database")
        finally:
            # Stop the batch processing thread
            self.stop_event.set()
            self.batch_thread.join(timeout=2)
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database with a natural language question.
        """
        try:
            # Generate embedding for the question
            question_embedding = self.model.encode([question]).astype('float32')
            
            # Search the FAISS index
            if self.index.ntotal == 0:
                return {"ids": [], "distances": [], "metadatas": [], "documents": []}
                
            distances, indices = self.index.search(question_embedding, min(n_results, self.index.ntotal))
            
            # Gather results
            results = {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": []
            }
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.ids) and idx >= 0:
                    results["ids"].append(self.ids[idx])
                    results["distances"].append(float(distances[0][i]))
                    results["metadatas"].append(self.metadatas[idx])
                    
                    # Generate document text from metadata
                    doc_text = self.create_text_representation(self.metadatas[idx])
                    results["documents"].append(doc_text)
            
            return results
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}
    
    def get_similar_logs(self, question: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get logs similar to a natural language question.
        
        Args:
            question: Natural language question
            n_results: Number of results to return
            
        Returns:
            List of similar log entries
        """
        # Query the vector database
        results = self.query(question, n_results)
        
        # Extract the metadata (original log entries)
        similar_logs = results.get('metadatas', [])
        
        return similar_logs
    
    def detect_heavy_usage(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        # Query for logs with low free memory
        query = f"Logs with less than {(1-threshold)*100:.0f}% free memory"
        results = self.query(query, n_results=10)
        
        # Filter results by free memory threshold
        heavy_usage_logs = []
        for metadata in results.get('metadatas', []):
            if isinstance(metadata, dict):  # Ensure metadata is a dictionary
                free_memory = metadata.get('system_metrics_free_memory_percent', 100)
                if free_memory < (1-threshold) * 100:
                    heavy_usage_logs.append(metadata)
        
        return heavy_usage_logs
    
    def detect_high_fragmentation(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect logs with high memory fragmentation.
        
        Args:
            threshold: Fragmentation threshold (0.0-1.0)
            
        Returns:
            List of logs with high fragmentation
        """
        # Query for logs with high fragmentation
        query = f"Logs with high memory fragmentation"
        results = self.query(query, n_results=10)
        
        # Filter results by fragmentation threshold
        high_frag_logs = []
        for metadata in results.get('metadatas', [[]])[0]:
            if isinstance(metadata, dict):  # Ensure metadata is a dictionary
                fragmentation = metadata.get('system_metrics_fragmentation_index', 0)
                if fragmentation > threshold:
                    high_frag_logs.append(metadata)
        
        return high_frag_logs
    
    def generate_healing_actions(self, threshold_memory: float = 0.75, 
                                threshold_fragmentation: float = 0.5) -> List[Dict[str, Any]]:
        """
        Generate healing actions based on memory analysis.
        
        Args:
            threshold_memory: Memory usage threshold (0.0-1.0)
            threshold_fragmentation: Fragmentation threshold (0.0-1.0)
            
        Returns:
            List of healing actions
        """
        actions = []
        
        # Check for heavy memory usage
        heavy_usage_logs = self.detect_heavy_usage(threshold_memory)
        if heavy_usage_logs:
            # Find top memory consumers
            process_usage = {}
            for log in heavy_usage_logs:
                for i in range(1, 4):  # Check top1, top2, top3
                    proc_name = log.get(f'top{i}_process')
                    proc_memory = log.get(f'top{i}_memory_mb', 0)
                    
                    if proc_name:
                        if proc_name in process_usage:
                            process_usage[proc_name].append(proc_memory)
                        else:
                            process_usage[proc_name] = [proc_memory]
            
            # Calculate averages and sort
            if process_usage:
                avg_usage = {name: sum(values)/len(values) for name, values in process_usage.items()}
                top_processes = sorted(avg_usage.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for name, avg_mb in top_processes:
                    actions.append({
                        'action_type': 'terminate_process',
                        'target': name,
                        'reason': f'High memory usage ({avg_mb:.2f} MB)',
                        'priority': 'high' if avg_mb > 500 else 'medium'
                    })
        
        # Check for high fragmentation
        high_frag_logs = self.detect_high_fragmentation(threshold_fragmentation)
        if high_frag_logs:
            # Get average fragmentation
            avg_frag = sum(log.get('system_metrics_fragmentation_index', 0) for log in high_frag_logs) / len(high_frag_logs)
            
            actions.append({
                'action_type': 'defragment_memory',
                'reason': f'High memory fragmentation (index: {avg_frag:.2f})',
                'priority': 'high' if avg_frag > 0.7 else 'medium'
            })
            
            # Check if we need to compact memory
            if avg_frag > 0.8:
                actions.append({
                    'action_type': 'compact_memory',
                    'reason': f'Critical memory fragmentation (index: {avg_frag:.2f})',
                    'priority': 'high'
                })
        
        # Check for low health scores
        query = "Logs with low memory health scores"
        results = self.query(query, n_results=5)
        
        low_health_logs = []
        for metadata in results.get('metadatas', []):
            if isinstance(metadata, dict):  # Ensure metadata is a dictionary
                health_score = metadata.get('avg_health_score', 1.0)
                if health_score < 0.6:
                    low_health_logs.append(metadata)
        
        if low_health_logs:
            # Get average health score
            avg_health = sum(log.get('avg_health_score', 0) for log in low_health_logs) / len(low_health_logs)
            
            actions.append({
                'action_type': 'reallocate_memory',
                'reason': f'Low memory health score ({avg_health:.2f})',
                'priority': 'medium'
            })
        
        return actions
    
    def send_healing_actions_to_rust(self, actions: List[Dict[str, Any]], 
                                     action_file: str = "d:/clg/COA/2/Self_healing_memory/data/healing_actions.jsonl") -> bool:
        """
        Send healing actions to the Rust memory allocator.
        
        Args:
            actions: List of healing actions
            action_file: Path to the file where actions will be written
            
            
        Returns:
            True if actions were sent successfully, False otherwise
        """
        if not actions:
            return False
            
        try:
            # Add timestamp to each action
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for action in actions:
                action['timestamp'] = timestamp
                
                # Write action to file as JSON line
                with open(action_file, 'a') as f:
                    f.write(json.dumps(action) + '\n')
                    
            print(f"Sent {len(actions)} healing actions to Rust allocator")
            return True
            
        except Exception as e:
            print(f"Error sending healing actions: {e}")
            return False
    
    def run_healing_cycle(self, interval: int = 60):
        """
        Run a continuous healing cycle.
        
        Args:
            interval: Interval between healing cycles in seconds
        """
        print(f"Starting healing cycle (interval: {interval} seconds)")
        try:
            while not self.stop_event.is_set():
                # Generate healing actions
                actions = self.generate_healing_actions()
                
                # Send actions to Rust
                if actions:
                    self.send_healing_actions_to_rust(actions)
                    print(f"Generated {len(actions)} healing actions")
                else:
                    print("No healing actions needed")
                
                # Wait for next cycle
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Stopped healing cycle")
    
    def cleanup(self):
        """
        Clean up resources.
        """
        self.stop_event.set()
        if self.batch_thread.is_alive():
            self.batch_thread.join(timeout=2)

# Example usage
if __name__ == "__main__":
    # Initialize the RAG pipeline
    rag_pipeline = MemoryRAGPipeline(
        batch_size=16,
        max_logs=10000
    )
    
    try:
        # Process existing logs
        print("\n--- Processing existing logs ---")
        rag_pipeline.process_batch_logs()
        
        # Example query
        print("\n--- Example query ---")
        question = "What processes are using the most memory?"
        similar_logs = rag_pipeline.get_similar_logs(question)
        
        print(f"Query: {question}")
        print(f"Found {len(similar_logs)} similar logs")
        
        if similar_logs:
            print("\nTop memory consumers from similar logs:")
            
            # Extract top processes from metadata
            process_usage = {}
            for log in similar_logs:
                for i in range(1, 4):  # Check top1, top2, top3
                    proc_name = log.get(f'top{i}_process')
                    proc_memory = log.get(f'top{i}_memory_mb', 0)
                    
                    if proc_name:
                        if proc_name in process_usage:
                            process_usage[proc_name].append(proc_memory)
                        else:
                            process_usage[proc_name] = [proc_memory]
            
            # Calculate averages and sort
            avg_usage = {name: sum(values)/len(values) for name, values in process_usage.items()}
            top_processes = sorted(avg_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (name, avg_mb) in enumerate(top_processes, 1):
                print(f"{i}. {name}: {avg_mb:.2f} MB (average)")
        
        # Check for memory issues
        print("\n--- Checking for memory issues ---")
        heavy_usage = rag_pipeline.detect_heavy_usage(threshold=0.75)
        if heavy_usage:
            print(f"Found {len(heavy_usage)} logs with heavy memory usage")
            
        high_frag = rag_pipeline.detect_high_fragmentation(threshold=0.5)
        if high_frag:
            print(f"Found {len(high_frag)} logs with high memory fragmentation")
        
        # Start healing cycle in a separate thread
        healing_thread = threading.Thread(
            target=rag_pipeline.run_healing_cycle,
            args=(30,),  # Run healing cycle every 30 seconds
            daemon=True
        )
        healing_thread.start()
        print("\n--- Started healing cycle (every 30 seconds) ---")
        
        # Start streaming logs to vector database
        print("\n--- Starting real-time log streaming (press Ctrl+C to stop) ---")
        rag_pipeline.start_streaming_to_vectordb()
        
    finally:
        # Clean up resources
        rag_pipeline.cleanup()