import os
import time
import threading
import json
import hashlib
import os
import time
import json
import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle

# Fix the warning filter - the previous one had a syntax error with the backtick
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS not available. Vector database functionality will be disabled.")
    FAISS_AVAILABLE = False

class MemoryLogIngestion:
    """
    Memory log ingestion system.
    Handles reading logs from file and adding them to a vector database.
    """
    
    def __init__(self, log_path: str = "d:\\clg\\COA\\Self_healing_memory\\data\\memory_events.jsonl",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db_path: str = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store"):
        """
        Initialize the ingestion system.
        
        Args:
            log_path: Path to the memory log file
            embedding_model: Name of the sentence transformer model to use
            vector_db_path: Path to the vector database
        """
        self.log_path = log_path
        self.last_position = 0
        self.last_timestamp = None
        self.processed_hashes = set()
        self.vector_db_path = vector_db_path
        
        # Set up threading for real-time processing
        self.stop_event = threading.Event()
        
        # Set up the embedding model
        print(f"Loading embedding model: {embedding_model}")
        # Updated to use the model without triggering the resume_download warning
        try:
            from huggingface_hub import snapshot_download
            # First ensure the model is downloaded
            snapshot_download(repo_id=f"sentence-transformers/{embedding_model}")
            # Then load it
            self.model = SentenceTransformer(embedding_model)
        except ImportError:
            # Fall back to direct loading if huggingface_hub is not available
            self.model = SentenceTransformer(embedding_model)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Set up FAISS if available
        self.index = None
        self.metadatas = []
        self.documents = []
        self.ids = []
        
        if FAISS_AVAILABLE:
            try:
                # Create directory for vector DB persistence
                os.makedirs(vector_db_path, exist_ok=True)
                
                # Define paths for saving index and metadata
                self.index_path = os.path.join(vector_db_path, "faiss_index.bin")
                self.metadata_path = os.path.join(vector_db_path, "metadata.pkl")
                self.documents_path = os.path.join(vector_db_path, "documents.pkl")
                self.ids_path = os.path.join(vector_db_path, "ids.pkl")
                
                # Load existing index if available
                if os.path.exists(self.index_path):
                    print(f"Loading existing FAISS index from: {self.index_path}")
                    self.index = faiss.read_index(self.index_path)
                    
                    with open(self.metadata_path, 'rb') as f:
                        self.metadatas = pickle.load(f)
                    with open(self.documents_path, 'rb') as f:
                        self.documents = pickle.load(f)
                    with open(self.ids_path, 'rb') as f:
                        self.ids = pickle.load(f)
                    
                    print(f"Loaded index with {len(self.ids)} entries")
                else:
                    print(f"Creating new FAISS index at: {vector_db_path}")
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                
                print(f"Connected to vector database at: {vector_db_path}")
            except Exception as e:
                print(f"Error setting up vector database: {e}")
                self.index = None
    
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
        if log_hash in self.processed_hashes:
            return False
        
        self.processed_hashes.add(log_hash)
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
            # Handle case where metrics is not a dictionary
            metrics = {}
        
        text += f"Free Memory: {metrics.get('free_memory_percent', 0):.2f}%\n"
        text += f"Fragmentation Index: {metrics.get('fragmentation_index', 0):.2f}\n"
        text += f"System Load: {metrics.get('load', 0):.2f}\n"
        
        # Add memory blocks summary - ensure blocks is a list
        blocks = log_data.get('memory_blocks', [])
        if blocks and isinstance(blocks, list):
            allocated_blocks = sum(1 for b in blocks if b.get('is_allocated', False))
            free_blocks = len(blocks) - allocated_blocks
            text += f"Memory Blocks: {len(blocks)} total, {allocated_blocks} allocated, {free_blocks} free\n"
            
            # Add health score information
            health_scores = [b.get('health_score', 0) for b in blocks]
            avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
            text += f"Average Health Score: {avg_health:.2f}\n"
        
        # Add top processes - ensure processes is a list
        processes = log_data.get('processes', [])
        if processes and isinstance(processes, list):
            top_processes = sorted(processes, key=lambda p: p.get('memory_kb', 0), reverse=True)[:5]
            text += "Top Memory Consumers:\n"
            for proc in top_processes:
                text += f"- {proc.get('name')}: {proc.get('memory_kb', 0)/1024:.2f} MB\n"
        
        return text
    
    def add_to_vector_db(self, log_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Add a log entry to the vector database.
        
        Args:
            log_id: Unique ID for the log entry
            text: Text representation of the log entry
            embedding: Embedding vector for the text
            metadata: Metadata for the log entry
            
        Returns:
            True if the log was added successfully, False otherwise
        """
        if not FAISS_AVAILABLE or self.index is None:
            return False
        
        try:
            # Add to FAISS index
            self.index.add(np.array([embedding]).astype('float32'))
            
            # Store metadata, document, and ID
            self.metadatas.append(metadata)
            self.documents.append(text)
            self.ids.append(log_id)
            
            # Save periodically (every 100 entries)
            if len(self.ids) % 100 == 0:
                self._save_index()
                
            return True
        except Exception as e:
            print(f"Error adding log to vector database: {e}")
            return False
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk"""
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadatas, f)
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.ids_path, 'wb') as f:
                pickle.dump(self.ids, f)
            print(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def read_latest_logs(self) -> pd.DataFrame:
        """
        Read only the latest logs from the file using pandas.
        
        Returns:
            DataFrame containing the latest logs
        """
        try:
            # Read the JSONL file into a pandas DataFrame
            df = pd.read_json(self.log_path, lines=True)
            
            # Filter by timestamp if we have a last timestamp:
            if self.last_timestamp and 'timestamp' in df.columns:
                df = df[df['timestamp'] > self.last_timestamp]
                
            # Update last timestamp if we have new logs
            if not df.empty and 'timestamp' in df.columns:
                self.last_timestamp = df['timestamp'].max()
                
            return df
        except Exception as e:
            print(f"Error reading logs with pandas: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def filter_high_usage_processes(self, df: pd.DataFrame, threshold_mb: float = 100.0) -> List[Dict[str, Any]]:
        """
        Filter processes with high memory usage from the logs.
        
        Args:
            df: DataFrame containing logs
            threshold_mb: Memory usage threshold in MB
            
        Returns:
            List of high-usage processes with their details
        """
        high_usage_processes = []
        
        try:
            # Iterate through each row
            for _, row in df.iterrows():
                processes = row.get('processes', [])
                timestamp = row.get('timestamp', '')
                
                # Ensure processes is a list
                if not isinstance(processes, list):
                    continue
                    
                # Find high-usage processes
                for proc in processes:
                    if not isinstance(proc, dict):
                        continue
                        
                    memory_kb = proc.get('memory_kb', 0)
                    if not isinstance(memory_kb, (int, float)):
                        continue
                        
                    memory_mb = memory_kb / 1024
                    if memory_mb > threshold_mb:
                        high_usage_processes.append({
                            'timestamp': timestamp,
                            'process_name': proc.get('name', 'unknown'),
                            'memory_mb': memory_mb,
                            'pid': proc.get('pid', 0)
                        })
            
            return high_usage_processes
        except Exception as e:
            print(f"Error filtering high-usage processes: {e}")
            return high_usage_processes
    
    def process_logs_with_pandas(self, max_logs: int = 1000, threshold_mb: float = 100.0) -> int:
        """
        Process logs using pandas for efficient reading and filtering.
        
        Args:
            max_logs: Maximum number of logs to process
            threshold_mb: Memory usage threshold in MB for filtering high-usage processes
            
        Returns:
            Number of logs processed
        """
        processed_count = 0
        
        try:
            # Read latest logs
            df = self.read_latest_logs()
            
            if df.empty:
                return 0
                
            # Find high-usage processes
            high_usage = self.filter_high_usage_processes(df, threshold_mb)
            if high_usage:
                print(f"Found {len(high_usage)} high-usage processes")
                for proc in high_usage[:5]:  # Show top 5
                    print(f"  - {proc['process_name']}: {proc['memory_mb']:.2f} MB")
            
            # Process logs in batches
            batch_size = 100
            logs_to_process = df.head(max_logs).to_dict('records')
            
            for i in range(0, len(logs_to_process), batch_size):
                batch = logs_to_process[i:i+batch_size]
                
                # Create batches for vector DB
                batch_texts = []
                batch_ids = []
                batch_metadata = []
                
                for log_data in batch:
                    if self.is_new_log(log_data):
                        # Create text representation
                        text = self.create_text_representation(log_data)
                        
                        # Create metadata - FIXED: Properly handle system_metrics
                        system_metrics = log_data.get("system_metrics", {})
                        if not isinstance(system_metrics, dict):
                            system_metrics = {}
                            
                        metadata = {
                            "timestamp": log_data.get("timestamp", ""),
                            "free_memory_percent": float(system_metrics.get("free_memory_percent", 0)),
                            "fragmentation_index": float(system_metrics.get("fragmentation_index", 0)),
                            "system_load": float(system_metrics.get("load", 0))
                        }
                        
                        # Generate a unique ID
                        log_id = self._hash_log(log_data)
                        
                        # Add to batch
                        batch_texts.append(text)
                        batch_ids.append(log_id)
                        batch_metadata.append(metadata)
                
                # Process batch if not empty
                if batch_texts:
                    # Batch embed all texts at once
                    batch_embeddings = self.model.encode(batch_texts)
                    
                    # Add all to vector DB in one batch
                    if FAISS_AVAILABLE and self.index is not None:
                        try:
                            # Add to FAISS index
                            self.index.add(np.array(batch_embeddings).astype('float32'))
                            
                            # Store metadata, documents, and IDs
                            self.metadatas.extend(batch_metadata)
                            self.documents.extend(batch_texts)
                            self.ids.extend(batch_ids)
                            
                            processed_count += len(batch_ids)
                            print(f"Added {len(batch_ids)} log entries to vector database in batch")
                            
                            # Save periodically
                            if len(self.ids) % 100 == 0:
                                self._save_index()
                                
                        except Exception as e:
                            print(f"Error adding batch to vector database: {e}")
            
            return processed_count
            
        except Exception as e:
            print(f"Error in pandas log processing: {e}")
            import traceback
            traceback.print_exc()  # Add this to get more detailed error information
            return processed_count
    
    def start_streaming_with_pandas(self, check_interval: float = 2.0, threshold_mb: float = 100.0):
        """
        Start streaming logs to the vector database using pandas for efficient reading.
        
        Args:
            check_interval: Interval in seconds to check for new logs
            threshold_mb: Memory usage threshold in MB for filtering high-usage processes
        """
        print(f"Starting pandas-based log streaming from: {self.log_path}")
        
        try:
            while not self.stop_event.is_set():
                processed = self.process_logs_with_pandas(max_logs=100, threshold_mb=threshold_mb)
                if processed > 0:
                    print(f"Processed {processed} new log entries")
                
                # Sleep for the check interval
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("Log streaming stopped by user")
    
    def start_streaming_in_background(self, check_interval: float = 2.0, threshold_mb: float = 100.0):
        """
        Start streaming logs to the vector database in a background thread.
        
        Args:
            check_interval: Interval in seconds to check for new logs
            threshold_mb: Memory usage threshold in MB for filtering high-usage processes
            
        Returns:
            Background thread object
        """
        self.stop_event.clear()
        thread = threading.Thread(
            target=self.start_streaming_with_pandas,
            args=(check_interval, threshold_mb),
            daemon=True
        )
        thread.start()
        print(f"Started background log streaming thread (interval: {check_interval}s)")
        return thread
    
    def stop_streaming(self):
        """
        Stop all background streaming threads.
        """
        self.stop_event.set()
        print("Stopping all background streaming threads")
        
        # Save the index before stopping
        self._save_index()
    
    def query_similar_logs(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for logs similar to the query text.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar logs with their metadata
        """
        if not FAISS_AVAILABLE or self.index is None:
            print("Vector database not available for querying")
            return []
        
        try:
            # Encode the query text
            query_embedding = self.model.encode(query_text).reshape(1, -1).astype('float32')
            
            # Query the FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            similar_logs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.ids) and idx >= 0:
                    similar_logs.append({
                        "id": self.ids[idx],
                        "text": self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "distance": float(distances[0][i])
                    })
            
            return similar_logs
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize the ingestion system
    ingestion = MemoryLogIngestion()
    
    try:
        # Process existing logs with pandas
        print("Processing existing logs with pandas...")
        num_processed = ingestion.process_logs_with_pandas(threshold_mb=100.0)
        print(f"Processed {num_processed} existing log entries")
        
        # Start streaming logs in real-time with pandas
        print("Starting real-time log streaming with pandas...")
        ingestion.start_streaming_in_background(check_interval=2.0, threshold_mb=100.0)
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping ingestion...")
        ingestion.stop_streaming()