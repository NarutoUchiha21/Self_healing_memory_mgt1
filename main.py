# -*- coding: utf-8 -*-

"""
Self-Healing Memory System - Main Entry Point
=============================================

This script connects and runs all components of the Self-Healing Memory System.
It initializes the monitoring engine, ingestion system, RAG pipeline, and all agent types
(monitor, predictor, healer) and provides a web interface for interaction.
"""

import os
import sys
# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import json
import logging
import warnings
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import the modules with correct paths
from app.monitoring_engine import MemoryMonitoringEngine
from app.ingestion import MemoryLogIngestion  # This is the correct import
from app.rag_pipeline import MemoryRAGPipeline
from app.monitor_agent import MemoryMonitorAgent
from app.predictor_agent import MemoryPredictorAgent
from app.healer_agent import MemoryHealerAgent

# Filter Faiss GPU warnings
warnings.filterwarnings("ignore", message="Failed to load GPU Faiss")

# Set up base directories
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
MONITOR_CACHE_DIR = os.path.join(DATA_DIR, "monitor_cache")
PREDICTOR_CACHE_DIR = os.path.join(DATA_DIR, "predictor_cache")
HEALER_CACHE_DIR = os.path.join(DATA_DIR, "healer_cache")

# Load environment variables from .env file
load_dotenv()

# Set Mistral API keys if not already in environment
if not os.environ.get("MISTRAL_API_KEY_MONITOR"):
    os.environ["MISTRAL_API_KEY_MONITOR"] = "Ys2Wr6LkIWHg02CWH7Ny1LTC7mN1iRq9"
    os.environ["MISTRAL_API_KEY_HEALER"] = "aGVJ9IAXzi7CyqzDXL9XdSzkOk2jx9Jx"
    os.environ["MISTRAL_API_KEY_PREDICTOR"] = "yQDw5Cwu0M3ZjIsXrwzHTosT96VOM7jm"
    os.environ["MISTRAL_API_KEY_EXPLAINER"] = "5czbUFRKmTpJbWbjCiZA0Ch72NbqOep5"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter out specific library warnings
for lib_logger in ["faiss", "sentence_transformers"]:
    logging.getLogger(lib_logger).setLevel(logging.ERROR)

def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        LOGS_DIR, 
        DATA_DIR, 
        CACHE_DIR,
        VECTOR_STORE_DIR,
        MONITOR_CACHE_DIR,
        PREDICTOR_CACHE_DIR,
        HEALER_CACHE_DIR
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def start_agents():
    """Initialize and start memory management agents."""
    try:
        # Import required modules
       
        
        ingestion = MemoryLogIngestion()    
        ingestion.start()
        
        # Create RAG pipeline
        rag = MemoryRAGPipeline()
        
        # Initialize monitoring engine
        monitor_engine = MemoryMonitoringEngine()
        monitor_engine.start()

        # Initialize agents with the RAG pipeline
        monitor = MemoryMonitorAgent(rag)
        predictor = MemoryPredictorAgent(rag)
        healer = MemoryHealerAgent(rag)
        
        # Start agents in separate threads
        monitor.start_monitoring()
        predictor.start_prediction_service()
        healer.start_healing_service()
        
        logger.info("All agents started successfully")
        
        return monitor, predictor, healer, ingestion
    except Exception as e:
        logger.error(f"Failed to start agents: {str(e)}")
        raise

def start_web_interface():
    """Start the web interface in a separate thread."""
    logger.info("Starting web interface...")
    
    try:
        from web_interface import app
        
        # Run Flask in a separate thread
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        )
        flask_thread.daemon = True
        flask_thread.start()
        
        logger.info("Web interface started at http://localhost:5000")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start web interface: {str(e)}")
        return False

def main():
    """Main entry point for the self-healing memory system."""
    print("=" * 80)
    print("Self-Healing Memory System")
    print("=" * 80)
    
    # Ensure all required directories exist
    ensure_directories()
    
    try:
        # Start all agents
        agents = start_agents()
        
        # Start web interface
        start_web_interface()
        
        print("\nSystem is running. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down Self-Healing Memory System...")
        logger.info("Shutting down Self-Healing Memory System")
        # Cleanup could be added here
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nError: {str(e)}")
        
    finally:
        print("System shutdown complete.")

if __name__ == "__main__":
    main()