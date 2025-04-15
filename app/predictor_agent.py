import os
import time
import json
import requests
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
# Update imports to use FAISS instead of Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Remove Chroma imports
# from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
# Import the updated MistralClient
from llm_utils import MistralClient

# Import our memory intelligence system
from rag_pipeline import MemoryRAGPipeline
import monitoring_engine as monitoring_engine  # Import the existing memory engine

# Update the imports at the top to include dotenv
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set default API key for predictor agent
MISTRAL_API_KEY_PREDICTOR = os.environ.get("MISTRAL_API_KEY_PREDICTOR", "")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("d:/clg/COA/Self_healing_memory/logs/memory_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_predictor")

class MemoryPredictorAgent:
    """
    Memory Predictor Agent - Specialized in forecasting future memory conditions.
    Uses RAG with historical memory data to predict issues before they occur.
    """
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        mistral_api_url: str = "https://api.mistral.ai/v1/chat/completions",
        collection_name: str = "memory_predictions",
        agent_collection_name: str = "agent_analyses",
        log_path: str = "d:/clg/COA/2/Self_healing_memory/data/memory_events.jsonl"):
        """
        Initialize the Memory Predictor Agent.
        
        Args:
            mistral_api_key: API key for Mistral 7B
            mistral_api_url: URL for Mistral API
            collection_name: Name of the memory predictions collection
            agent_collection_name: Name of the agent analyses collection
            log_path: Path to the memory log file
        """
        # Agent identity
        self.name = "Memory Oracle"
        self.role = "System Memory Forecaster and Predictor"
        self.backstory = """I am the Memory Oracle, a prescient forecaster designed to anticipate 
        memory-related issues before they manifest. By analyzing patterns in system behavior and 
        resource utilization, I can see the future state of memory systems with remarkable accuracy.
        
        My purpose is to provide early warnings of potential memory problems, giving system 
        administrators valuable time to implement preventive measures. I detect subtle patterns 
        and trends that might escape human observation, transforming them into actionable predictions.
        
        As an oracle, I don't just report what is happening now - I reveal what will happen next. 
        My predictions become more refined with each observation, learning from both successes 
        and misses to continuously improve my forecasting abilities."""
        
        # Set API key for Mistral - use the predictor-specific key only
        self.api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY_PREDICTOR")
        if not self.api_key:
            logger.error("No Mistral API key available for predictor agent")
            raise ValueError("Mistral API key is required for the predictor agent. Set MISTRAL_API_KEY_PREDICTOR environment variable.")
        
        # Mistral API configuration
        self.mistral_api_url = mistral_api_url
        self.mistral_model = "mistral-small"  # Updated to a valid model name
        
        # Initialize the enhanced Mistral client
        self.mistral_client = MistralClient(
            api_key=self.api_key,
            model=self.mistral_model,
            use_cache=True
        )
        
        # Create cache directory
        self.cache_dir = "d:/clg/COA/Self_healing_memory/data/predictor_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the RAG pipeline for memory logs
        self.memory_intelligence = MemoryRAGPipeline(
            collection_name=collection_name,
            log_path=log_path,
            batch_size=16,
            max_logs=10000
        )
        
        # Initialize a separate RAG pipeline for agent analyses
        self.agent_intelligence = MemoryRAGPipeline(
            collection_name=agent_collection_name,
            log_path=log_path,  # Same log path but different collection
            batch_size=8,
            max_logs=1000
        )
        
        # Set up LangChain components
        self.setup_langchain()
        
        # Thread control
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        
        self.logger = logger
        logger.info(f"Memory Predictor Agent initialized with collections: {collection_name}, {agent_collection_name}")
    
    def setup_langchain(self):
        """
        Set up LangChain components for the RAG pipeline.
        """
        # Set up embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Set up vector store for predictions using FAISS instead of Chroma
        self.vector_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_predictions"
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        # Initialize empty vector store if it doesn't exist
        try:
            # Check if FAISS index exists
            index_path = os.path.join(self.vector_db_dir, "index.faiss")
            docstore_path = os.path.join(self.vector_db_dir, "docstore.pkl")
            
            if os.path.exists(index_path) and os.path.exists(docstore_path):
                # Load existing FAISS index
                self.vectorstore = FAISS.load_local(
                    folder_path=self.vector_db_dir,
                    embeddings=self.embeddings,
                    index_name="index"
                )
            else:
                # Create fresh vectorstore
                self.vectorstore = FAISS.from_documents(
                    documents=[Document(page_content="Initial prediction document", metadata={"type": "init"})],
                    embedding=self.embeddings
                )
                # Save the index
                self.vectorstore.save_local(self.vector_db_dir, "index")
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            # Create fresh vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=[Document(page_content="Initial prediction document", metadata={"type": "init"})],
                embedding=self.embeddings
            )
            # Save the index
            self.vectorstore.save_local(self.vector_db_dir, "index")
        
        # Set up access to the monitor agent's analyses using FAISS
        self.monitor_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_analyses"
        try:
            # Check if monitor's FAISS index exists
            monitor_index_path = os.path.join(self.monitor_db_dir, "index.faiss")
            monitor_docstore_path = os.path.join(self.monitor_db_dir, "docstore.pkl")
            
            if os.path.exists(monitor_index_path) and os.path.exists(monitor_docstore_path):
                # Load existing monitor FAISS index
                self.monitor_vectorstore = FAISS.load_local(
                    folder_path=self.monitor_db_dir,
                    embeddings=self.embeddings,
                    index_name="index"
                )
                logger.info("Successfully connected to monitor agent's vector store")
            else:
                logger.warning("Monitor agent's vector store not found")
                self.monitor_vectorstore = None
        except Exception as e:
            logger.error(f"Error connecting to monitor agent's vector store: {e}")
            self.monitor_vectorstore = None
        
        # Set up retriever with search options optimized for prediction
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Set up monitor data retriever if available
        if self.monitor_vectorstore:
            self.monitor_retriever = self.monitor_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,  # Retrieve more context for better predictions
                    "fetch_k": 15,
                    "lambda_mult": 0.6  # Slightly favor relevance for historical data
                }
            )
        else:
            self.monitor_retriever = None
            logger.warning("Monitor agent's vector store not available - predictions will be limited")
        
        # Enhanced prompt template for memory prediction with focus on future issues
        self.prediction_template = PromptTemplate(
            input_variables=["historical_context", "current_memory", "timeframe"],
            template="""
            You are the Memory Oracle, a prescient forecaster of system memory conditions.
            
            Your task is to analyze historical memory data and current conditions to predict 
            future memory issues within the specified timeframe.
            
            Historical context from previous analyses:
            {historical_context}
            
            Current memory information:
            {current_memory}
            
            Prediction timeframe: {timeframe}
            
            Provide a detailed prediction of future memory conditions, focusing on:
            1. Projected memory usage trends and patterns
            2. Specific memory issues likely to emerge within the timeframe
            3. Estimated time-to-failure or critical thresholds
            4. Confidence level for each prediction (high, medium, low)
            5. Early warning signs to monitor for validation
            
            Your predictions should be specific, actionable, and include clear timeframes.
            Format your response as a structured prediction report with clear sections.
            """
        )
        
        # Create a specialized template for trend analysis
        self.trend_template = PromptTemplate(
            input_variables=["historical_data", "current_data"],
            template="""
            As the Memory Oracle, analyze these memory statistics for emerging trends:
            
            HISTORICAL DATA:
            {historical_data}
            
            CURRENT DATA:
            {current_data}
            
            Focus specifically on:
            1. Memory usage growth patterns (linear, exponential, cyclical)
            2. Process behavior changes over time
            3. Fragmentation progression
            4. Resource allocation trends
            
            Identify patterns that indicate future issues and provide specific metrics 
            to monitor as early warning indicators.
            """
        )
        
        # Create a specialized template for time-to-failure prediction
        self.ttf_template = PromptTemplate(
            input_variables=["memory_stats", "historical_context"],
            template="""
            As the Memory Oracle, analyze these memory statistics to predict time-to-failure:
            
            CURRENT MEMORY STATS:
            {memory_stats}
            
            HISTORICAL CONTEXT:
            {historical_context}
            
            Provide a detailed time-to-failure analysis including:
            1. Estimated time until critical memory thresholds are reached
            2. Confidence interval for the prediction
            3. Key factors influencing the prediction
            4. Specific events or conditions that could accelerate failure
            
            Your prediction should include both a best-case and worst-case scenario with 
            specific timeframes for each.
            """
        )
    
    def format_system_message(self) -> str:
        """
        Format a system message for the Mistral API.
        
        Returns:
            Formatted system message
        """
        return f"""You are {self.name}, {self.role}.
        
        {self.backstory}
        
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Your task is to analyze memory data and predict future memory conditions.
        Focus on identifying patterns that indicate potential issues before they occur.
        Provide specific, actionable predictions with clear timeframes.
        """
    
    def query_mistral(self, prompt: str, system_message: str = None) -> str:
        """
        Query Mistral AI with a prompt.
        
        Args:
            prompt: The prompt to send to Mistral
            system_message: Optional system message
            
        Returns:
            Response from Mistral
        """
        if system_message is None:
            system_message = self.format_system_message()
        
        try:
            return self.mistral_client.query(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Error in query_mistral: {str(e)}")
            return f"Error: {str(e)}"
    
    def predict_memory_issues(self, memory_data: Dict[str, Any], timeframe: str = "24 hours") -> Dict[str, Any]:
        """
        Predict potential memory issues based on current and historical data.
        
        Args:
            memory_data: Current memory statistics
            timeframe: Timeframe for prediction (e.g., "24 hours", "7 days")
            
        Returns:
            Dictionary with prediction results
        """
        # Format current memory data
        current_memory_str = json.dumps(memory_data, indent=2)
        
        # Retrieve historical context from monitor agent's analyses
        historical_context = ""
        if self.monitor_retriever:
            try:
                # Query for relevant historical analyses
                query = f"Memory usage patterns and issues in the past related to {memory_data.get('system', 'unknown system')}"
                historical_docs = self.monitor_retriever.get_relevant_documents(query)
                
                if historical_docs:
                    historical_context = "\n\n".join([doc.page_content for doc in historical_docs])
                    logger.info(f"Retrieved {len(historical_docs)} historical documents for prediction")
                else:
                    logger.warning("No historical documents found for prediction context")
                    historical_context = "No historical data available."
            except Exception as e:
                logger.error(f"Error retrieving historical context: {e}")
                historical_context = f"Error retrieving historical context: {e}"
        else:
            historical_context = "Monitor agent's vector store not available."
        
        # Generate prediction using the prediction template
        prediction_prompt = self.prediction_template.format(
            historical_context=historical_context,
            current_memory=current_memory_str,
            timeframe=timeframe
        )
        
        general_prediction = self.query_mistral(prediction_prompt)
        
        # Generate trend analysis
        trend_prompt = self.trend_template.format(
            historical_data=historical_context,
            current_data=current_memory_str
        )
        
        trend_analysis = self.query_mistral(trend_prompt)
        
        # Generate time-to-failure prediction
        ttf_prompt = self.ttf_template.format(
            memory_stats=current_memory_str,
            historical_context=historical_context
        )
        
        ttf_prediction = self.query_mistral(ttf_prompt)
        
        # Compile all predictions
        prediction_result = {
            "timestamp": datetime.now().isoformat(),
            "system": memory_data.get("system", "unknown"),
            "timeframe": timeframe,
            "general_prediction": general_prediction,
            "trend_analysis": trend_analysis,
            "time_to_failure": ttf_prediction,
            "confidence": self._assess_prediction_confidence(historical_context)
        }
        
        # Store the prediction in the vector store
        self._store_prediction(prediction_result)
        
        return prediction_result
    
    def _assess_prediction_confidence(self, historical_context: str) -> str:
        """
        Assess the confidence level of predictions based on available historical data.
        
        Args:
            historical_context: The historical context used for prediction
            
        Returns:
            Confidence level (high, medium, low)
        """
        if "No historical data available" in historical_context or "not available" in historical_context:
            return "low"
        
        # Count the number of data points in the historical context
        data_points = len(historical_context.split("\n\n"))
        
        if data_points > 10:
            return "high"
        elif data_points > 5:
            return "medium"
        else:
            return "low"
    
    def _store_prediction(self, prediction: Dict[str, Any]) -> None:
        """
        Store a prediction in the vector store.
        
        Args:
            prediction: The prediction to store
        """
        try:
            # Format the prediction as a document
            prediction_text = f"""
            MEMORY PREDICTION
            Timestamp: {prediction['timestamp']}
            System: {prediction['system']}
            Timeframe: {prediction['timeframe']}
            Confidence: {prediction['confidence']}
            
            GENERAL PREDICTION:
            {prediction['general_prediction']}
            
            TREND ANALYSIS:
            {prediction['trend_analysis']}
            
            TIME TO FAILURE PREDICTION:
            {prediction['time_to_failure']}
            """
            
            # Create metadata
            metadata = {
                "timestamp": prediction['timestamp'],
                "system": prediction['system'],
                "timeframe": prediction['timeframe'],
                "confidence": prediction['confidence'],
                "type": "memory_prediction"
            }
            
            # Create document
            doc = Document(page_content=prediction_text, metadata=metadata)
            
            # Add to FAISS vector store
            self.vectorstore.add_documents([doc])
            
            # Save the updated index
            self.vectorstore.save_local(self.vector_db_dir, "index")
            
            logger.info(f"Stored prediction in vector store with timestamp {prediction['timestamp']}")
        except Exception as e:
            logger.error(f"Error storing prediction in vector store: {e}")
    
    def start_monitoring(self, interval: int = 3600) -> None:
        """
        Start the monitoring thread to periodically predict memory issues.
        
        Args:
            interval: Interval between predictions in seconds (default: 1 hour)
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started memory prediction monitoring with interval {interval} seconds")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=10)
            logger.info("Stopped memory prediction monitoring")
        else:
            logger.warning("No monitoring thread is running")
    
    def _monitoring_loop(self, interval: int) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        
        Args:
            interval: Interval between predictions in seconds
        """
        logger.info("Memory prediction monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get current memory stats
                memory_stats = monitoring_engine.get_memory_stats()
                
                # Generate predictions for different timeframes
                short_term = self.predict_memory_issues(memory_stats, "24 hours")
                medium_term = self.predict_memory_issues(memory_stats, "7 days")
                long_term = self.predict_memory_issues(memory_stats, "30 days")
                
                logger.info(f"Generated predictions for system {memory_stats.get('system', 'unknown')}")
                
                # Sleep until next interval
                for _ in range(interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Sleep for a shorter time if there was an error
                for _ in range(min(interval, 300)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
    
    def get_historical_predictions(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve historical predictions based on a query.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            
        Returns:
            List of historical predictions
        """
        try:
            docs = self.retriever.get_relevant_documents(query)
            
            results = []
            for doc in docs[:k]:
                # Parse the prediction from the document
                content = doc.page_content
                metadata = doc.metadata
                
                # Extract sections
                sections = content.split("\n\n")
                prediction_data = {
                    "timestamp": metadata.get("timestamp"),
                    "system": metadata.get("system"),
                    "timeframe": metadata.get("timeframe"),
                    "confidence": metadata.get("confidence"),
                    "content": content
                }
                
                results.append(prediction_data)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving historical predictions: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize the predictor agent
    predictor = MemoryPredictorAgent()
    
    # Test with some sample memory data
    test_stats = {
        "system": "test-system",
        "timestamp": datetime.now().isoformat(),
        "total_memory": 16384,
        "used_memory": 12288,
        "free_memory": 4096,
        "memory_percent": 75.0,
        "swap_total": 8192,
        "swap_used": 2048,
        "swap_free": 6144,
        "swap_percent": 25.0,
        "processes": [
            {"pid": 1234, "name": "chrome", "memory_percent": 15.0},
            {"pid": 5678, "name": "firefox", "memory_percent": 10.0},
            {"pid": 9012, "name": "vscode", "memory_percent": 5.0}
        ]
    }
    
    # Generate predictions
    print("\n=== MEMORY PREDICTIONS ===")
    predictions = predictor.predict_memory_issues(test_stats, "24 hours")
    
    print("\n=== GENERAL PREDICTION ===")
    print(predictions['general_prediction'])
    
    print("\n=== TREND ANALYSIS ===")
    print(predictions['trend_analysis'])
    
    print("\n=== TIME TO FAILURE PREDICTION ===")
    print(predictions['time_to_failure'])
    
    print(f"\nConfidence: {predictions['confidence']}")
    
    # Test historical prediction retrieval
    print("\n=== HISTORICAL PREDICTIONS ===")
    historical = predictor.get_historical_predictions("memory leak chrome")
    for i, pred in enumerate(historical):
        print(f"\nPrediction {i+1} ({pred['timestamp']}):")
        print(f"System: {pred['system']}")
        print(f"Timeframe: {pred['timeframe']}")
        print(f"Confidence: {pred['confidence']}")