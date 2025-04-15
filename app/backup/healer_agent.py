import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import ConnectionError, Timeout

# Import the MistralClient from llm_utils
from llm_utils import MistralClient

# Load environment variables
load_dotenv()

# Set up logging
log_dir = "d:/clg/COA/2/Self_healing_memory/logs"
os.makedirs(log_dir, exist_ok=True)

# Set default API key for healer agent
MISTRAL_API_KEY_HEALER = os.environ.get("MISTRAL_API_KEY_HEALER", "")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/healer_agent.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("healer_agent")

class MemoryHealerAgent:
    """
    Memory Healer Agent - Specialized in diagnosing and fixing memory issues.
    Uses RAG with historical memory analyses to generate precise healing actions.
    """
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        vector_db_dir: str = "d:/clg/COA/Self_healing_memory/data/vector_store",
        monitor_db_path: str = "d:/clg/COA/Self_healing_memory/data/vector_store/agent_analyses",
        cache_dir: str = "d:/clg/COA/Self_healing_memory/data/healer_cache"
    ):
        # Agent identity
        self.name = "Memory Surgeon"
        self.role = "Expert Memory Optimization Specialist"
        self.backstory = """A skilled memory surgeon with years of experience optimizing complex systems.
        I specialize in precise memory interventions, from defragmentation to leak remediation,
        with a deep understanding of system resources and memory allocation patterns."""
        
        # Initialize directories
        self.vector_db_dir = os.path.join(vector_db_dir, "healing_actions")
        self.monitor_db_path = monitor_db_path
        self.cache_dir = cache_dir
        os.makedirs(self.vector_db_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set API key for Mistral - use the healer-specific key only
        self.api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY_HEALER")
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
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("HuggingFace embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise RuntimeError(f"Critical error: Could not initialize embeddings: {e}")
        
        # Initialize vector stores
        self.setup_vector_stores()
        
        # Initialize prompt templates
        self.setup_prompt_templates()
        
        # Initialize monitoring thread
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        
        logger.info(f"Memory Healer Agent initialized: {self.name}")
    
    def setup_vector_stores(self):
        """Set up vector stores for healing actions and monitor data access"""
        try:
            # Initialize healer's vector store
            self.vectorstore = Chroma(
                collection_name="healing_actions",
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_dir
            )
            logger.info("Healing actions vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Chroma for healing actions: {e}")
            logger.info("Creating fresh vectorstore for healing actions")
            # Create fresh vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=[Document(page_content="Initial healing document", metadata={"type": "init"})],
                embedding=self.embeddings,
                collection_name="healing_actions",
                persist_directory=self.vector_db_dir
            )
        
        try:
            # Access monitor agent's vector store (read-only)
            self.monitor_vectorstore = Chroma(
                collection_name="agent_analyses",
                embedding_function=self.embeddings,
                persist_directory=self.monitor_db_path
            )
            logger.info("Monitor agent vector store accessed successfully")
        except Exception as e:
            logger.error(f"Error accessing monitor agent's vector store: {e}")
            logger.warning("Will operate without access to monitor agent's historical data")
            # Create placeholder if can't access
            self.monitor_vectorstore = None
        
        # Set up retrievers
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
        
        if self.monitor_vectorstore:
            self.monitor_retriever = self.monitor_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            )
        else:
            self.monitor_retriever = None
            logger.warning("Monitor retriever not available")
    
    def setup_prompt_templates(self):
        """Set up prompt templates for healing recommendations"""
        # Main healing template
        self.healing_template = PromptTemplate(
            input_variables=["context", "memory_stats"],
            template="""
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
        )
        
        # Specialized template for defragmentation
        self.defrag_template = PromptTemplate(
            input_variables=["memory_stats", "historical_context"],
            template="""
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
        )
        
        # Specialized template for process termination
        self.process_template = PromptTemplate(
            input_variables=["memory_stats", "historical_context"],
            template="""
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
        )
    
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
    
    def retrieve_monitor_analyses(self, query: str) -> List[Document]:
        """
        Retrieve similar analyses from the monitor agent's vector database.
        
        Args:
            query: Query string
            
        Returns:
            List of similar documents
        """
        if self.monitor_retriever:
            try:
                return self.monitor_retriever.invoke(query)
            except Exception as e:
                logger.error(f"Error retrieving monitor analyses: {e}")
        
        return []
    
    def retrieve_healing_actions(self, query: str) -> List[Document]:
        """
        Retrieve similar healing actions from the vector database.
        
        Args:
            query: Query string
            
        Returns:
            List of similar documents
        """
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving healing actions: {e}")
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
            historical_analyses = self.retrieve_monitor_analyses(stats_str)
            historical_context = "\n\n".join([doc.page_content for doc in historical_analyses])
            
            # Retrieve relevant healing actions
            healing_actions = self.retrieve_healing_actions(stats_str)
            healing_context = "\n\n".join([doc.page_content for doc in healing_actions])
            
            # Combine contexts
            combined_context = f"""HISTORICAL ANALYSES:
{historical_context}

PREVIOUS HEALING ACTIONS:
{healing_context}
"""
            
            # Generate main healing plan
            healing_prompt = self.healing_template.format(
                context=combined_context,
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
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "healing_id": healing_id,
                "agent_name": self.name,
                "agent_role": self.role,
                "type": "healing_plan"
            }
            
            # Add flattened memory stats (avoid nested dictionaries)
            flattened_stats = {}
            for key, value in memory_stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (dict, list)):
                            flattened_stats[f"{key}_{subkey}"] = subvalue
                elif not isinstance(value, (dict, list)):
                    flattened_stats[key] = value
            
            # Merge flattened stats
            metadata.update(flattened_stats)
            
            # Create text representation for vector storage
            text_repr = f"""HEALING PLAN: {healing_id}
            
GENERAL HEALING PLAN:
{healing_plan}

DEFRAGMENTATION PLAN:
{defrag_plan}

PROCESS TERMINATION PLAN:
{process_plan}

RECOMMENDATIONS:
{json.dumps(recommendations, indent=2)}

TIMESTAMP: {metadata['timestamp']}
"""
            
            # Add to vector store
            self.vectorstore.add_documents([
                Document(page_content=text_repr, metadata=metadata)
            ])
            
            # Cache the healing plan
            cache_file = os.path.join(self.cache_dir, f"{healing_id}.json")
            with open(cache_file, 'w') as f:
                json.dump({
                    "healing_id": healing_id,
                    "timestamp": metadata['timestamp'],
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
    
    def get_recent_healing_plans(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent healing plans from cache.
        
        Args:
            limit: Maximum number of plans to return
            
        Returns:
            List of recent healing plans
        """
        plans = []
        
        try:
            files = os.listdir(self.cache_dir)
            json_files = [f for f in files if f.endswith('.json') and f.startswith('healing_')]
            
            # Sort by modification time (newest first)
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x)), reverse=True)
            
            # Load the most recent files
            for file in json_files[:limit]:
                with open(os.path.join(self.cache_dir, file), 'r') as f:
                    plans.append(json.load(f))
                    
        except Exception as e:
            logger.error(f"Error getting recent healing plans: {e}")
        
        return plans
    
    def start_monitoring(self, monitor_agent, interval: int = 300):
        """
        Start monitoring memory conditions using the monitor agent.
        
        Args:
            monitor_agent: Monitor agent instance
            interval: Interval between checks in seconds
        """
        with threading.Lock():
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                logger.warning("Healing monitoring already running")
                return
                
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self.monitor_and_heal,
                args=(monitor_agent, interval),
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info(f"Memory healing monitoring started with interval: {interval} seconds")
    
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
        
    def monitor_and_heal(self, monitor_agent, predictor_agent=None, interval: int = 300):
        """
        Continuously monitor memory conditions and generate healing plans.
        
        Args:
            monitor_agent: Monitor agent instance
            predictor_agent: Optional predictor agent instance
            interval: Interval between checks in seconds
        """
        logger.info(f"Starting memory healing monitoring (interval: {interval} seconds)")
        
        while not self.stop_event.is_set():
            try:
                # Get current memory stats from monitor agent
                stats = monitor_agent.retrieve_memory_stats()
                
                # Get monitor agent's analysis
                analysis = monitor_agent.analyze_memory_condition(stats)
                
                # Get predictor suggestions if predictor agent is available
                predictor_suggestions = None
                if predictor_agent:
                    try:
                        predictor_suggestions = predictor_agent.get_predictions(stats)
                        logger.info(f"Received {len(predictor_suggestions)} suggestions from predictor agent")
                    except Exception as e:
                        logger.error(f"Error getting predictions from predictor agent: {e}")
                
                # Check if there are issues that need healing
                issues = analysis.get('issues', {})
                if issues:
                    logger.info(f"Memory issues detected ({len(issues)}), generating healing plan")
                    
                    # Generate healing plan with predictor suggestions
                    healing_plan = self.generate_healing_plan(stats, predictor_suggestions)
                    
                    # Send healing suggestions to terminator
                    self.send_healing_suggestions_to_terminator(
                        healing_plan['recommendations'], 
                        stats
                    )
                    
                    print("\n=== MEMORY HEALING PLAN ===")
                    print("\n=== GENERAL HEALING PLAN ===")
                    print(healing_plan['general_plan'].strip())
                    
                    print("\n=== DEFRAGMENTATION PLAN ===")
                    print(healing_plan['defragmentation_plan'].strip())
                    
                    print("\n=== PROCESS TERMINATION PLAN ===")
                    print(healing_plan['process_termination_plan'].strip())
                    
                    print("\n=== RECOMMENDATIONS ===")
                    for i, rec in enumerate(healing_plan['recommendations'], 1):
                        print(f"{i}. {rec.strip()}")
                else:
                    logger.info(f"No memory issues detected, skipping healing plan generation")
                
                # Wait for next check
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory healing monitoring: {e}")
                time.sleep(10)  # Wait a bit before retrying
    
    def stop(self):
        """
        Stop memory healing monitoring.
        """
        self.stop_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        logger.info("Memory healing monitoring stopped")


# Example usage
if __name__ == "__main__":
    # Get Mistral API key from environment variable with proper fallbacks
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Try to get healer-specific key first
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY_HEALER")
    
    if not MISTRAL_API_KEY:
        print("ERROR: No Mistral API key found for healer agent. Please set MISTRAL_API_KEY_HEALER environment variable.")
        exit(1)
    
    try:
        # Initialize the Memory Healer Agent
        healer = MemoryHealerAgent(
            mistral_api_key=MISTRAL_API_KEY
        )
        
        # Example memory stats for testing
        test_stats = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "free_memory_percent": 15.2,
                "fragmentation_index": 0.72,
                "total": 16000000000,
                "available": 2432000000
            },
            "processes": [
                {"pid": 1234, "name": "chrome.exe", "memory_usage": 1200000000},
                {"pid": 5678, "name": "firefox.exe", "memory_usage": 800000000},
                {"pid": 9012, "name": "suspicious_process.exe", "memory_usage": 500000000}
            ]
        }
        
        # Generate a healing plan
        print("\n=== MEMORY HEALING PLAN ===")
        healing_plan = healer.generate_healing_plan(test_stats)
        
        print("\n=== GENERAL HEALING PLAN ===")
        print(healing_plan['general_plan'].strip())
        
        print("\n=== DEFRAGMENTATION PLAN ===")
        print(healing_plan['defragmentation_plan'].strip())
        
        print("\n=== PROCESS TERMINATION PLAN ===")
        print(healing_plan['process_termination_plan'].strip())
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(healing_plan['recommendations'], 1):
            print(f"{i}. {rec}")
            
    except Exception as e:
        print(f"\nERROR: Failed to initialize or run Memory Healer Agent: {e}")
        print("Please check the logs for more details.")
    finally:
        if 'healer' in locals():
            healer.stop()