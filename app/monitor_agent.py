import os
import time
import json
import requests
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
# Update imports to use FAISS instead of Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
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

# Set default API key for monitor agent
MISTRAL_API_KEY_MONITOR = os.environ.get("MISTRAL_API_KEY_MONITOR", "")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("d:/clg/COA/Self_healing_memory/logs/memory_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_monitor")

class MemoryMonitorAgent:
    """
    Memory Monitor Agent - Specialized in monitoring and analyzing memory conditions.
    Uses RAG with historical memory data to detect issues and provide recommendations.
    """
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        mistral_api_url: str = "https://api.mistral.ai/v1/chat/completions",
        collection_name: str = "memory_logs",
        agent_collection_name: str = "agent_analyses",
        log_path: str = "d:/clg/COA/Self_healing_memory/data/memory_events.jsonl"):
        """
        Initialize the Memory Monitor Agent.
        
        Args:
            mistral_api_key: API key for Mistral 7B
            mistral_api_url: URL for Mistral API
            collection_name: Name of the memory logs collection
            agent_collection_name: Name of the agent analyses collection
            log_path: Path to the memory log file
        """
        # Agent identity with more concise backstory
        self.name = "Memory Sentinel"
        self.role = "System Memory Monitor and Analyst"
        self.backstory = """Memory Sentinel: An AI guardian that monitors system memory, 
        predicts issues before they occur, and provides actionable recommendations to 
        prevent crashes and optimize performance."""
        
        # Set API key for Mistral - use the monitor-specific key only
        self.api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY_MONITOR")
        if not self.api_key:
            logger.error("No Mistral API key available for monitor agent")
            raise ValueError("Mistral API key is required for the monitor agent. Set MISTRAL_API_KEY_MONITOR environment variable.")
        
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
        self.cache_dir = "d:/clg/COA/Self_healing_memory/data/monitor_cache"
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
        logger.info(f"Memory Monitor Agent initialized with collections: {collection_name}, {agent_collection_name}")
    
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
        
        # Set up vector store for analyses using FAISS instead of Chroma
        self.vector_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_analyses"
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
                    documents=[Document(page_content="Initial document", metadata={"type": "init"})],
                    embedding=self.embeddings
                )
                # Save the index
                self.vectorstore.save_local(self.vector_db_dir, "index")
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            # Create fresh vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=[Document(page_content="Initial document", metadata={"type": "init"})],
                embedding=self.embeddings
            )
            # Save the index
            self.vectorstore.save_local(self.vector_db_dir, "index")
        
        # Set up retriever with search options optimized for memory analysis
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Enhanced prompt template for memory analysis with more focus on OOM and fragmentation
        self.analysis_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are the Memory Sentinel, a vigilant guardian of system memory.
            
            Your task is to analyze memory conditions and detect potential issues before they become critical.
            
            Context from previous analyses:
            {context}
            
            Current memory information to analyze:
            {question}
            
            Provide a concise analysis of the memory conditions, focusing on:
            1. Overall memory health assessment
            2. Detection of abnormal conditions (OOM risks, fragmentation, leaks)
            3. Risk assessment for potential memory-related failures
            4. Specific recommendations for optimization or preventive actions
            
            Keep your response brief and actionable.
            """
        )
        
        # Create a specialized template for OOM detection
        self.oom_detection_template = PromptTemplate(
            input_variables=["memory_stats", "historical_context"],
            template="""
            As the Memory Sentinel, analyze these memory statistics for OOM risk:
            
            CURRENT MEMORY STATS:
            {memory_stats}
            
            HISTORICAL CONTEXT:
            {historical_context}
            
            Focus specifically on:
            1. Current free memory percentage and trend
            2. Memory consumption rate
            3. Large memory consumers
            4. Historical patterns that preceded OOM events
            
            Provide a risk assessment with confidence level and timeframe if applicable.
            """
        )
        
        # Create a specialized template for fragmentation analysis
        self.fragmentation_template = PromptTemplate(
            input_variables=["memory_stats", "historical_context"],
            template="""
            As the Memory Sentinel, analyze these memory statistics for fragmentation issues:
            
            CURRENT MEMORY STATS:
            {memory_stats}
            
            HISTORICAL CONTEXT:
            {historical_context}
            
            Focus specifically on:
            1. Current fragmentation index and trend
            2. Memory allocation patterns
            3. Process memory usage patterns
            4. Historical patterns of increasing fragmentation
            
            Provide a fragmentation assessment with severity level and impact on system performance.
            """
        )
    
    def format_system_message(self) -> str:
        """
        Format the system message for the Mistral API.
        
        Returns:
            Formatted system message
        """
        return f"""You are {self.name}, {self.role}.
        
{self.backstory}

Analyze the memory information provided and give detailed insights.
Focus on detecting abnormal conditions, potential issues, and recommendations.
Be concise but thorough in your analysis.
"""
    
    def query_mistral(self, prompt: str, system_message: str = None) -> str:
        """
        Query the Mistral model with a prompt using the enhanced client.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            
        Returns:
            Model response or error message
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
    
    def retrieve_memory_stats(self) -> Dict[str, Any]:
        """
        Retrieve memory statistics from the monitoring engine.
        """
        try:
            # Try to get stats from monitoring_engine
            return monitoring_engine.get_memory_stats()
        except (AttributeError, ImportError) as e:
            logger.error(f"Could not retrieve stats from monitoring_engine: {e}")
            
            # Fallback implementation using psutil
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "timestamp": datetime.now().isoformat(),
                    "system_metrics": {
                        "free_memory_percent": memory.percent,
                        "fragmentation_index": 0.0,
                        "total": memory.total,
                        "available": memory.available
                    }
                }
            except ImportError:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "system_metrics": {
                        "free_memory_percent": 0,
                        "fragmentation_index": 0
                    },
                    "note": "No memory data available"
                }
    
    def retrieve_similar_analyses(self, query: str) -> List[Document]:
        """
        Retrieve similar analyses from the vector database.
        """
        return self.retriever.invoke(query)  # Updated from get_relevant_documents
    
    def analyze_memory_condition(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze memory conditions using Mistral 7B and RAG.
        
        Args:
            stats: Memory statistics to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Format stats for the model
        formatted_stats = json.dumps(stats, indent=2)
        
        # Retrieve similar past analyses
        similar_analyses = self.retrieve_similar_analyses(formatted_stats)
        context = "\n\n".join([doc.page_content for doc in similar_analyses])
        
        # Create prompt with context
        prompt = self.analysis_template.format(
            context=context,
            question=formatted_stats
        )
        
        # Query Mistral
        analysis_text = self.query_mistral(prompt)
        
        # Perform specialized OOM analysis
        oom_prompt = self.oom_detection_template.format(
            memory_stats=formatted_stats,
            historical_context=context
        )
        oom_analysis = self.query_mistral(oom_prompt)
        
        # Perform specialized fragmentation analysis
        frag_prompt = self.fragmentation_template.format(
            memory_stats=formatted_stats,
            historical_context=context
        )
        frag_analysis = self.query_mistral(frag_prompt)
        
        # Detect issues
        issues = self._detect_issues(stats, analysis_text, oom_analysis, frag_analysis)
        
        # Store this analysis
        analysis_id = self.store_analysis(formatted_stats, analysis_text, stats)
        
        # Return structured analysis
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "memory_stats": stats,
            "analysis": analysis_text,
            "oom_analysis": oom_analysis,
            "fragmentation_analysis": frag_analysis,
            "issues": issues,
            "recommendations": self._extract_recommendations(analysis_text)
        }
    
    def _detect_issues(self, stats: Dict[str, Any], analysis_text: str, 
                      oom_analysis: str = None, frag_analysis: str = None) -> Dict[str, Any]:
        """
        Detect memory issues based on stats and analysis.
        
        Args:
            stats: Memory statistics
            analysis_text: Analysis text from Mistral
            oom_analysis: Specialized OOM analysis
            frag_analysis: Specialized fragmentation analysis
            
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        # Extract metrics
        metrics = stats.get('system_metrics', {})
        free_percent = metrics.get('free_memory_percent', 100)
        frag_index = metrics.get('fragmentation_index', 0)
        
        # Check for OOM risk
        if free_percent < 10:
            issues['oom_risk'] = {
                'severity': 'critical',
                'description': f'Only {free_percent:.1f}% memory available',
                'recommendation': 'Terminate non-essential processes immediately'
            }
        elif free_percent < 20:
            issues['low_memory'] = {
                'severity': 'warning',
                'description': f'Only {free_percent:.1f}% memory available',
                'recommendation': 'Consider closing memory-intensive applications'
            }
        
        # Check for fragmentation
        if frag_index > 0.7:
            issues['high_fragmentation'] = {
                'severity': 'warning',
                'description': f'Memory fragmentation index at {frag_index:.2f}',
                'recommendation': 'Consider memory defragmentation or system restart'
            }
        elif frag_index > 0.5:
            issues['moderate_fragmentation'] = {
                'severity': 'info',
                'description': f'Memory fragmentation index at {frag_index:.2f}',
                'recommendation': 'Monitor fragmentation trend over time'
            }
        
        # Check for memory leaks (simplified detection)
        if "leak" in analysis_text.lower() or "leaking" in analysis_text.lower():
            issues['potential_memory_leak'] = {
                'severity': 'warning',
                'description': 'Potential memory leak detected in analysis',
                'recommendation': 'Monitor memory usage over time and identify leaking processes'
            }
        
        # Incorporate insights from specialized analyses
        if oom_analysis and "high risk" in oom_analysis.lower():
            if 'oom_risk' not in issues:
                issues['oom_risk_predicted'] = {
                    'severity': 'warning',
                    'description': 'OOM risk predicted based on historical patterns',
                    'recommendation': 'Proactively free memory resources'
                }
                
        if frag_analysis and "severe fragmentation" in frag_analysis.lower():
            if 'high_fragmentation' not in issues:
                issues['fragmentation_risk_predicted'] = {
                    'severity': 'warning',
                    'description': 'Increasing fragmentation risk based on memory patterns',
                    'recommendation': 'Schedule system maintenance to address fragmentation'
                }
        
        return issues
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """
        Extract recommendations from analysis text.
        
        Args:
            analysis_text: Analysis text
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Look for recommendations section
        if "recommendation" in analysis_text.lower():
            lines = analysis_text.split('\n')
            in_recommendations = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we're in the recommendations section
                if in_recommendations:
                    # Check if we've reached the end of the section
                    if line.startswith('#') or (line == '' and len(recommendations) > 0):
                        in_recommendations = False
                    elif line and not line.startswith('#'):
                        # Clean up bullet points and numbering
                        clean_line = line
                        for prefix in ['- ', '• ', '* ', '. ']:
                            if clean_line.startswith(prefix):
                                clean_line = clean_line[len(prefix):]
                                break
                        
                        # Remove numbering (e.g., "1. ")
                        if clean_line and clean_line[0].isdigit() and '. ' in clean_line[:4]:
                            clean_line = clean_line[clean_line.index('. ')+2:]
                        
                        if clean_line:
                            recommendations.append(clean_line)
                
                # Check if this line starts the recommendations section
                elif "recommendation" in line.lower() and (':' in line or line.startswith('#')):
                    in_recommendations = True
        
        # If no structured recommendations found, use Mistral to extract them
        if not recommendations:
            prompt = f"""
            Extract specific actionable recommendations from this memory analysis:
            
            {analysis_text}
            
            List each recommendation on a separate line without numbering or bullet points.
            """
            
            try:
                response = self.query_mistral(prompt)
                recommendations = [line.strip() for line in response.split('\n') if line.strip()]
            except Exception as e:
                logger.error(f"Error extracting recommendations: {e}")
                recommendations = ["No specific recommendations available"]
        
        return recommendations
    
    def store_analysis(self, query: str, response: str, metadata: Dict[str, Any] = None) -> str:
        try:
            # Create analysis record with flattened agent info to avoid nested dictionaries
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "agent_name": self.name,
                "agent_role": self.role
            }
            
            # Add flattened metadata (avoid nested dictionaries)
            flattened_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (dict, list)):
                            flattened_metadata[f"{key}_{subkey}"] = subvalue
                elif not isinstance(value, (dict, list)):
                    flattened_metadata[key] = value
            
            # Merge flattened metadata
            analysis.update(flattened_metadata)
            
            # Generate a unique ID
            analysis_id = f"analysis_{int(time.time())}_{hash(query) % 10000}"
            
            # Create text representation for vector storage
            text_repr = f"Query: {query}\n\nResponse: {response}\n\nTimestamp: {analysis['timestamp']}"
            
            # Add to FAISS vector store
            self.vectorstore.add_documents([
                Document(page_content=text_repr, metadata=analysis)
            ])
            # Save the updated index
            self.vectorstore.save_local(self.vector_db_dir, "index")
            
            # Also add to our RAG pipeline using FAISS
            # Note: This assumes MemoryRAGPipeline has been updated to use FAISS
            self.agent_intelligence.add_log_to_vectordb({
                "text": text_repr,
                "metadata": analysis,
                "id": analysis_id
            })
            
            self.logger.info(f"Stored analysis with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            self.logger.error(f"Error storing analysis: {str(e)}")
            return f"error_{hash(query) % 10000}"
    
    def query_memory_history(self, question: str) -> Dict[str, Any]:
        """
        Query the memory history with a natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with query results
        """
        # Get similar logs from the memory intelligence
        similar_logs = self.memory_intelligence.get_similar_logs(question)
        
        # Format logs for the model
        formatted_logs = json.dumps(similar_logs, indent=2)
        
        # Create prompt
        prompt = f"""
        You are the Memory Sentinel, analyzing memory history.
        
        Question: {question}
        
        Similar memory logs:
        {formatted_logs}
        
        Based on these logs, provide a detailed analysis that answers the question.
        Focus on patterns, trends, and potential issues.
        """
        
        # Query Mistral
        analysis = self.query_mistral(prompt)
        
        # Return results
        return {
            "question": question,
            "similar_logs": similar_logs,
            "analysis": analysis
        }
    
    def monitor_memory(self, interval: int = 300):
        """
        Continuously monitor memory conditions.
        
        Args:
            interval: Interval between checks in seconds
        """
        logger.info(f"Starting memory monitoring (interval: {interval} seconds)")
        
        while not self.stop_event.is_set():
            try:
                # Get current memory stats
                stats = self.retrieve_memory_stats()
                
                # Analyze memory condition
                analysis = self.analyze_memory_condition(stats)
                
                # Check for issues
                issues = analysis.get('issues', {})
                if issues:
                    logger.info(f"\n=== MEMORY ISSUES DETECTED ({len(issues)}) ===")
                    for issue_type, issue in issues.items():
                        severity = issue.get('severity', 'info').upper()
                        description = issue.get('description', 'No description')
                        recommendation = issue.get('recommendation', 'No recommendation')
                        
                        logger.info(f"[{severity}] {description}")
                        logger.info(f"Recommendation: {recommendation}")
                else:
                    logger.info(f"\n=== MEMORY CONDITION NORMAL ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
                
                # Wait for next check
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(10)  # Wait a bit before retrying
    
    def start_monitoring(self, interval: int = 300):
        with threading.Lock():
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.logger.warning("Monitoring already running")
                return
                
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self.monitor_memory,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info(f"Memory monitoring started with interval: {interval} seconds")
    
    def stop(self):
        """
        Stop memory monitoring.
        """
        self.stop_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        logger.info("Memory monitoring stopped")

# Example usage
if __name__ == "__main__":
    # Get Mistral API key from environment variable with proper fallbacks
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Try to get monitor-specific key first
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY_MONITOR")
    
    if not MISTRAL_API_KEY:
        print("ERROR: No Mistral API key found for monitor agent. Please set MISTRAL_API_KEY_MONITOR environment variable.")
        exit(1)
    
    try:
        # Initialize the Memory Monitor Agent
        agent = MemoryMonitorAgent(
            mistral_api_key=MISTRAL_API_KEY
        )
        
        # Process any existing logs
        agent.memory_intelligence.process_batch_logs()
        
        # Analyze current memory condition
        print("\n=== MEMORY CONDITION ANALYSIS ===")
        stats = agent.retrieve_memory_stats()
        analysis = agent.analyze_memory_condition(stats)
        print(analysis['analysis'])
        
        print("\n=== OOM RISK ANALYSIS ===")
        print(analysis['oom_analysis'])
        
        print("\n=== FRAGMENTATION ANALYSIS ===")
        print(analysis['fragmentation_analysis'])
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Example query
        print("\n=== MEMORY HISTORY QUERY ===")
        query_result = agent.query_memory_history("Is there a risk of memory fragmentation based on current patterns?")
        print(query_result['analysis'])
        
        # Start monitoring
        agent.start_monitoring(interval=300)  # Check every 5 minutes
        
        # Keep running until interrupted
        print("\nAgent running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping Memory Monitor Agent...")
    finally:
        if 'agent' in locals():
            agent.stop()


def get_dashboard_stats(self, timeframe: str = "all") -> Dict[str, Any]:
    """
    Get formatted memory analysis statistics for dashboard display.
    
    Args:
        timeframe: Filter analyses by timeframe ("24 hours", "7 days", "30 days", or "all")
        
    Returns:
        Dictionary with formatted analysis statistics for dashboard
    """
    try:
        # Query for analyses based on timeframe
        if timeframe == "all":
            query = "memory analyses"
        else:
            query = f"memory analyses from the last {timeframe}"
        
        # Get relevant analyses
        docs = self.retriever.get_relevant_documents(query)
        
        # Extract and format analysis data
        analyses = []
        for doc in docs:
            metadata = doc.metadata
            
            # Skip if not within timeframe
            if timeframe != "all":
                try:
                    timestamp = datetime.fromisoformat(metadata.get("timestamp", "").replace('Z', '+00:00'))
                    now = datetime.now()
                    
                    if timeframe == "24 hours" and (now - timestamp).total_seconds() > 86400:
                        continue
                    elif timeframe == "7 days" and (now - timestamp).total_seconds() > 604800:
                        continue
                    elif timeframe == "30 days" and (now - timestamp).total_seconds() > 2592000:
                        continue
                except (ValueError, TypeError):
                    # If timestamp parsing fails, include the analysis anyway
                    pass
            
            # Extract key information
            analysis_data = {
                "timestamp": metadata.get("timestamp"),
                "analysis_id": metadata.get("analysis_id", f"unknown_{len(analyses)}"),
                "free_memory_percent": metadata.get("system_metrics_free_memory_percent"),
                "fragmentation_index": metadata.get("system_metrics_fragmentation_index"),
            }
            
            # Extract issues from content
            content = doc.page_content
            
            # Determine memory health status
            if "critical" in content.lower() or "severe" in content.lower():
                analysis_data["health_status"] = "critical"
            elif "warning" in content.lower() or "moderate" in content.lower():
                analysis_data["health_status"] = "warning"
            elif "healthy" in content.lower() or "normal" in content.lower():
                analysis_data["health_status"] = "healthy"
            else:
                analysis_data["health_status"] = "unknown"
            
            # Extract issue count
            import re
            issue_match = re.search(r'(\d+)\s+issues?\s+detected', content, re.IGNORECASE)
            if issue_match:
                analysis_data["issue_count"] = int(issue_match.group(1))
            else:
                # Count occurrences of common issue keywords
                issue_keywords = ["risk", "warning", "critical", "issue", "problem", "leak", "fragmentation", "oom"]
                issue_count = sum(1 for keyword in issue_keywords if keyword in content.lower())
                analysis_data["issue_count"] = min(issue_count, 10)  # Cap at 10 to avoid overestimation
            
            # Extract recommendation count
            rec_match = re.search(r'(\d+)\s+recommendations?', content, re.IGNORECASE)
            if rec_match:
                analysis_data["recommendation_count"] = int(rec_match.group(1))
            else:
                # Count recommendation markers
                rec_markers = ["recommend", "should", "consider", "advised", "suggested"]
                rec_count = sum(1 for marker in rec_markers if marker in content.lower())
                analysis_data["recommendation_count"] = min(rec_count, 5)  # Cap at 5
            
            analyses.append(analysis_data)
        
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Calculate summary statistics
        summary = {
            "total_analyses": len(analyses),
            "critical_count": sum(1 for a in analyses if a.get("health_status") == "critical"),
            "warning_count": sum(1 for a in analyses if a.get("health_status") == "warning"),
            "healthy_count": sum(1 for a in analyses if a.get("health_status") == "healthy"),
            "avg_free_memory": None,
            "avg_fragmentation": None,
            "total_issues": sum(a.get("issue_count", 0) for a in analyses),
            "total_recommendations": sum(a.get("recommendation_count", 0) for a in analyses)
        }
        
        # Calculate averages (excluding None values)
        free_memory_values = [a.get("free_memory_percent") for a in analyses if a.get("free_memory_percent") is not None]
        if free_memory_values:
            summary["avg_free_memory"] = sum(free_memory_values) / len(free_memory_values)
        
        frag_values = [a.get("fragmentation_index") for a in analyses if a.get("fragmentation_index") is not None]
        if frag_values:
            summary["avg_fragmentation"] = sum(frag_values) / len(frag_values)
        
        # Format the final dashboard data
        dashboard_data = {
            "summary": summary,
            "recent_analyses": analyses[:10],  # Only include the 10 most recent analyses
            "analysis_trends": self._calculate_analysis_trends(analyses)
        }
        
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Error generating dashboard stats: {e}")
        return {
            "error": str(e),
            "summary": {"total_analyses": 0},
            "recent_analyses": [],
            "analysis_trends": {}
        }

def _calculate_analysis_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate trends from analysis data for dashboard visualization.
    
    Args:
        analyses: List of analysis data
        
    Returns:
        Dictionary with trend data
    """
    # Skip if no analyses
    if not analyses:
        return {}
    
    # Group analyses by day
    from datetime import datetime
    from collections import defaultdict
    
    daily_data = defaultdict(lambda: {
        "count": 0, 
        "critical": 0, 
        "warning": 0, 
        "healthy": 0,
        "free_memory_sum": 0,
        "free_memory_count": 0,
        "fragmentation_sum": 0,
        "fragmentation_count": 0
    })
    
    for analysis in analyses:
        try:
            # Parse timestamp
            timestamp = analysis.get("timestamp", "")
            if not timestamp:
                continue
                
            date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_str = date_obj.strftime("%Y-%m-%d")
            
            # Update daily counts
            daily_data[date_str]["count"] += 1
            
            # Update health status counts
            health_status = analysis.get("health_status", "unknown")
            if health_status == "critical":
                daily_data[date_str]["critical"] += 1
            elif health_status == "warning":
                daily_data[date_str]["warning"] += 1
            elif health_status == "healthy":
                daily_data[date_str]["healthy"] += 1
            
            # Update memory metrics
            free_memory = analysis.get("free_memory_percent")
            if free_memory is not None:
                daily_data[date_str]["free_memory_sum"] += free_memory
                daily_data[date_str]["free_memory_count"] += 1
            
            fragmentation = analysis.get("fragmentation_index")
            if fragmentation is not None:
                daily_data[date_str]["fragmentation_sum"] += fragmentation
                daily_data[date_str]["fragmentation_count"] += 1
                
        except Exception as e:
            logger.error(f"Error processing analysis for trends: {e}")
            continue
    
    # Convert to lists for charting
    dates = sorted(daily_data.keys())
    trends = {
        "dates": dates,
        "total_counts": [daily_data[date]["count"] for date in dates],
        "critical_counts": [daily_data[date]["critical"] for date in dates],
        "warning_counts": [daily_data[date]["warning"] for date in dates],
        "healthy_counts": [daily_data[date]["healthy"] for date in dates],
        "avg_free_memory": [
            daily_data[date]["free_memory_sum"] / daily_data[date]["free_memory_count"] 
            if daily_data[date]["free_memory_count"] > 0 else None 
            for date in dates
        ],
        "avg_fragmentation": [
            daily_data[date]["fragmentation_sum"] / daily_data[date]["fragmentation_count"]
            if daily_data[date]["fragmentation_count"] > 0 else None
            for date in dates
        ]
    }
    
    return trends