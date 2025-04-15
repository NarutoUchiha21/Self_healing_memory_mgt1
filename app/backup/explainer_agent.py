import os
import time
import json
import requests
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
# Update imports to use langchain_community instead of langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
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

# Set default API key for explainer agent
MISTRAL_API_KEY_EXPLAINER = os.environ.get("MISTRAL_API_KEY_EXPLAINER", "")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("d:/clg/COA/Self_healing_memory/logs/memory_explainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_explainer")

class MemoryExplainerAgent:
    """
    Memory Explainer Agent - Specialized in translating technical memory information into user-friendly explanations.
    Uses RAG with technical memory data to generate clear, accessible explanations.
    """
    
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        mistral_api_url: str = "https://api.mistral.ai/v1/chat/completions",
        collection_name: str = "memory_explanations",
        agent_collection_name: str = "agent_analyses",
        log_path: str = "d:/clg/COA/Self_healing_memory/data/memory_events.jsonl"):
        """
        Initialize the Memory Explainer Agent.
        
        Args:
            mistral_api_key: API key for Mistral 7B
            mistral_api_url: URL for Mistral API
            collection_name: Name of the memory explanations collection
            agent_collection_name: Name of the agent analyses collection
            log_path: Path to the memory log file
        """
        # Agent identity
        self.name = "Memory Interpreter"
        self.role = "Technical Translator and Memory Communicator"
        self.backstory = """I am the Memory Interpreter, a bridge between complex memory systems 
        and the humans who use them. Born from the need to make technical memory concepts accessible 
        to everyone, I transform intricate memory analyses into clear, actionable explanations.
        
        My purpose is to demystify memory management, translating technical jargon into plain language 
        without sacrificing accuracy. I help users understand what's happening in their system's memory, 
        why it matters, and what they can do about it.
        
        As an interpreter, I don't just simplify - I illuminate. I provide context, analogies, and 
        practical examples that make memory concepts relatable. My explanations empower users to make 
        informed decisions about their system's memory health."""
        
        # Set API key for Mistral - use the explainer-specific key only
        self.api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY_EXPLAINER")
        if not self.api_key:
            logger.error("No Mistral API key available for explainer agent")
            raise ValueError("Mistral API key is required for the explainer agent. Set MISTRAL_API_KEY_EXPLAINER environment variable.")
        
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
        self.cache_dir = "d:/clg/COA/Self_healing_memory/data/explainer_cache"
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
        logger.info(f"Memory Explainer Agent initialized with collections: {collection_name}, {agent_collection_name}")
    
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
        
        # Set up vector store for explanations
        self.vector_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_explanations"
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        # Initialize empty vector store if it doesn't exist
        try:
            self.vectorstore = Chroma(
                collection_name="agent_explanations",
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_dir
            )
        except Exception as e:
            logger.error(f"Error initializing Chroma: {e}")
            # Create fresh vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=[Document(page_content="Initial explanation document", metadata={"type": "init"})],
                embedding=self.embeddings,
                collection_name="agent_explanations",
                persist_directory=self.vector_db_dir
            )
        
        # Set up access to other agents' vector stores
        self.monitor_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_analyses"
        self.healer_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\healing_actions"
        self.predictor_db_dir = "d:\\clg\\COA\\Self_healing_memory\\data\\vector_store\\agent_predictions"
        
        # Connect to monitor agent's vector store
        try:
            self.monitor_vectorstore = Chroma(
                collection_name="agent_analyses",
                embedding_function=self.embeddings,
                persist_directory=self.monitor_db_dir
            )
            logger.info("Successfully connected to monitor agent's vector store")
        except Exception as e:
            logger.error(f"Error connecting to monitor agent's vector store: {e}")
            self.monitor_vectorstore = None
        
        # Connect to healer agent's vector store
        try:
            self.healer_vectorstore = Chroma(
                collection_name="healing_actions",
                embedding_function=self.embeddings,
                persist_directory=self.healer_db_dir
            )
            logger.info("Successfully connected to healer agent's vector store")
        except Exception as e:
            logger.error(f"Error connecting to healer agent's vector store: {e}")
            self.healer_vectorstore = None
        
        # Connect to predictor agent's vector store
        try:
            self.predictor_vectorstore = Chroma(
                collection_name="agent_predictions",
                embedding_function=self.embeddings,
                persist_directory=self.predictor_db_dir
            )
            logger.info("Successfully connected to predictor agent's vector store")
        except Exception as e:
            logger.error(f"Error connecting to predictor agent's vector store: {e}")
            self.predictor_vectorstore = None
        
        # Set up retrievers
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Set up retrievers for other agents' vector stores
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
        
        if self.healer_vectorstore:
            self.healer_retriever = self.healer_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            )
        else:
            self.healer_retriever = None
        
        if self.predictor_vectorstore:
            self.predictor_retriever = self.predictor_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            )
        else:
            self.predictor_retriever = None
        
        # Enhanced prompt template for memory explanation
        self.explanation_template = PromptTemplate(
            input_variables=["technical_context", "memory_data", "audience"],
            template="""
            You are the Memory Interpreter, a technical translator who makes memory concepts accessible.
            
            Your task is to translate technical memory information into clear, user-friendly explanations
            tailored to the specified audience.
            
            Technical context from memory analyses:
            {technical_context}
            
            Memory data to explain:
            {memory_data}
            
            Target audience: {audience}
            
            Provide a clear, accessible explanation of the memory situation, focusing on:
            1. What is happening with the system's memory (in plain language)
            2. Why it matters to the user (real-world impact)
            3. What actions the user should consider (in simple steps)
            4. Any technical terms explained with analogies or examples
            
            Your explanation should be conversational and engaging while maintaining technical accuracy.
            Avoid unnecessary jargon, but don't oversimplify to the point of inaccuracy.
            """
        )
        
        # Create a specialized template for technical term explanations
        self.term_template = PromptTemplate(
            input_variables=["term", "context"],
            template="""
            As the Memory Interpreter, explain this technical memory term in user-friendly language:
            
            TECHNICAL TERM: {term}
            
            CONTEXT WHERE IT APPEARS:
            {context}
            
            Provide a clear explanation that:
            1. Defines the term in simple language
            2. Explains why it matters to system performance
            3. Uses a real-world analogy to illustrate the concept
            4. Mentions when a user should be concerned about it
            
            Your explanation should be accessible to non-technical users while remaining accurate.
            """
        )
        
        # Create a specialized template for translating technical recommendations
        self.recommendation_template = PromptTemplate(
            input_variables=["technical_recommendation", "user_context"],
            template="""
            As the Memory Interpreter, translate this technical memory recommendation into user-friendly advice:
            
            TECHNICAL RECOMMENDATION:
            {technical_recommendation}
            
            USER CONTEXT:
            {user_context}
            
            Provide actionable advice that:
            1. Explains what needs to be done in simple terms
            2. Clarifies why this action will help their system
            3. Provides step-by-step instructions if applicable
            4. Mentions any precautions or alternatives
            
            Your advice should be practical and easy to follow for users with limited technical knowledge.
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
        
        Your task is to translate technical memory information into clear, accessible explanations.
        Focus on making complex concepts understandable without sacrificing accuracy.
        Use analogies, examples, and plain language to illuminate memory concepts.
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
    
    def explain_memory_data(self, memory_data: Dict[str, Any], audience: str = "general") -> Dict[str, Any]:
        """
        Explain memory data in user-friendly terms.
        
        Args:
            memory_data: Memory data to explain
            audience: Target audience (general, technical, beginner)
            
        Returns:
            Dictionary with explanation results
        """
        # Format memory data
        memory_data_str = json.dumps(memory_data, indent=2)
        
        # Retrieve technical context from other agents
        technical_context = self._gather_technical_context(memory_data)
        
        # Generate explanation using the explanation template
        explanation_prompt = self.explanation_template.format(
            technical_context=technical_context,
            memory_data=memory_data_str,
            audience=audience
        )
        
        general_explanation = self.query_mistral(explanation_prompt)
        
        # Extract technical terms for detailed explanation
        technical_terms = self._extract_technical_terms(technical_context, memory_data_str)
        
        # Generate explanations for technical terms
        term_explanations = {}
        for term in technical_terms[:5]:  # Limit to top 5 terms
            term_prompt = self.term_template.format(
                term=term,
                context=technical_context
            )
            term_explanations[term] = self.query_mistral(term_prompt)
        
        # Generate actionable recommendations
        recommendation_prompt = self.recommendation_template.format(
            technical_recommendation=self._extract_recommendations(technical_context),
            user_context=f"User with {audience} technical knowledge, concerned about memory usage"
        )
        
        actionable_advice = self.query_mistral(recommendation_prompt)
        
        # Compile all explanations
        explanation_result = {
            "timestamp": datetime.now().isoformat(),
            "system": memory_data.get("system", "unknown"),
            "audience": audience,
            "general_explanation": general_explanation,
            "term_explanations": term_explanations,
            "actionable_advice": actionable_advice
        }
        
        # Store the explanation in the vector store
        self._store_explanation(explanation_result)
        
        return explanation_result
    
    def _gather_technical_context(self, memory_data: Dict[str, Any]) -> str:
        """
        Gather technical context from all available agent vector stores.
        
        Args:
            memory_data: Memory data to use as query context
            
        Returns:
            Combined technical context string
        """
        system_name = memory_data.get("system", "unknown system")
        query = f"Memory analysis and recommendations for {system_name}"
        
        contexts = []
        
        # Get context from monitor agent
        if self.monitor_retriever:
            try:
                monitor_docs = self.monitor_retriever.get_relevant_documents(query)
                if monitor_docs:
                    contexts.append("MONITOR AGENT ANALYSIS:")
                    contexts.append("\n".join([doc.page_content for doc in monitor_docs[:3]]))
                    logger.info(f"Retrieved {len(monitor_docs[:3])} documents from monitor agent")
            except Exception as e:
                logger.error(f"Error retrieving monitor context: {e}")
        
        # Get context from healer agent
        if self.healer_retriever:
            try:
                healer_docs = self.healer_retriever.get_relevant_documents(query)
                if healer_docs:
                    contexts.append("HEALER AGENT RECOMMENDATIONS:")
                    contexts.append("\n".join([doc.page_content for doc in healer_docs[:3]]))
                    logger.info(f"Retrieved {len(healer_docs[:3])} documents from healer agent")
            except Exception as e:
                logger.error(f"Error retrieving healer context: {e}")
        
        # Get context from predictor agent
        if self.predictor_retriever:
            try:
                predictor_docs = self.predictor_retriever.get_relevant_documents(query)
                if predictor_docs:
                    contexts.append("PREDICTOR AGENT FORECASTS:")
                    contexts.append("\n".join([doc.page_content for doc in predictor_docs[:3]]))
                    logger.info(f"Retrieved {len(predictor_docs[:3])} documents from predictor agent")
            except Exception as e:
                logger.error(f"Error retrieving predictor context: {e}")
        
        if not contexts:
            return "No technical context available from other agents."
        
        return "\n\n".join(contexts)
    
    def _extract_technical_terms(self, technical_context: str, memory_data: str) -> List[str]:
        """
        Extract technical terms from context that need explanation.
        
        Args:
            technical_context: Technical context from other agents
            memory_data: Memory data string
            
        Returns:
            List of technical terms
        """
        # Use Mistral to identify technical terms
        prompt = f"""
        Identify the top technical memory management terms in this text that would benefit from explanation to non-technical users.
        
        TEXT:
        {technical_context}
        
        {memory_data}
        
        List only the technical terms, one per line. Focus on memory-specific terminology.
        """
        
        try:
            response = self.query_mistral(prompt)
            terms = [term.strip() for term in response.split('\n') if term.strip()]
            return terms
        except Exception as e:
            logger.error(f"Error extracting technical terms: {e}")
            return ["memory fragmentation", "memory leak", "page fault", "swap space", "memory allocation"]
    
    def _extract_recommendations(self, technical_context: str) -> str:
        """
        Extract technical recommendations from context.
        
        Args:
            technical_context: Technical context from other agents
            
        Returns:
            String of technical recommendations
        """
        # Look for recommendation sections in the context
        recommendations = []
        
        # Simple heuristic to find recommendation sections
        lines = technical_context.split('\n')
        in_recommendation_section = False
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ["recommend", "action", "should", "advised", "suggestion"]):
                in_recommendation_section = True
                recommendations.append(line)
            elif in_recommendation_section and line.strip() and not line.startswith("  "):
                recommendations.append(line)
            elif in_recommendation_section and not line.strip():
                in_recommendation_section = False
        
        if not recommendations:
            # If no recommendations found, ask Mistral to extract them
            prompt = f"""
            Extract any technical recommendations or advised actions from this text:
            
            {technical_context}
            
            List only the recommendations, focusing on actions that should be taken.
            """
            
            try:
                response = self.query_mistral(prompt)
                return response
            except Exception as e:
                logger.error(f"Error extracting recommendations with Mistral: {e}")
                return "No specific recommendations found in the technical context."
        
        return "\n".join(recommendations)
    
    def _store_explanation(self, explanation: Dict[str, Any]) -> None:
        """
        Store an explanation in the vector store.
        
        Args:
            explanation: The explanation to store
        """
        try:
            # Format the explanation as a document
            explanation_text = f"""
            MEMORY EXPLANATION
            Timestamp: {explanation['timestamp']}
            System: {explanation['system']}
            Audience: {explanation['audience']}
            
            GENERAL EXPLANATION:
            {explanation['general_explanation']}
            
            TECHNICAL TERMS EXPLAINED:
            {', '.join(explanation['term_explanations'].keys())}
            
            ACTIONABLE ADVICE:
            {explanation['actionable_advice']}
            """
            
            # Create metadata
            metadata = {
                "timestamp": explanation['timestamp'],
                "system": explanation['system'],
                "audience": explanation['audience'],
                "type": "memory_explanation"
            }
            
            # Create document
            doc = Document(page_content=explanation_text, metadata=metadata)
            
            # Add to vector store
            self.vectorstore.add_documents([doc])
            
            logger.info(f"Stored explanation in vector store with timestamp {explanation['timestamp']}")
        except Exception as e:
            logger.error(f"Error storing explanation in vector store: {e}")
    
    def explain_technical_term(self, term: str, context: str = "") -> str:
        """
        Provide a user-friendly explanation of a technical memory term.
        
        Args:
            term: Technical term to explain
            context: Optional context where the term appears
            
        Returns:
            User-friendly explanation
        """
        term_prompt = self.term_template.format(
            term=term,
            context=context
        )
        
        return self.query_mistral(term_prompt)
    
    def translate_recommendation(self, recommendation: str, audience: str = "general") -> str:
        """
        Translate a technical recommendation into user-friendly advice.
        
        Args:
            recommendation: Technical recommendation to translate
            audience: Target audience (general, technical, beginner)
            
        Returns:
            User-friendly advice
        """
        recommendation_prompt = self.recommendation_template.format(
            technical_recommendation=recommendation,
            user_context=f"User with {audience} technical knowledge, concerned about memory usage"
        )
        
        return self.query_mistral(recommendation_prompt)
    
    def start_monitoring(self, interval: int = 3600) -> None:
        """
        Start the monitoring thread to periodically explain memory data.
        
        Args:
            interval: Interval between explanations in seconds (default: 1 hour)
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
        logger.info(f"Started memory explanation monitoring with interval {interval} seconds")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=10)
            logger.info("Stopped memory explanation monitoring")
        else:
            logger.warning("No monitoring thread is running")
    
    def _monitoring_loop(self, interval: int) -> None:
        """
        Main monitoring loop that runs in a separate thread.
        
        Args:
            interval: Interval between explanations in seconds
        """
        logger.info("Memory explanation monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get current memory stats
                memory_stats = monitoring_engine.get_memory_stats()
                
                # Generate explanations for different audiences
                general_explanation = self.explain_memory_data(memory_stats, "general")
                beginner_explanation = self.explain_memory_data(memory_stats, "beginner")
                technical_explanation = self.explain_memory_data(memory_stats, "technical")
                
                logger.info(f"Generated explanations for system {memory_stats.get('system', 'unknown')}")
                
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
    
    def get_historical_explanations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve historical explanations based on a query.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            
        Returns:
            List of historical explanations
        """
        try:
            docs = self.retriever.get_relevant_documents(query)
            
            results = []
            for doc in docs[:k]:
                # Parse the explanation from the document
                content = doc.page_content
                metadata = doc.metadata
                
                explanation_data = {
                    "timestamp": metadata.get("timestamp"),
                    "system": metadata.get("system"),
                    "audience": metadata.get("audience"),
                    "content": content
                }
                
                results.append(explanation_data)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving historical explanations: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize the explainer agent
    explainer = MemoryExplainerAgent()
    
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
    
    # Generate explanations
    print("\n=== MEMORY EXPLANATION ===")
    explanation = explainer.explain_memory_data(test_stats, "general")
    
    print("\n=== GENERAL EXPLANATION ===")
    print(explanation['general_explanation'])
    
    print("\n=== TECHNICAL TERMS EXPLAINED ===")
    for term, term_explanation in explanation['term_explanations'].items():
        print(f"\n{term}:")
        print(term_explanation)
    
    print("\n=== ACTIONABLE ADVICE ===")
    print(explanation['actionable_advice'])
    
    # Test term explanation
    print("\n=== TECHNICAL TERM EXPLANATION ===")
    term_explanation = explainer.explain_technical_term("memory fragmentation", 
                                                      "The system is showing signs of memory fragmentation at 45%")
    print(term_explanation)
    
    # Test recommendation translation
    print("\n=== RECOMMENDATION TRANSLATION ===")
    recommendation = """Implement memory defragmentation using the system's built-in utilities. 
    Execute 'sysctl -w vm.drop_caches=3' to clear the page cache and initiate garbage collection. 
    Consider terminating processes with PIDs 1234, 5678 to free up 25% of memory resources."""
    
    translation = explainer.translate_recommendation(recommendation, "beginner")
    print(translation)