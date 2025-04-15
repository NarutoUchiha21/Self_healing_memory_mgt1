import os
import time
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from crewai import Crew, Agent, Task, Process
from dotenv import load_dotenv

# Import all four agent implementations
from monitor_agent import MemoryMonitorAgent
from healer_agent import MemoryHealerAgent
from predictor_agent import MemoryPredictorAgent
from explainer_agent import MemoryExplainerAgent

# Import memory engine for direct data access
import monitoring_engine

# Load environment variables
load_dotenv()

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("d:/clg/COA/Self_healing_memory/logs/crew_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crew_manager")

class MemoryCrewManager:
    """
    Memory Crew Manager - Coordinates the four memory agents using CrewAI.
    Implements workflows for memory analysis, emergency response, and prediction.
    """
    
    def __init__(
        self,
        monitor_api_key: Optional[str] = None,
        healer_api_key: Optional[str] = None,
        predictor_api_key: Optional[str] = None,
        explainer_api_key: Optional[str] = None
    ):
        """
        Initialize the Memory Crew Manager.
        
        Args:
            monitor_api_key: API key for Monitor Agent
            healer_api_key: API key for Healer Agent
            predictor_api_key: API key for Predictor Agent
            explainer_api_key: API key for Explainer Agent
        """
        # Initialize all agents
        self.monitor_agent_impl = MemoryMonitorAgent(
            mistral_api_key=monitor_api_key or os.environ.get("MISTRAL_API_KEY_MONITOR")
        )
        
        self.healer_agent_impl = MemoryHealerAgent(
            mistral_api_key=healer_api_key or os.environ.get("MISTRAL_API_KEY_HEALER")
        )
        
        self.predictor_agent_impl = MemoryPredictorAgent(
            mistral_api_key=predictor_api_key or os.environ.get("MISTRAL_API_KEY_PREDICTOR")
        )
        
        self.explainer_agent_impl = MemoryExplainerAgent(
            mistral_api_key=explainer_api_key or os.environ.get("MISTRAL_API_KEY_EXPLAINER")
        )
        
        # Create CrewAI agents
        self.monitor_agent = self._create_monitor_agent()
        self.healer_agent = self._create_healer_agent()
        self.predictor_agent = self._create_predictor_agent()
        self.explainer_agent = self._create_explainer_agent()
        
        logger.info("Memory Crew Manager initialized with all agents")
    
    def _create_monitor_agent(self) -> Agent:
        """Create the Monitor Agent for CrewAI."""
        from langchain_mistralai.chat_models import ChatMistralAI
        
        return Agent(
            name="Memory Monitor",
            role="Memory System Analyst",
            goal="Analyze system memory patterns and identify anomalies",
            backstory=getattr(self.monitor_agent_impl, 'backstory', 
                     "I am a specialized agent that monitors memory systems and identifies issues."),
            verbose=True,
            allow_delegation=True,
            llm=ChatMistralAI(
                mistral_api_key=os.environ.get("MISTRAL_API_KEY_MONITOR"),
                model="mistral-small"
            )
        )
    
    def _create_healer_agent(self) -> Agent:
        """Create the Healer Agent for CrewAI."""
        from langchain_mistralai.chat_models import ChatMistralAI
        
        return Agent(
            name="Memory Healer",
            role="Memory Optimization Specialist",
            goal="Resolve memory issues and optimize system performance",
            backstory=getattr(self.healer_agent_impl, 'backstory', 
                     "I am a specialized agent that fixes memory issues and optimizes performance."),
            verbose=True,
            allow_delegation=True,
            llm=ChatMistralAI(
                mistral_api_key=os.environ.get("MISTRAL_API_KEY_HEALER"),
                model="mistral-small"
            )
        )
    
    def _create_predictor_agent(self) -> Agent:
        """Create the Predictor Agent for CrewAI."""
        from langchain_mistralai.chat_models import ChatMistralAI
        
        return Agent(
            name="Memory Predictor",
            role="Memory Trend Forecaster",
            goal="Predict future memory patterns and potential issues",
            backstory=getattr(self.predictor_agent_impl, 'backstory', 
                     "I am a specialized agent that predicts future memory trends and potential issues."),
            verbose=True,
            allow_delegation=True,
            llm=ChatMistralAI(
                mistral_api_key=os.environ.get("MISTRAL_API_KEY_PREDICTOR"),
                model="mistral-small"
            )
        )
    
    def _create_explainer_agent(self) -> Agent:
        """Create the Explainer Agent for CrewAI."""
        from langchain_mistralai.chat_models import ChatMistralAI
        
        return Agent(
            name="Memory Interpreter",
            role="Technical Translator and Memory Communicator",
            goal="Translate technical memory information into user-friendly explanations",
            backstory=getattr(self.explainer_agent_impl, 'backstory', 
                     "I am a specialized agent that explains complex memory concepts in simple terms."),
            verbose=True,
            allow_delegation=True,
            llm=ChatMistralAI(
                mistral_api_key=os.environ.get("MISTRAL_API_KEY_EXPLAINER"),
                model="mistral-small"
            )
        )
    
    def memory_analysis_process(self, user_query: str = None) -> Dict[str, Any]:
        """
        Run a full memory analysis process with all agents.
        
        Args:
            user_query: Optional user query to guide the analysis
            
        Returns:
            Dictionary with analysis results from all agents
        """
        logger.info(f"Starting memory analysis process with query: {user_query}")
        
        try:
            # Create tasks for the analysis process
            monitor_task = Task(
                description="Analyze current memory status and identify any issues",
                agent=self.monitor_agent
            )
            
            predictor_task = Task(
                description="Predict future memory trends based on current analysis",
                agent=self.predictor_agent,
                context=[monitor_task]  # Depends on monitor task
            )
            
            healer_task = Task(
                description="Generate healing plan for identified issues",
                agent=self.healer_agent,
                context=[monitor_task, predictor_task]  # Depends on monitor and predictor tasks
            )
            
            explainer_task = Task(
                description="Explain the analysis, predictions, and healing plan in user-friendly terms",
                agent=self.explainer_agent,
                context=[monitor_task, predictor_task, healer_task]  # Depends on all previous tasks
            )
            
            # Create a crew for this analysis
            analysis_crew = Crew(
                agents=[self.monitor_agent, self.predictor_agent, self.healer_agent, self.explainer_agent],
                tasks=[monitor_task, predictor_task, healer_task, explainer_task],
                verbose=True,
                process=Process.sequential  # Sequential for full analysis
            )
            
            # Run the crew
            result = analysis_crew.kickoff()
            
            # Get the actual results from our agent implementations
            memory_stats = monitoring_engine.get_memory_stats()
            analysis = self.monitor_agent_impl.analyze_memory(memory_stats)
            prediction = self.predictor_agent_impl.predict_memory_trends(memory_stats)
            healing_plan = self.healer_agent_impl.generate_healing_plan(analysis)
            
            combined_data = {
                "analysis": analysis,
                "prediction": prediction,
                "healing_plan": healing_plan
            }
            explanation = self.explainer_agent_impl.explain_memory_data(combined_data, audience="general")
            
            # Structure the results
            structured_result = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "analysis": analysis,
                "prediction": prediction,
                "healing_plan": healing_plan,
                "explanation": explanation,
                "crew_result": result
            }
            
            logger.info("Memory analysis process completed successfully")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in memory analysis process: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "error": str(e)
            }
    
    def emergency_response_process(self, issue_description: str) -> Dict[str, Any]:
        """
        Run an emergency response process for critical memory issues.
        
        Args:
            issue_description: Description of the critical issue
            
        Returns:
            Dictionary with emergency response results
        """
        logger.info(f"Starting emergency response process for issue: {issue_description}")
        
        try:
            # Create tasks for the emergency process
            monitor_task = Task(
                description="Quickly analyze the critical memory issue",
                agent=self.monitor_agent
            )
            
            healer_task = Task(
                description="Generate and execute immediate healing actions",
                agent=self.healer_agent
            )
            
            explainer_task = Task(
                description="Explain the issue and actions taken in simple terms",
                agent=self.explainer_agent
            )
            
            # Create a crew for this emergency
            emergency_crew = Crew(
                agents=[self.monitor_agent, self.healer_agent, self.explainer_agent],
                tasks=[monitor_task, healer_task, explainer_task],
                verbose=True,
                process=Process.parallel  # Parallel for emergency response
            )
            
            # Run the crew
            result = emergency_crew.kickoff()
            
            # Get the actual results from our agent implementations
            memory_stats = monitoring_engine.get_memory_stats()
            analysis = self.monitor_agent_impl.analyze_memory(memory_stats, emergency=True)
            healing_actions = self.healer_agent_impl.generate_healing_plan(analysis, emergency=True)
            healing_results = self.healer_agent_impl.execute_healing_action(healing_actions)
            
            combined_data = {
                "issue": issue_description,
                "analysis": analysis,
                "healing_actions": healing_actions,
                "healing_results": healing_results
            }
            explanation = self.explainer_agent_impl.explain_memory_data(combined_data, audience="general")
            
            # Structure the results
            structured_result = {
                "timestamp": datetime.now().isoformat(),
                "issue": issue_description,
                "analysis": analysis,
                "healing_actions": healing_actions,
                "healing_results": healing_results,
                "explanation": explanation,
                "crew_result": result
            }
            
            logger.info("Emergency response process completed successfully")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in emergency response process: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "issue": issue_description,
                "error": str(e)
            }
    
    def prediction_process(self, timeframe: str = "24h", focus_area: str = None) -> Dict[str, Any]:
        """
        Run a prediction-focused process for future memory analysis.
        
        Args:
            timeframe: Timeframe for predictions (e.g., "24h", "7d", "30d")
            focus_area: Optional specific area to focus predictions on
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Starting prediction process for timeframe: {timeframe}, focus: {focus_area}")
        
        try:
            # Create tasks for the prediction process
            monitor_task = Task(
                description="Analyze current memory status as baseline for predictions",
                agent=self.monitor_agent
            )
            
            predictor_task = Task(
                description=f"Predict memory trends for the next {timeframe}" + 
                           (f" focusing on {focus_area}" if focus_area else ""),
                agent=self.predictor_agent,
                context=[monitor_task]  # Depends on monitor task
            )
            
            explainer_task = Task(
                description="Explain the predictions and their implications in user-friendly terms",
                agent=self.explainer_agent,
                context=[monitor_task, predictor_task]  # Depends on monitor and predictor tasks
            )
            
            # Create a crew for this prediction
            prediction_crew = Crew(
                agents=[self.monitor_agent, self.predictor_agent, self.explainer_agent],
                tasks=[monitor_task, predictor_task, explainer_task],
                verbose=True,
                process=Process.sequential  # Sequential for prediction
            )
            
            # Run the crew
            result = prediction_crew.kickoff()
            
            # Get the actual results from our agent implementations
            memory_stats = monitoring_engine.get_memory_stats()
            baseline = self.monitor_agent_impl.analyze_memory(memory_stats)
            prediction = self.predictor_agent_impl.predict_memory_trends(
                memory_stats, 
                timeframe=timeframe, 
                focus_area=focus_area
            )
            
            combined_data = {
                "baseline": baseline,
                "prediction": prediction,
                "timeframe": timeframe,
                "focus_area": focus_area
            }
            explanation = self.explainer_agent_impl.explain_memory_data(combined_data, audience="general")
            
            # Structure the results
            structured_result = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "focus_area": focus_area,
                "baseline": baseline,
                "prediction": prediction,
                "explanation": explanation,
                "crew_result": result
            }
            
            logger.info("Prediction process completed successfully")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in prediction process: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "focus_area": focus_area,
                "error": str(e)
            }
    
    def user_query_process(self, query: str, audience: str = "general") -> Dict[str, Any]:
        """
        Process a user query by routing to the appropriate agent(s).
        
        Args:
            query: User query about memory
            audience: Target audience level (general, technical, beginner)
            
        Returns:
            Dictionary with response to the user query
        """
        logger.info(f"Processing user query: {query} for audience: {audience}")
        
        try:
            # Determine which agent should handle the query
            if any(keyword in query.lower() for keyword in ["predict", "future", "trend", "forecast"]):
                # Prediction-focused query
                predictor_task = Task(
                    description=f"Answer user query about future memory trends: {query}",
                    agent=self.predictor_agent
                )
                
                explainer_task = Task(
                    description=f"Explain the prediction in {audience}-friendly terms",
                    agent=self.explainer_agent,
                    context=[predictor_task]
                )
                
                # Create a temporary crew for this query
                query_crew = Crew(
                    agents=[self.predictor_agent, self.explainer_agent],
                    tasks=[predictor_task, explainer_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                result = query_crew.kickoff()
                
                # Get the actual results from our agent implementations
                memory_stats = monitoring_engine.get_memory_stats()
                prediction = self.predictor_agent_impl.predict_memory_trends(memory_stats)
                explanation = self.explainer_agent_impl.translate_recommendation(
                    json.dumps(prediction), 
                    audience=audience
                )
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "audience": audience,
                    "prediction": prediction,
                    "explanation": explanation,
                    "crew_result": result
                }
                
            elif any(keyword in query.lower() for keyword in ["fix", "heal", "solve", "optimize", "improve"]):
                # Healing-focused query
                monitor_task = Task(
                    description=f"Analyze current memory status related to query: {query}",
                    agent=self.monitor_agent
                )
                
                healer_task = Task(
                    description=f"Generate healing plan for user query: {query}",
                    agent=self.healer_agent,
                    context=[monitor_task]
                )
                
                explainer_task = Task(
                    description=f"Explain the healing plan in {audience}-friendly terms",
                    agent=self.explainer_agent,
                    context=[monitor_task, healer_task]
                )
                
                # Create a temporary crew for this query
                query_crew = Crew(
                    agents=[self.monitor_agent, self.healer_agent, self.explainer_agent],
                    tasks=[monitor_task, healer_task, explainer_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                result = query_crew.kickoff()
                
                # Get the actual results from our agent implementations
                memory_stats = monitoring_engine.get_memory_stats()
                analysis = self.monitor_agent_impl.analyze_memory(memory_stats)
                healing_plan = self.healer_agent_impl.generate_healing_plan(analysis)
                explanation = self.explainer_agent_impl.translate_recommendation(
                    json.dumps(healing_plan),
                    audience=audience
                )
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "audience": audience,
                    "analysis": analysis,
                    "healing_plan": healing_plan,
                    "explanation": explanation,
                    "crew_result": result
                }
                
            elif any(keyword in query.lower() for keyword in ["explain", "what is", "how does", "mean"]):
                # Explanation-focused query
                explainer_task = Task(
                    description=f"Explain the memory concept in user query: {query}",
                    agent=self.explainer_agent
                )
                
                # Create a temporary crew for this query
                query_crew = Crew(
                    agents=[self.explainer_agent],
                    tasks=[explainer_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                result = query_crew.kickoff()
                
                # Get the actual results from our agent implementation
                explanation = self.explainer_agent_impl.explain_technical_term(query, audience=audience)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "audience": audience,
                    "explanation": explanation,
                    "crew_result": result
                }
                
            else:
                # General analysis query
                monitor_task = Task(
                    description=f"Analyze current memory status related to query: {query}",
                    agent=self.monitor_agent
                )
                
                explainer_task = Task(
                    description=f"Explain the analysis in {audience}-friendly terms",
                    agent=self.explainer_agent,
                    context=[monitor_task]
                )
                
                # Create a temporary crew for this query
                query_crew = Crew(
                    agents=[self.monitor_agent, self.explainer_agent],
                    tasks=[monitor_task, explainer_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                result = query_crew.kickoff()
                
                # Get the actual results from our agent implementations
                memory_stats = monitoring_engine.get_memory_stats()
                analysis = self.monitor_agent_impl.analyze_memory(memory_stats)
                explanation = self.explainer_agent_impl.explain_memory_data(
                    analysis,
                    audience=audience
                )
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "audience": audience,
                    "analysis": analysis,
                    "explanation": explanation,
                    "crew_result": result
                }
                
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "audience": audience,
                "error": str(e)
            }


def run_interactive_mode():
    """
    Run the Memory Crew Manager in interactive mode, accepting user queries.
    """
    crew_manager = MemoryCrewManager()
    print("\n=== Memory Management System Interactive Mode ===")
    print("Type 'exit' or 'quit' to end the session")
    print("Available commands:")
    print("  analyze - Run a full memory analysis")
    print("  emergency [issue] - Handle an emergency memory issue")
    print("  predict [timeframe] [focus] - Predict future memory trends")
    print("  query [question] - Ask a specific question about memory")
    print("  help - Show this help message")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                break
                
            if user_input.lower() == 'help':
                print("Available commands:")
                print("  analyze - Run a full memory analysis")
                print("  emergency [issue] - Handle an emergency memory issue")
                print("  predict [timeframe] [focus] - Predict future memory trends")
                print("  query [question] - Ask a specific question about memory")
                print("  help - Show this help message")
                continue
                
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            
            if command == 'analyze':
                query = parts[1] if len(parts) > 1 else None
                result = crew_manager.memory_analysis_process(user_query=query)
                print("\n=== ANALYSIS RESULTS ===")
                print(f"Explanation: {result.get('explanation', 'No explanation available')}")
                print("\nWould you like to see the full technical details? (yes/no)")
                if input().lower().startswith('y'):
                    print(json.dumps(result, indent=2))
                    
            elif command == 'emergency':
                if len(parts) > 1:
                    issue = parts[1]
                    result = crew_manager.emergency_response_process(issue_description=issue)
                    print("\n=== EMERGENCY RESPONSE ===")
                    print(f"Explanation: {result.get('explanation', 'No explanation available')}")
                    print("\nWould you like to see the full technical details? (yes/no)")
                    if input().lower().startswith('y'):
                        print(json.dumps(result, indent=2))
                else:
                    print("Please provide an issue description. Example: emergency System memory usage spiked to 95%")
                    
            elif command == 'predict':
                remaining = parts[1] if len(parts) > 1 else ""
                predict_parts = remaining.split(maxsplit=1)
                
                timeframe = predict_parts[0] if predict_parts else "24h"
                focus = predict_parts[1] if len(predict_parts) > 1 else None
                
                result = crew_manager.prediction_process(timeframe=timeframe, focus_area=focus)
                print("\n=== PREDICTION RESULTS ===")
                print(f"Explanation: {result.get('explanation', 'No explanation available')}")
                print("\nWould you like to see the full technical details? (yes/no)")
                if input().lower().startswith('y'):
                    print(json.dumps(result, indent=2))
                    
            elif command == 'query':
                if len(parts) > 1:
                    query = parts[1]
                    print("\nSelect audience level: (1) beginner, (2) general, (3) technical")
                    audience_choice = input("Choice (default is general): ").strip()
                    
                    audience = "general"
                    if audience_choice == "1":
                        audience = "beginner"
                    elif audience_choice == "3":
                        audience = "technical"
                        
                    result = crew_manager.user_query_process(query=query, audience=audience)
                    print("\n=== QUERY RESPONSE ===")
                    print(f"Explanation: {result.get('explanation', 'No explanation available')}")
                    print("\nWould you like to see the full technical details? (yes/no)")
                    if input().lower().startswith('y'):
                        print(json.dumps(result, indent=2))
                else:
                    print("Please provide a query. Example: query What is memory fragmentation?")
                    
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error processing command: {e}")


def parse_command_line():
    """
    Parse command line arguments for the Memory Crew Manager.
    """
    parser = argparse.ArgumentParser(description="Memory Management System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run a full memory analysis")
    analyze_parser.add_argument("--query", type=str, help="Optional query to guide the analysis")
    
    # Emergency command
    emergency_parser = subparsers.add_parser("emergency", help="Handle an emergency memory issue")
    emergency_parser.add_argument("issue", type=str, help="Description of the critical issue")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict future memory trends")
    predict_parser.add_argument("--timeframe", type=str, default="24h", 
                               help="Timeframe for predictions (e.g., 24h, 7d, 30d)")
    predict_parser.add_argument("--focus", type=str, help="Specific area to focus predictions on")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a specific question about memory")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.add_argument("--audience", type=str, default="general", 
                             choices=["beginner", "general", "technical"],
                             help="Target audience level")
    
    return parser.parse_args()


# Example usage
# Replace the existing if __name__ == "__main__" block with this:

if __name__ == "__main__":
    args = parse_command_line()
    crew_manager = MemoryCrewManager()
    
    if args.command == "interactive" or args.command is None:
        # Run in interactive mode if specified or if no command is given
        run_interactive_mode()
        
    elif args.command == "analyze":
        # Run a full memory analysis
        result = crew_manager.memory_analysis_process(user_query=args.query)
        print(json.dumps(result, indent=2))
        
    elif args.command == "emergency":
        # Handle an emergency memory issue
        result = crew_manager.emergency_response_process(issue_description=args.issue)
        print(json.dumps(result, indent=2))
        
    elif args.command == "predict":
        # Predict future memory trends
        result = crew_manager.prediction_process(
            timeframe=args.timeframe,
            focus_area=args.focus
        )
        print(json.dumps(result, indent=2))
        
    elif args.command == "query":
        # Process a user query
        result = crew_manager.user_query_process(
            query=args.question,
            audience=args.audience
        )
        print(json.dumps(result, indent=2))