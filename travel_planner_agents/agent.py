from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import time
import logging
import signal
from contextlib import contextmanager
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_NAME = "travel_planner_app"
USER_ID = "travel_user_01"
SESSION_ID = "travel_session_01"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
MAX_RETRIES = 1
RETRY_DELAY = 5  # seconds

# Constants for state keys
STATE_FLIGHT_OPTIONS = "flight_options"
STATE_HOTEL_OPTIONS = "hotel_options"
STATE_WEATHER_FORECAST = "weather_forecast"
STATE_ITINERARY = "itinerary"
STATE_BUDGET_ANALYSIS = "budget_analysis"
STATE_APPROVAL_STATUS = "approval_status"
STATE_PLAN_TRIP = "plan_trip"
# Model constants
OLLAMA_MODEL = "ollama/gemma3:12b"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Add these constants
WORKFLOW_TIMEOUT = 300  # 5 minutes timeout
MAX_ITERATIONS = 2     # Maximum iterations for loop agent
ITERATION_TIMEOUT = 120 # 60 seconds per iteration

class TimeoutException(Exception):
    """Custom exception for timeout"""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutException("Process timed out")

    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class TravelCoordinator:
    def __init__(self):
        self.stop_flag = threading.Event()
        # Create function tools
        self.search_flights = FunctionTool(func=self._search_flights)
        self.search_hotels = FunctionTool(func=self._search_hotels)
        self.create_itinerary = FunctionTool(func=self._create_itinerary)
        self.check_budget = FunctionTool(func=self._check_budget)
        self.request_human_approval = FunctionTool(func=self._request_human_approval)
        self.check_flight_options = FunctionTool(func=self._check_flight_options)

        # Create specialized agents with direct tools
        self.flight_finder_agent = self._create_flight_finder()
        self.hotel_booker_agent = self._create_hotel_booker()
        self.itinerary_designer_agent = self._create_itinerary_designer()
        self.budget_checker_agent = self._create_budget_checker()
        self.human_approval_agent = self._create_human_approval_tool()

        # Create the workflow
        self.workflow1 = SequentialAgent(
            name="TravelWorkflow",
            sub_agents=[
                self.flight_finder_agent,
                self.hotel_booker_agent,
                self.itinerary_designer_agent,
                self.budget_checker_agent,
                self.human_approval_agent
            ]
        )
        # self.workflow = SequentialAgent(
        #     name="TravelWorkflow",
        #     sub_agents=[
        #         # Phase 1: Data Collection (runs once)
        #         SequentialAgent(
        #             name="InitialDataCollection",
        #             sub_agents=[
        #                 ParallelAgent(
        #                     name="DataCollection",
        #                     sub_agents=[
        #                         self.flight_finder_agent,    # Output stored in STATE_FLIGHT_OPTIONS
        #                         self.hotel_booker_agent,     # Output stored in STATE_HOTEL_OPTIONS
        #                     ]
        #                 )
        #             ]
        #         ),
        #         # Phase 2: Iterative Planning with timeout
        #         SequentialAgent(
        #             name="PlanningPhase",
        #             sub_agents=[
        #                 LoopAgent(
        #                     name="PlanningRefinement",
        #                     sub_agents=[
        #                         self.itinerary_designer_agent,  # Output stored in STATE_ITINERARY
        #                         self.budget_checker_agent       # Output stored in STATE_BUDGET_ANALYSIS
        #                     ],
        #                     max_iterations=MAX_ITERATIONS
        #                 )
        #             ]
        #         ),
        #         # Phase 3: Final Approval
        #         SequentialAgent(
        #             name="ApprovalPhase",
        #             sub_agents=[
        #                 self.human_approval_agent    # Output stored in STATE_APPROVAL_STATUS
        #             ]
        #         )
        #     ]
        # )

        # Initialize session service and runner
        self.session_service = InMemorySessionService()
        self.session = self.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        self.runner = Runner(
            agent=self.workflow,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    def _create_flight_finder(self) -> LlmAgent:
        return LlmAgent(
            name="flight_finder",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Find and compare flight options based on origin, destination, and dates.
            First check if flight options already exist in state key '{STATE_FLIGHT_OPTIONS}'.
            If they exist, return those options.
            If not, search for new flight options.
            Consider price, duration, and layovers.
            Return flight options in a structured format.
            """,
            tools=[self.search_flights, self.check_flight_options],
            output_key=STATE_FLIGHT_OPTIONS
        )

    def _create_hotel_booker(self) -> LlmAgent:
        return LlmAgent(
            name="hotel_booker",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Find and compare hotel options based on location, dates, and budget.
            First check if hotel options already exist in state key '{STATE_HOTEL_OPTIONS}'.
            If they exist, return those options.
            If not, search for new hotel options.
            Consider amenities, ratings, and location.
            Return hotel options in a structured format.
            """,
            tools=[self.search_hotels],
            output_key=STATE_HOTEL_OPTIONS
        )

    def _create_itinerary_designer(self) -> LlmAgent:
        return LlmAgent(
            name="itinerary_designer",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Design a detailed daily itinerary based on available activities and user preferences.
            Use the following data from state:
            - {STATE_FLIGHT_OPTIONS}: Available flights
            - {STATE_HOTEL_OPTIONS}: Available hotels
            Return a structured daily schedule in JSON format.
            """,
            tools=[self.create_itinerary],
            output_key=STATE_ITINERARY
        )

    def _create_budget_checker(self) -> LlmAgent:
        return LlmAgent(
            name="budget_checker",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Verify if the total cost of the trip is within budget.
            Use the following data from state:
            - {STATE_FLIGHT_OPTIONS}: Flight costs
            - {STATE_HOTEL_OPTIONS}: Hotel costs
            - {STATE_ITINERARY}: Activity costs
            Break down costs by category.
            Suggest alternatives if over budget.
            Return a dictionary with 'within_budget' boolean and 'total_cost' float.
            """,
            tools=[self.check_budget],
            output_key=STATE_BUDGET_ANALYSIS
        )

    def _create_human_approval_tool(self) -> LlmAgent:
        return LlmAgent(
            name="human_approval",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Handle cases requiring human approval.
            Use the following data from state:
            - {STATE_BUDGET_ANALYSIS}: Budget compliance status
            - {STATE_ITINERARY}: Proposed itinerary
            Present clear options and reasoning.
            Wait for human input before proceeding.
            Return a dictionary with 'approved' boolean and 'reason' string.
            """,
            tools=[self.request_human_approval],
            output_key=STATE_APPROVAL_STATUS
        )
    
    def plan_trip(self, query: str) -> Dict[str, Any]:
        """Main entry point for trip planning with natural language query"""
        logger.info(f"Planning trip with query: {query}")
        
        # Reset workflow state
        self.reset_workflow()
        
        # Convert query to ADK content format
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        # Set overall timeout
        start_time = datetime.now()
        
        try:
            with timeout(WORKFLOW_TIMEOUT):
                # Execute the workflow with retry logic
                for attempt in range(MAX_RETRIES):
                    try:
                        if self.stop_flag.is_set():
                            return {
                                "status": "stopped",
                                "message": "Workflow execution was stopped"
                            }

                        events = self.runner.run(
                            user_id=USER_ID,
                            session_id=SESSION_ID,
                            new_message=content
                        )

                        # Process events and return final response
                        for event in events:
                            # Check for timeout
                            if (datetime.now() - start_time).total_seconds() > WORKFLOW_TIMEOUT:
                                self.stop_workflow()
                                return {
                                    "status": "timeout",
                                    "message": "Workflow execution timed out"
                                }

                            if event.is_final_response():
                                try:
                                    return json.loads(event.content.parts[0].text)
                                except json.JSONDecodeError:
                                    return {
                                        "status": "success",
                                        "response": event.content.parts[0].text
                                    }

                        return {"status": "error", "message": "No final response received"}

                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                        if attempt < MAX_RETRIES - 1:
                            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        else:
                            return {
                                "status": "error",
                                "message": f"Failed after {MAX_RETRIES} attempts: {str(e)}"
                            }

        except TimeoutException:
            self.stop_workflow()
            return {
                "status": "timeout",
                "message": f"Workflow execution timed out after {WORKFLOW_TIMEOUT} seconds"
            }
        finally:
            self.reset_workflow()

    def stop_workflow(self):
        """Stop the workflow execution"""
        logger.info("Stopping workflow execution...")
        self.stop_flag.set()

    def reset_workflow(self):
        """Reset the workflow state"""
        logger.info("Resetting workflow state...")
        self.stop_flag.clear()

    # Tool implementations
    def _search_flights(self, origin: str, destination: str, dates: Dict[str, str]) -> Dict[str, Any]:
        """Mock implementation for flight search"""
        logger.info(f"Searching flights from {origin} to {destination} for dates {dates}")
        return {
            "status": "success",
            "flights": [
                {
                    "airline": "Sample Airline",
                    "price": 500,
                    "duration": "12h",
                    "stops": 1,
                    "departure": "2024-05-01T08:00:00",
                    "arrival": "2024-05-01T20:00:00",
                    "class": "economy"
                },
                {
                    "airline": "Another Airline",
                    "price": 450,
                    "duration": "10h",
                    "stops": 0,
                    "departure": "2024-05-01T09:00:00",
                    "arrival": "2024-05-01T19:00:00",
                    "class": "economy"
                }
            ]
        }

    def _search_hotels(self, location: str, dates: Dict[str, str], budget: float) -> Dict[str, Any]:
        """Mock implementation for hotel search"""
        logger.info(f"Searching hotels in {location} for dates {dates} with budget {budget}")
        return {
            "status": "success",
            "hotels": [
                {
                    "name": "Sample Hotel",
                    "price": 200,
                    "rating": 4.5,
                    "amenities": ["pool", "gym", "free Wi-Fi"],
                    "location": "Shinjuku, Tokyo",
                    "check_in": "2024-05-01",
                    "check_out": "2024-05-07"
                },
                {
                    "name": "Luxury Hotel",
                    "price": 350,
                    "rating": 5.0,
                    "amenities": ["spa", "restaurant", "bar"],
                    "location": "Shibuya, Tokyo",
                    "check_in": "2024-05-01",
                    "check_out": "2024-05-07"
                }
            ]
        }

    def _create_itinerary(self, flight_options: Dict[str, Any], hotel_options: Dict[str, Any], 
                         preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for itinerary creation"""
        logger.info("Creating itinerary based on flight and hotel options")
        return {
            "status": "success",
            "itinerary": {
                "day_1": {
                    "activities": [
                        {
                            "time": "09:00",
                            "activity": "Visit Senso-ji Temple",
                            "duration": "2h",
                            "cost": 0
                        },
                        {
                            "time": "12:00",
                            "activity": "Lunch at a local sushi restaurant",
                            "duration": "1.5h",
                            "cost": 50
                        },
                        {
                            "time": "14:00",
                            "activity": "Explore Akihabara",
                            "duration": "3h",
                            "cost": 100
                        }
                    ]
                },
                "day_2": {
                    "activities": [
                        {
                            "time": "10:00",
                            "activity": "Shopping in Shibuya",
                            "duration": "3h",
                            "cost": 200
                        },
                        {
                            "time": "13:00",
                            "activity": "Visit Meiji Shrine",
                            "duration": "2h",
                            "cost": 0
                        },
                        {
                            "time": "15:00",
                            "activity": "Relax at Yoyogi Park",
                            "duration": "2h",
                            "cost": 0
                        }
                    ]
                }
            }
        }

    def _check_budget(self, flight_cost: float, hotel_cost: float, 
                     activity_costs: List[float], total_budget: float) -> Dict[str, Any]:
        """Mock implementation for budget checking"""
        logger.info(f"Checking budget with flight cost: {flight_cost}, hotel cost: {hotel_cost}, activity costs: {activity_costs}, total budget: {total_budget}")
        total_cost = flight_cost + hotel_cost + sum(activity_costs)
        return {
            "status": "success",
            "within_budget": total_cost <= total_budget,
            "total_cost": total_cost,
            "breakdown": {
                "flights": flight_cost,
                "hotel": hotel_cost,
                "activities": sum(activity_costs)
            },
            "suggestions": [
                "Consider a cheaper hotel option",
                "Reduce shopping budget",
                "Look for flight deals"
            ] if total_cost > total_budget else []
        }

    def _request_human_approval(self, budget_analysis: Dict[str, Any], 
                              itinerary: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for human approval"""
        logger.info("Requesting human approval for budget overrun")
        return {
            "status": "pending",
            "approved": False,
            "reason": "Budget exceeds the initial estimate",
            "options": [
                {
                    "option": "Reduce hotel budget",
                    "suggested_price": 150
                },
                {
                    "option": "Choose a cheaper flight",
                    "suggested_price": 400
                }
            ],
            "requires_human_approval": True
        }

    def _check_flight_options(self, flight_options: Dict[str, Any]) -> Dict[str, Any]:
        """Check if flight options exist and are valid"""
        logger.info("Checking flight options")
        if not flight_options or not flight_options.get("flights"):
            return {
                "status": "error",
                "message": "No flight options available"
            }
        return {
            "status": "success",
            "message": "Flight options are valid",
            "options": flight_options
        }

# Create the main coordinator instance
travel_coordinator = TravelCoordinator()

# Expose the root agent for ADK CLI
root_agent = travel_coordinator.workflow

# Example usage with timeout handling
def call_travel_planner(request: Dict):
    """
    Example function to call the travel planner with timeout handling.
    """
    try:
        result = travel_coordinator.plan_trip(request)
        if result.get("status") in ["timeout", "stopped"]:
            logger.warning(f"Travel planning {result['status']}: {result['message']}")
        print("Travel Plan:", json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error in travel planning: {str(e)}")
        travel_coordinator.stop_workflow()  # Ensure workflow is stopped on error

# Function to get the city weather
def get_weather(city: str) -> object:
    """
    Function to get the weather information for a given city.
    """
    # API call to a weather service would go here
    api_key = WEATHER_API_KEY
    base_url = WEATHER_API_URL
    params = {
        "key": api_key,
        "q": city,
        "days": 7,
        "aqi": "no"
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "forecast": data["forecast"]["forecastday"]
            }
        return {"status": "error", "message": "Failed to fetch weather data"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Function to get the current time
def get_current_time(city: str) -> object:
    """
    Function to get the current time for a given city.
    """
    # API call to a time service would go here
    api_key = WEATHER_API_KEY
    base_url = WEATHER_API_URL

    try:
        response = requests.get(f"{base_url}?key={api_key}&q={city}")
        data = response.json()
        if "error" in data:
            return f"Error: {data['error']['message']}"
        else:
            tzlocal = data["location"]["tz_id"]
            location = data["location"]["name"]
            region = data["location"]["region"]
            country = data["location"]["country"]
            local_time = data["location"]["localtime"]
            result = f"The current time in {location}, {region}, {country} is {local_time}. Timezone: {tzlocal}."
            return {
                "status": "success",
                "report": result,
            }
    except Exception as e:
        return {
            "status": "error",
            "report": f"An error occurred: {str(e)}",
        }

