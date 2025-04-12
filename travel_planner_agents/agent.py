from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import Tool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_NAME = "travel_planner_app"
USER_ID = "travel_user_01"
SESSION_ID = "travel_session_01"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Constants for state keys
STATE_FLIGHT_OPTIONS = "flight_options"
STATE_HOTEL_OPTIONS = "hotel_options"
STATE_WEATHER_FORECAST = "weather_forecast"
STATE_ITINERARY = "itinerary"
STATE_BUDGET_ANALYSIS = "budget_analysis"
STATE_APPROVAL_STATUS = "approval_status"

# Model constants
OLLAMA_MODEL = "ollama/gemma3:12b"
GEMINI_MODEL = "gemini-2.0-flash-exp"

class TravelCoordinator:
    def __init__(self):
        # Create specialized agents with direct tools
        self.flight_finder = self._create_flight_finder()
        self.hotel_booker = self._create_hotel_booker()
        self.itinerary_designer = self._create_itinerary_designer()
        self.budget_checker = self._create_budget_checker()
        self.human_approval = self._create_human_approval_tool()

        # Create the workflow
        self.workflow = SequentialAgent(
            name="TravelWorkflow",
            sub_agents=[
                # Phase 1: Parallel Data Collection
                ParallelAgent(
                    name="DataCollection",
                    sub_agents=[
                        self.flight_finder,    # Output stored in STATE_FLIGHT_OPTIONS
                        self.hotel_booker,     # Output stored in STATE_HOTEL_OPTIONS
                    ]
                ),
                # Phase 2: Iterative Planning
                LoopAgent(
                    name="PlanningRefinement",
                    sub_agents=[
                        self.itinerary_designer,  # Output stored in STATE_ITINERARY
                        self.budget_checker       # Output stored in STATE_BUDGET_ANALYSIS
                    ],
                    max_iterations=3,
                    condition=lambda state: not state.get(STATE_BUDGET_ANALYSIS, {}).get("within_budget", True)
                ),
                # Phase 3: Final Approval
                SequentialAgent(
                    name="ApprovalProcess",
                    sub_agents=[
                        self.human_approval  # Output stored in STATE_APPROVAL_STATUS
                    ]
                )
            ]
        )

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
            Consider price, duration, and layovers.
            Return flight options in a structured format.
            """,
            tools=[
                Tool(
                    name="search_flights",
                    func=self._search_flights,
                    description="Search for available flights based on criteria"
                )
            ],
            output_key=STATE_FLIGHT_OPTIONS
        )

    def _create_hotel_booker(self) -> LlmAgent:
        return LlmAgent(
            name="hotel_booker",
            model=LiteLlm(model=OLLAMA_MODEL),
            instruction=f"""Find and compare hotel options based on location, dates, and budget.
            Consider amenities, ratings, and location.
            Return hotel options in a structured format.
            """,
            tools=[
                Tool(
                    name="search_hotels",
                    func=self._search_hotels,
                    description="Search for available hotels based on criteria"
                )
            ],
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
            tools=[
                Tool(
                    name="create_itinerary",
                    func=self._create_itinerary,
                    description="Create a detailed daily itinerary"
                )
            ],
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
            tools=[
                Tool(
                    name="check_budget",
                    func=self._check_budget,
                    description="Verify if total costs are within budget"
                )
            ],
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
            tools=[
                Tool(
                    name="request_human_approval",
                    func=self._request_human_approval,
                    description="Request human approval for budget overruns"
                )
            ],
            output_key=STATE_APPROVAL_STATUS
        )

    # Tool implementations
    def _search_flights(self, origin: str, destination: str, dates: Dict[str, str]) -> Dict[str, Any]:
        # Implementation for flight search
        pass

    def _search_hotels(self, location: str, dates: Dict[str, str], budget: float) -> Dict[str, Any]:
        # Implementation for hotel search
        pass

    def _create_itinerary(self, flight_options: Dict[str, Any], hotel_options: Dict[str, Any], 
                         preferences: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for itinerary creation
        pass

    def _check_budget(self, flight_cost: float, hotel_cost: float, 
                     activity_costs: List[float], total_budget: float) -> Dict[str, Any]:
        # Implementation for budget checking
        pass

    def _request_human_approval(self, budget_analysis: Dict[str, Any], 
                              itinerary: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for human approval
        pass

    def plan_trip(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the travel planning workflow."""
        return self.workflow.run(request)

# Create the main coordinator instance
travel_coordinator = TravelCoordinator()

# Expose the root agent for ADK CLI
root_agent = travel_coordinator.workflow

# Example usage
def call_travel_planner(request: Dict):
    """
    Example function to call the travel planner with a request.
    """
    result = travel_coordinator.plan_trip(request)
    print("Travel Plan:", json.dumps(result, indent=2))

# Example request
example_request = {
    "intent": "leisure",
    "budget": 3000,
    "origin": "New York",
    "destination": "Tokyo",
    "dates": {
        "start": "2024-05-01",
        "end": "2024-05-07"
    },
    "preferred_activities": ["sushi-making class", "temples", "shopping"],
    "constraints": {
        "dietary_restrictions": ["vegetarian"],
        "mobility": "normal"
    }
}

# Call the travel planner
call_travel_planner(example_request)

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

