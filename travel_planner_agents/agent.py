from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import requests
import json
from typing import Dict, List, Optional
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
GEMINI_MODEL = "gemini-2.0-flash-exp"
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

class TravelCoordinator:
    def __init__(self):
        # Create tool agents first
        self.search_flights_agent = self._create_search_flights_agent()
        self.search_hotels_agent = self._create_search_hotels_agent()
        self.get_weather_agent = self._create_get_weather_agent()
        self.create_itinerary_agent = self._create_create_itinerary_agent()
        self.check_budget_agent = self._create_check_budget_agent()
        self.request_human_approval_agent = self._create_request_human_approval_agent()

        # Create specialized agents that depend on tool agents
        self.flight_finder = self._create_flight_finder()
        self.hotel_booker = self._create_hotel_booker()
        self.weather_fetcher = self._create_weather_fetcher()
        self.itinerary_designer = self._create_itinerary_designer()
        self.budget_checker = self._create_budget_checker()
        self.human_approval = self._create_human_approval_tool()

        # Create the workflow using all three agent types
        self.workflow = SequentialAgent(
            name="TravelWorkflow",
            sub_agents=[
                # Phase 1: Parallel Data Collection
                ParallelAgent(
                    name="DataCollection",
                    sub_agents=[
                        self.flight_finder,    # Output stored in STATE_FLIGHT_OPTIONS
                        self.hotel_booker,     # Output stored in STATE_HOTEL_OPTIONS
                        self.weather_fetcher   # Output stored in STATE_WEATHER_FORECAST
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
                    # condition=lambda state: not state.get(STATE_BUDGET_ANALYSIS, {}).get("within_budget", True)
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

    def _create_search_flights_agent(self) -> Agent:
        return Agent(
            name="search_flights",
            model=GEMINI_MODEL,
            instruction="""Search for flights based on origin, destination, and dates.
            Return flight options in a structured format.
            """,
            description="Searches for available flights"
        )

    def _create_search_hotels_agent(self) -> Agent:
        return Agent(
            name="search_hotels",
            model=GEMINI_MODEL,
            instruction="""Search for hotels based on location, dates, and budget.
            Return hotel options in a structured format.
            """,
            description="Searches for available hotels"
        )

    def _create_get_weather_agent(self) -> Agent:
        return Agent(
            name="get_weather",
            model=GEMINI_MODEL,
            instruction="""Fetch weather information for a given city.
            Return weather forecast in a structured format.
            """,
            description="Fetches weather information"
        )

    def _create_create_itinerary_agent(self) -> Agent:
        return Agent(
            name="create_itinerary",
            model=GEMINI_MODEL,
            instruction="""Create a daily itinerary based on activities and weather.
            Return itinerary in a structured format.
            """,
            description="Creates travel itinerary"
        )

    def _create_check_budget_agent(self) -> Agent:
        return Agent(
            name="check_budget",
            model=GEMINI_MODEL,
            instruction="""Check if total costs are within budget.
            Return budget analysis in a structured format.
            """,
            description="Checks budget compliance"
        )

    def _create_request_human_approval_agent(self) -> Agent:
        return Agent(
            name="request_human_approval",
            model=GEMINI_MODEL,
            instruction="""Request human approval for specific decisions.
            Return approval status in a structured format.
            """,
            description="Handles human approval requests"
        )

    def _create_flight_finder(self) -> Agent:
        return Agent(
            name="flight_finder",
            model=GEMINI_MODEL,
            instruction="""Find and compare flight options based on origin, destination, and dates.
            Consider price, duration, and layovers.
            Return flight options in a structured format.
            """,
            tools=[AgentTool(agent=self.search_flights_agent)],
            output_key=STATE_FLIGHT_OPTIONS  # Store results in state
        )

    def _create_hotel_booker(self) -> Agent:
        return Agent(
            name="hotel_booker",
            model=GEMINI_MODEL,
            instruction="""Find and compare hotel options based on location, dates, and budget.
            Consider amenities, ratings, and location.
            Return hotel options in a structured format.
            """,
            tools=[AgentTool(agent=self.search_hotels_agent)],
            output_key=STATE_HOTEL_OPTIONS  # Store results in state
        )

    def _create_weather_fetcher(self) -> Agent:
        return Agent(
            name="weather_fetcher",
            model=GEMINI_MODEL,
            instruction="""Fetch weather information for the destination.
            Consider current and forecasted weather.
            Return weather information in a structured format.
            """,
            tools=[AgentTool(agent=self.get_weather_agent)],
            output_key=STATE_WEATHER_FORECAST  # Store results in state
        )

    def _create_itinerary_designer(self) -> Agent:
        return Agent(
            name="itinerary_designer",
            model=GEMINI_MODEL,
            instruction=f"""Design a detailed daily itinerary based on available activities,
            weather conditions, and user preferences.
            Use the following data from state:
            - {STATE_FLIGHT_OPTIONS}: Available flights
            - {STATE_HOTEL_OPTIONS}: Available hotels
            - {STATE_WEATHER_FORECAST}: Weather information
            Return a structured daily schedule in JSON format.
            """,
            tools=[AgentTool(agent=self.create_itinerary_agent)],
            output_key=STATE_ITINERARY  # Store results in state
        )

    def _create_budget_checker(self) -> Agent:
        return Agent(
            name="budget_checker",
            model=GEMINI_MODEL,
            instruction=f"""Verify if the total cost of the trip is within budget.
            Use the following data from state:
            - {STATE_FLIGHT_OPTIONS}: Flight costs
            - {STATE_HOTEL_OPTIONS}: Hotel costs
            - {STATE_ITINERARY}: Activity costs
            Break down costs by category.
            Suggest alternatives if over budget.
            Return a dictionary with 'within_budget' boolean and 'total_cost' float.
            """,
            tools=[AgentTool(agent=self.check_budget_agent)],
            output_key=STATE_BUDGET_ANALYSIS  # Store results in state
        )

    def _create_human_approval_tool(self) -> Agent:
        return Agent(
            name="human_approval",
            model=GEMINI_MODEL,
            instruction=f"""Handle cases requiring human approval.
            Use the following data from state:
            - {STATE_BUDGET_ANALYSIS}: Budget compliance status
            - {STATE_ITINERARY}: Proposed itinerary
            Present clear options and reasoning.
            Wait for human input before proceeding.
            Return a dictionary with 'approved' boolean and 'reason' string.
            """,
            tools=[AgentTool(agent=self.request_human_approval_agent)],
            output_key=STATE_APPROVAL_STATUS  # Store results in state
        )

    # Tool implementations
    def _search_flights(self, origin: str, destination: str, dates: Dict[str, str]) -> Dict:
        # Implementation would call actual flight API
        return {
            "status": "success",
            "flights": [
                {
                    "airline": "Sample Airline",
                    "price": 500,
                    "duration": "12h",
                    "stops": 1
                }
            ]
        }

    def _search_hotels(self, location: str, dates: Dict[str, str], budget: float) -> Dict:
        # Implementation would call actual hotel API
        return {
            "status": "success",
            "hotels": [
                {
                    "name": "Sample Hotel",
                    "price": 200,
                    "rating": 4.5,
                    "amenities": ["pool", "gym"]
                }
            ]
        }

    def _get_weather(self, city: str) -> Dict:
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

    def _create_itinerary(self, activities: List[Dict], weather: Dict) -> Dict:
        # Implementation would create detailed daily itinerary
        return {
            "status": "success",
            "itinerary": {
                "day_1": {
                    "activities": [],
                    "weather": "sunny"
                }
            }
        }

    def _check_budget(self, costs: Dict[str, float], budget: float) -> Dict:
        total = sum(costs.values())
        return {
            "status": "success",
            "within_budget": total <= budget,
            "total_cost": total,
            "breakdown": costs
        }

    def _request_human_approval(self, reason: str, options: List[Dict]) -> Dict:
        return {
            "status": "pending",
            "reason": reason,
            "options": options,
            "requires_human_approval": True
        }

    def plan_trip(self, request: Dict) -> Dict:
        """
        Main orchestration function that coordinates all agents to plan a trip.
        """
        # Convert request to ADK content format
        content = types.Content(
            role='user',
            parts=[types.Part(text=json.dumps(request))]
        )

        # Execute the workflow with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                events = self.runner.run(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=content
                )

                # Process events and return final response
                for event in events:
                    if event.is_final_response():
                        return json.loads(event.content.parts[0].text)

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
        "aqi": "no",
    }

    try:
        response = requests.get(base_url, params=params)

        # Check if the response is successful
        if response.status_code != 200:
            return f"Error: Unable to fetch data from the weather service. Status code: {response.status_code}"

        data = response.json()
        if "error" in data:
            return f"Error: {data['error']['message']}"
        else:
            location = data["location"]["name"]
            region = data["location"]["region"]
            country = data["location"]["country"]
            temp_c = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            return {
                "status": "success",
                "report": f"The current temperature in {location}, {region}, {country} is {temp_c}°C with {condition}.",
            }
    except Exception as e:
        return {
            "status": "error",
            "report": f"An error occurred: {str(e)}",
        }

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

