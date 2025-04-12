from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm
import os
import requests

# --- Constants ---
APP_NAME = "travel_planner_app"
USER_ID = "dev_user_01"
SESSION_ID = "planner_session_01"
GEMINI_MODEL = "gemini-2.0-flash-exp"
OLLAMA_MODEL = "ollama/gemma3:12b"

# Weather and Time API
WEATHER_TIME_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_TIME_API_BASE_URL = os.getenv("WEATHER_API_URL")


# Function to get the city weather
def get_weather(city: str) -> object:
    """
    Function to get the weather information for a given city.
    """
    # API call to a weather service would go here
    api_key = WEATHER_TIME_API_KEY
    base_url = WEATHER_TIME_API_BASE_URL
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
    api_key = WEATHER_TIME_API_KEY
    base_url = WEATHER_TIME_API_BASE_URL
    params = {
        "key": api_key,
        "q": city,
        "aqi": "no",
    }

# --- 1. Define Sub-Agents for Each Pipeline Stage ---


# Greeting Agent
greeting_agent = LlmAgent(
    name="GreetingAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""You are a Greeting AI. Your responsbility is to narrow down what date and which city the user want to travel to.
    You will greet the user and ask the user to provide the destination city to travel to.
    You will also ask the user to provide the start date of the trip.
    You will also ask the user to provide the end date of the trip.
    You will ask the user to provide the number of days the user wants to stay in the destination city if the user does not provide the end date.
    You will also ask the user to provide the number of people traveling to the destination city.
    You will also ask the user to provide the budget for the trip.
    You will also ask the user to provide the type of the trip.
    You will also ask the user to provide the activities the user wants to do in the destination city.
    You will also ask the user to provide the transportation mode the user wants to use in the destination city.
    You will also ask the other details about the trip if the user can provide them.
    You will give a confirmation message to the user to confirm the details of the trip.
    *** Note: You should not suggest the user a city or date if the user has provided the destination city and date. ***
    *** Note: Make sure to follow the following steps in order. ***
    If the user does not provide the destination city of the trip, you should ask the user to provide the destination city.
    If the user does not provide the start date of the trip, you should ask the user to provide the start date.
    If the user is not sure which city to travel to, if the user knows the country they want to travel to, you should suggest the user a city based on the country the user wants to travel to.
    """,
    description="Greeting AI, can greet the user and ask the user to provide the destination city to travel to.",
    output_key="confirmation_message",
)

# Weather and time  Agent
weather_time_agent = LlmAgent(
    name="WeatherTimeAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""You are a Weather and Time AI. You can check the weather and time for the destination city from the state key 'confirmation_message'.
    """,
    description="Weather and Time AI, can check the weather and time for the destination city.",
    tools=[get_weather, get_current_time],
    output_key="weather_time_info"
)

# Flight Finder Agent
# Takes the initial specification (from user query) and finds the flights.
flight_finder_agent = LlmAgent(
    name="FlightFinderAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""You are a Flight Finder AI.
    Based on the user's request, find the flights for the start date and end date from the state key 'confirmation_message'.
    Output *only* the flight options.
    """,
    description="Flight Finder AI, finds the flights based on the user's request.",
    # Stores its output (the flight options) into the session state
    # under the key 'flight_options'.
    output_key="flight_options"
)

# Trip Itinerary Designer Agent
# Based on the flight options, design the trip itinerary.
itinerary_designer_agent = LlmAgent(
    name="ItineraryDesignerAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""
    You are a Trip Itinerary Designer AI.
    Based on the flight options from state key 'flight_options' and weather conditions from state key 'weather_time_info' design the trip itinerary.
    If the weather is not good, you should suggest the user to change the destination city or date.
    Output *only* the itinerary.
    """,
    description="Designs the trip itinerary based on the flight options.",
    # Stores its output (the itinerary) into the session state
    # under the key 'itinerary'.
    output_key="itinerary"
)

# User Approval Agent
# Takes the itinerary and the user's approval status (read from state) and provides feedback.
user_approval_agent = LlmAgent(
    name="UserApprovalAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""You are a User Approval AI.
    Take the itinerary provided in the session state key 'itinerary'
    and the user's approval status found in the session state key 'approval_status'.
    Ask the user to provide feedback on the itinerary from state key 'itinerary'.
    If the user is not satisfied, return the itinerary to the itinerary designer agent to change the itinerary.
    """,
    description="Provides feedback on the itinerary and the user's approval status.",
    # Stores its output (the feedback) into the session state
    # under the key 'feedback'.
    output_key="feedback"
)

# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.
root_agent = SequentialAgent(
    name="TravelPlannerAgent",
    sub_agents=[greeting_agent, weather_time_agent, flight_finder_agent, itinerary_designer_agent, user_approval_agent]
    # The agents will run in the order provided: Finder -> Designer -> Approver
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

# --- 3. Run the Agent ---
# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("perform travel planning")

