from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm

# --- Constants ---
APP_NAME = "travel_planner_app"
USER_ID = "dev_user_01"
SESSION_ID = "planner_session_01"
GEMINI_MODEL = "gemini-2.0-flash-exp"
OLLAMA_MODEL = "ollama/gemma3:12b"
# --- 1. Define Sub-Agents for Each Pipeline Stage ---

# Flight Finder Agent
# Takes the initial specification (from user query) and finds the flights.
flight_finder_agent = LlmAgent(
    name="FlightFinderAgent",
    model=LiteLlm(model=OLLAMA_MODEL),
    instruction="""You are a Flight Finder AI.
    Based on the user's request, find the flights.
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
    Based on the flight options, design the trip itinerary.
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
    Provide feedback on the itinerary and the user's approval status.
    Output *only* the feedback.
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
    sub_agents=[flight_finder_agent, itinerary_designer_agent, user_approval_agent]
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

