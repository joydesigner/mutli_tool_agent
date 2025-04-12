# Multi-Tool Agent Projects

This repository contains AI agent projects built using Google ADK (Agent Development Kit).

## Projects

### 1. Time & Weather Agent
A multi-tool agent that provides current time and weather information for any city.

### 2. Travel Planner Agent
An intelligent travel planning system that coordinates multiple specialized agents to create personalized travel itineraries.

## Travel Planner Architecture

### Multi-Agent System
The travel planner uses a hierarchical agent architecture:

1. **Travel Coordinator** (Main Agent)
   - Orchestrates the entire planning process
   - Manages workflow and agent coordination
   - Handles budget validation and human approval

2. **Specialized Agents**
   - **Flight Finder**: Searches and compares flight options
   - **Hotel Booker**: Finds and books accommodations
   - **Weather Fetcher**: Checks weather conditions
   - **Itinerary Designer**: Creates daily schedules
   - **Budget Checker**: Validates costs against budget
   - **Human Approval**: Handles budget overruns

### Agent Orchestration
The system uses a sequential workflow with parallel processing:

1. **Parallel Data Collection**
   - Concurrently runs flight, hotel, and weather searches
   - Uses `ParallelAgent` for efficient data gathering

2. **Iterative Planning**
   - Refines itinerary and budget using `LoopAgent`
   - Maximum 3 iterations for optimization
   - Timeout protection to prevent infinite loops

3. **Final Approval**
   - Human-in-the-loop for budget overruns
   - Final validation before completion

### Tools & Capabilities
- **Flight Search**: Find and compare flight options
- **Hotel Booking**: Search accommodations by location and budget
- **Weather Checking**: Verify weather conditions
- **Itinerary Design**: Create detailed daily schedules
- **Budget Analysis**: Track and validate costs
- **Human Approval**: Escalate for budget overruns

## Deployment and Usage

### Prerequisites
- Python 3.8 or later
- LLM model access (Ollama/Gemini)
- Google ADK installed

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install google-adk litellm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Agent
```bash
# Start the agent
adk run travel_planner

# Access web interface
adk web
# Visit http://localhost:8000/
```

### Example Usage
```python
from travel_planner_agents.agent import TravelCoordinator

# Initialize the coordinator
coordinator = TravelCoordinator()

# Plan a trip
request = {
    "intent": "plan_trip",
    "budget": 3000,
    "origin": "New York",
    "destination": "Japan",
    "dates": {
        "start": "2024-05-01",
        "end": "2024-05-07"
    },
    "preferred_activities": ["sushi-making class"],
    "constraints": ["vegetarian"]
}

result = coordinator.plan_trip(request)
```

## Features
- **Parallel Processing**: Efficient data collection
- **Iterative Refinement**: Optimize plans within budget
- **Human-in-the-Loop**: Approval for budget overruns
- **Timeout Protection**: Prevents infinite loops
- **Error Handling**: Graceful failure recovery

## Contributing
Feel free to submit issues and enhancement requests!

## time_weather_agent
This is a multi-tool agent that can answer questions about the current time and weather for a given city. 

## Deployment and Usage

### Prerequisites
- Python 3.8 or later
- LLM model access

### Create & Activate Virtual Environment & Installation
```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate

# install dependencies
pip install google-adk -q
pip install litellm -q
pip install mcp -q

mkdir multi_tool_agent
cd multi_tool_agent
mkdir time_weather_agent/
echo "from . import agent" > multi_tool_agent/__init__.py
touch time_weather_agent/agent.py
touch time_weather_agent/.env

GOOGLE_GENAI_USE_VERTEXAI="False"
GOOGLE_API_KEY="gemini api key"
OPENAI_API_KEY="openai api key"

adk run my_agent

adk web

# localhost
http://localhost:8000/

# llm api key
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

