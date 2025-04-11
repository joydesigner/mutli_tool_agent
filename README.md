# mutli_tool_agent
This repository contains a couple of AI agent projects.

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

