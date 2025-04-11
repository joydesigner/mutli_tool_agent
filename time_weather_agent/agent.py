from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm
import requests

# Function to get the city weather
def get_weather(city: str) -> object:
    """
    Function to get the weather information for a given city.
    """
    # API call to a weather service would go here
    api_key = "9a73228d8b9a4cb3aee114043251104"
    base_url = "http://api.weatherapi.com/v1/current.json"
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
    api_key = "9a73228d8b9a4cb3aee114043251104"
    base_url = "http://api.weatherapi.com/v1/current.json"

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

# Create a Root Agent - Gemini Model agent
root_agent = Agent(
    name="weather_time_agent", # Agent name
    model="gemini-2.0-flash-exp", # Model name
    instruction=(
        "I'm a helpful assistant. "
        "I can give the information about the weather and time for the cities the user queries."
        "When the user asks about the weather, I will use the get_weather tool to extract the latest weather information."
        "When the user asks about the time, I will use the get_current_time tool to extract the accurate date and time."
        "I will answer in a polite and informative manner."
        "I will provide detailed information about the weather and time in clear and accurate manner."
    ), # Instruction for the agent
    description=(
        "A smart assistant. Can answer the questions of weather and time for different cities."
    ), # Description of the agent
    tools=[get_weather, get_current_time], # Tools available to the agent
)

# Create a LiteLLM instance - Ollama Gemma Model agent
# root_agent = LlmAgent(
#     name="helpful_agent",
#     model=LiteLlm(model="ollama/gemma3:12b"),
#     instruction="You are a helpful assistant.",
#     description="A helpful assistant.",
# )

