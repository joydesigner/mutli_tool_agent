# mutli_tool_agent
## A demo that uses the latest ADK (by Google) to build a powerful muti-agents tool.

## What is the Agent Development Kit? 
ADK is a flexible and modular framework designed for developing and deploying AI agents. It supports building conversational and non-conversational agents that can handle complex tasks and workflows. ADK is not only suitable for Gemini models in the Google ecosystem, but also compatible with other mainstream large language models (LLMs) and open source generative AI tools. The core goal of this framework is to enable developers to quickly build, manage, evaluate and deploy production-level agent applications.

## ADK Core Functions 
ADK covers every stage of the agent development lifecycle, from logic definition to production deployment, and provides a series of powerful tools and functions:

- **Simplified development:** Through intuitive Python code, developers can quickly build an AI agent with less than 100 lines of code.
- **Dynamic routing and behavior control:** Supports dynamic routing based on LLM drive and deterministic logic control, allowing developers to have more precise control over agent behavior.
- **Multimodal interaction:** ADK supports two-way audio and video streaming, making human-computer interaction more natural.
- **Preconfigured samples and tools:** A rich built-in sample library (Agent Garden) covers scenarios such as retail and customer service to help developers get started quickly.
- **Cross-platform deployment:** Supports multiple deployment environments, including local debugging, containerized runtimes (such as Kubernetes), and Google Vertex AI.

## Deep integration with Google ecosystem 
ADK has specially optimized integration with the Google Cloud ecosystem, such as seamless integration with the Gemini 2.5 Pro Experimental model and the Vertex AI platform. Through these integrations, developers can take full advantage of the enhanced reasoning capabilities of the Gemini model and directly deploy the agent to the enterprise-level runtime environment. In addition, ADK also supports Model Context Protocol (MCP), a data connection protocol created by Anthropic for standardized data transmission between different agents.

## Enterprise-level scalability
In order to meet the needs of enterprises, Google has also launched Agent Engine as an important supplement to ADK. This managed runtime interface provides a one-stop solution from proof of concept to production deployment, including context management, infrastructure monitoring, expansion, security assessment and other functions. In addition, Agent Engine also supports long-term and short-term memory functions, allowing agents to make more accurate decisions based on historical context.

## Application Scenarios and Future Outlook
ADK has shown strong application potential in multiple industries. For example:

- Retailers can use ADK to build dynamic pricing systems and optimize pricing strategies through multi-agent collaboration.
- The automotive industry uses ADK to analyze geographic and traffic data to provide decision support for the site selection of electric vehicle charging stations.
- Media companies use ADK for video analysis to greatly improve content processing efficiency.
- 
In the future, Google plans to further expand ADK and its related tools, such as introducing simulation environments to test the performance of agents in real scenarios, and promoting open protocols (such as Agent2Agent Protocol) to achieve seamless collaboration between cross-platform and multi-vendor agents.

Google's Agent Development Kit provides strong support for the development of multi-agent systems with its modular design, flexibility, and deep integration with the Google ecosystem. Whether it is a startup or a large enterprise, this framework can help developers quickly move from proof of concept to production deployment, bringing more possibilities for artificial intelligence applications.

## Deployment and Usage
### Prerequisites
- Python 3.8 or later
- Google ADK installed
- LLM model access

### Installation
```bash
pip install google-adk
pip install litellm
pip install mcp
pip install mcp-server-fetch


mkdir multi_tool_agent/
echo "from . import agent" > multi_tool_agent/__init__.py
touch multi_tool_agent/agent.py
touch multi_tool_agent/.env

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

