# EDP Agent

An EDP agent service built with LangGraph, FastAPI, and Streamlit .

## Core Technical Stack

- **Backend**: FastAPI service with async support
- **Agent Framework**: LangGraph
- **Frontend**: Streamlit (Streamlit frontend demo pursoe only)
- **Containerization**: Docker with compose watch support
- **Package Management**: uv (with pip fallback)

## Quick Start

### Local Python Setup

```sh
# Required: AWS Profile
echo 'AWS_PROFILE=aws_profile' >> .env

# Install dependencies
pip install uv && uv sync --frozen
source .venv/bin/activate

# Run services
python src/run_service.py
streamlit run src/streamlit_app.py  # In separate terminal
```

### Docker Setup

```sh
echo 'AWS_PROFILE=aws_profile' >> .env
docker compose watch
```

## Key Components

1. **Agent Service (Port 80)**

- Streaming/non-streaming endpoints
- Dynamic metadata via /info
- Multiple agent support via URL paths
- Content moderation with LlamaGuard

1. **Streamlit Interface (Port 8501)**

- Interactive chat UI
- Real-time streaming support

## Project Structure

```bash
src/
├── agents/        # Agent implementations
├── core/          # LLM configs & core modules
├── schema/        # Protocol definitions
├── service/       # FastAPI service
├── client/        # Agent client library
└── streamlit_app.py
```

## Integration

- Use AgentClient class for custom integrations
- Supports sync/async operations
- Streaming/non-streaming modes

Example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("What are QB EDP/SolveBio expressions?")
response.pretty_print()
```

## Environment

- Requires at least one LLM API key
- Supports multiple model providers
- Optional: LangSmith tracing, OpenWeatherMap API

See .env.example for all configuration options

### Key Files

The repository is structured as follows:

- `src/agents/`: Defines several agents with different capabilities
- `src/schema/`: Defines the protocol schema
- `src/core/`: Core modules including LLM definition and settings
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface
- `tests/`: Unit and integration tests
