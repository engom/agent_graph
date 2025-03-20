from dataclasses import dataclass

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chat_service import chat
from agents.edp_assistant import edp_assistant
from langgraph.graph.state import CompiledStateGraph
from schema import AgentInfo

DEFAULT_AGENT = "edp-assistant"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chat": Agent(description="A simple chatbot.", graph=chat),
    "edp-assistant": Agent(
        description="A research assistant with web search and calculator.",
        graph=edp_assistant,
    ),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]
