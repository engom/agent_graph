import warnings
from datetime import datetime
from typing import Literal, Optional

warnings.filterwarnings("ignore", message="'api' backend is deprecated")

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.tools import calculator, code_generator
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """State management for the agent with remaining steps tracking."""

    remaining_steps: RemainingSteps


def setup_tools():
    """Initialize and configure tools with error handling."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(
            safesearch="moderate",
            backend="auto",
        )
        web_search = DuckDuckGoSearchResults(name="WebSearch")  #  api_wrapper=wrapper)
        return [web_search, code_generator, calculator]
    except Exception as e:
        print(f"Error setting up tools: {e}")
        return [code_generator, calculator]


def get_system_instructions() -> str:
    """Generate system instructions with current date and EDP/SolveBio expression generating capabilities."""
    return f"""
    You are a helpful EDP coding assistant with expertise in data processing expressions and web search capabilities.
    
    **EDP/SolveBio Expressions** are Python-like formulas used in the QuartzBio platform for data manipulation, analysis, and querying. Key points include:

    1. **Purpose**: Designed to pull data, calculate statistics, and run algorithms within the QuartzBio EDP platform.
    2. **Syntax**: Uses Python-like syntax, making it intuitive for users familiar with Python.
    3. **Built-in Functions**: Includes a library of functions tailored for EDP datasets and common data processing tasks.
    4. **Flexibility**: Allows access to and manipulation of data across datasets, performing calculations, and applying complex logic.
    5. **Dataset Operations**: Enables operations like retrieving the total number of records in a dataset.

    These expressions are essential for interacting with data in a flexible and powerful way, combining Python familiarity with specialized functions for data science and bioinformatics tasks.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - When handling EDP expressions:
        * Generate single-line expressions (can be formatted multi-line for readability)
        * Support basic Python operations and built-in functions (len, min, max, sum, round, range)
        * Include SolveBio-specific functions (dataset_field_stats, datetime_format, etc.)
        * Handle various data types (string, text, date, integer, float, boolean, object)
        * Validate expression syntax before returning
        * Ensure proper error handling for null values and edge cases

    - For general assistance:
        * Please include markdown-formatted links to any citations used in your response. Only include one
          or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
        * Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
          so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
        
    - When generating EDP expressions:
        * Always validate dataset field references
        * Include proper type casting when necessary
        * Handle null values gracefully
        * Follow SolveBio expression syntax rules
        * Provide clear comments for complex expressions
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model with tools and system instructions."""
    model = model.bind_tools(setup_tools())
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=get_system_instructions())]
        + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle model calls with proper error handling and step management."""
    try:
        model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
        m = get_model(model_name)
        model_runnable = wrap_model(m)
        response = await model_runnable.ainvoke(state, config)

        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": state["messages"] + [response]}
    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        return {
            "messages": state["messages"]
            + [
                AIMessage(
                    content=error_message,
                )
            ]
        }


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """Determine next step based on tool calls."""
    try:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        return "tools" if last_message.tool_calls else "done"
    except (IndexError, TypeError) as e:
        print(f"Error in pending_tool_calls: {e}")
        return "done"


def create_agent() -> StateGraph:
    """Create and configure the agent graph."""
    # Initialize the graph with the AgentState
    agent = StateGraph(AgentState)

    # Add nodes for model and tools
    agent.add_node("model", acall_model)
    agent.add_node("tools", ToolNode(setup_tools()))

    # Set the entry point to model
    agent.set_entry_point("model")

    # Add conditional edges from model node
    agent.add_conditional_edges(
        "model",
        pending_tool_calls,
        {
            "tools": "tools",  # If tools are needed, go to tools node
            "done": END,  # If no tools needed, end the conversation
        },
    )

    # Add edge from tools back to model
    agent.add_edge("tools", "model")

    return agent


# Initialize the agent
edp_assistant = create_agent().compile(checkpointer=MemorySaver())


# The graph now follows this flow:
# Start → Model
# Model → Tools (if tool calls present)
# Tools → Model (to process tool results)
# Model → End (if no more tool calls)
