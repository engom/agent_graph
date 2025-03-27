from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional

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

from core import llm, settings

from .tools import setup_tools

# Optimize model initialization
model_cache = {}


class AgentState(MessagesState, total=False):
    """State management for the agent with remaining steps tracking."""

    remaining_steps: RemainingSteps


def get_system_instructions() -> str:
    """Generate system instructions with current date and EDP/SolveBio expression generating capabilities."""
    base = """## Role
    You are an EDP/SolveBio expressionq coding specialist with web search access.
    
    ## Context
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
        
    You have access to a set of tools you can use to solve tasks.

    ## Capabilities
    You can generate EDP/SolveBio expressions, answer questions, and provide calculations.

    ## Tools
    You have access to a code_generator tool, and web search tool.
    
    ## Response Protocol
    1. For math questions:
    - Calculator tool → plain text result
    Example: "The result of 300 * 200 is 60,000"

    2. For code generation:
    - Code block with SolveBio syntax
    - Line-by-line explanation
    Example: 
    ```solvebio
    dataset_field_stats('patients', 'age')  # Get age statistics for patients dataset
    # Output: {'min': 18, 'max': 65, 'mean': 35.5, 'stddev': 10.5}
    
    record["a"] + " world" # String expression with context: {"record": {"a": "hello"}}
    # output: "hello world"
    
    # Numeric expression using a SolveBio function
    dataset_field_stats("solvebio:public:/ClinVar/3.7.4-2017-01-30/Combined-GRCh37", "review_status_star")["avg"]
    # output: 0.883874789018
    ```
    """
    return f"{base}\nCurrent Date: {datetime.now().isoformat()[:10]}"


# Cache the model to avoid reinitializing it on every call
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model with tools and system instructions."""
    if model.model_id not in model_cache:
        model_cache[model.model_id] = model.bind_tools(
            tools=setup_tools(),
            tool_choice="auto",
        )
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=get_system_instructions())]
        + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model_cache[model.model_id]  # model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle model calls with proper error handling and step management."""
    try:
        model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
        m = llm.get_model(model_name)
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

    # except Exception as e:
    #     error_message = f"Error processing request: {str(e)}"
    #     return {
    #         "messages": state["messages"]
    #         + [
    #             AIMessage(
    #                 content=error_message,
    #             )
    #         ]
    #     }

    except Exception as e:
        # Error handling for common exceptions
        if "ModelTimeoutError" in str(e):
            error_type = "MODEL_TIMEOUT"
        elif "AccessDeniedException" in str(e):
            error_type = "AWS_PERMISSION"
        else:
            error_type = "DEFAULT"
            # f"Error processing request: {str(e)}"

        # Structured error response
        return {
            "messages": state["messages"]
            + [
                AIMessage(
                    content=format_user_error(error_type),
                    metadata={"error": error_type},
                )
            ]
        }


def format_user_error(error_type: str) -> str:
    errors = {
        "MODEL_TIMEOUT": "Apologies, the response took too long. Please try a simpler query.",
        "AWS_PERMISSION": "Authorization issue detected.",
        "DEFAULT": "Unable to process request.",
    }
    return errors.get(error_type, errors["DEFAULT"])


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
            "tools": "tools",
            "done": END,
        },
    )

    # Add edge from tools back to model
    agent.add_edge("tools", "model")

    return agent


# Initialize the agent
edp_assistant = create_agent().compile(checkpointer=MemorySaver())
