from typing import Annotated, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.tools import generate_code  # code_generator
from core import get_model, settings


class ExpressionResponse(BaseModel):
    """Respond to the user in this EDP expression synthax format."""

    expression: str = Field(
        description="Expressions resemble a single line of code in Python. Expressions can be one line of code but can contain line breaks. They do not support the declaration of variables or classes."
    )


class Message(BaseModel):
    """A message in the conversation."""

    role: str
    content: str

    def pretty_print(self) -> None:
        """Pretty print the message."""
        if self.role == "assistant":
            print(f"\n{self.content}\n")
        else:
            print(f"\n{self.role}: {self.content}\n")


class AgentState(TypedDict):
    messages: Annotated[list[dict], "The messages in the conversation"]


async def acall_model(state: AgentState, config: dict) -> AgentState:
    model = get_model(
        config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    )

    # Create the prompt template for the react agent
    prompt = PromptTemplate.from_template(
        """You are a helpful coding assistant. Use the tools available to help write code.
When responding, make sure the code follows the expression syntax format.

Tools available:
{tools}

Tool Names: {tool_names}

{agent_scratchpad}
"""
    )

    # Create the react agent with proper components
    agent = create_react_agent(
        llm=model,
        tools=[generate_code],
        prompt=prompt,
        # output_parser=ReActSingleInputOutputParser(),
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent, tools=[code_generator], verbose=True, handle_parsing_errors=True
    )

    # Get the last message content
    last_message = state["messages"][-1]
    input_content = (
        last_message["content"] if isinstance(last_message, dict) else last_message[1]
    )

    # Execute the agent
    response = await agent_executor.ainvoke({"input": input_content})

    # Create a Message object for the response
    formatted_response = Message(role="assistant", content=response["output"])

    # Return updated state with new Message object
    return {"messages": state["messages"] + [formatted_response]}


# Define the graph
agent = StateGraph(AgentState)

# Add the model node
agent.add_node("model", acall_model)

# Set the entry point
agent.set_entry_point("model")

# Add edge to END
agent.add_edge("model", END)

# Compile the graph
edp_coder = agent.compile(
    checkpointer=MemorySaver(),
)


# from typing import Annotated, TypedDict

# from langchain.agents import create_react_agent
# from langchain.tools import BaseTool
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, StateGraph
# from pydantic import BaseModel, Field

# from agents.tools import code_generator
# from core import get_model, settings


# class ExpressionResponse(BaseModel):
#     """Respond to the user in this EDP expression synthax format."""

#     conditions: str = Field(
#         description="Expressions resemble a single line of code in Python. Expressions can be one valid line of code but can contain line breaks. They do not support the declaration of variables or classes."
#     )


# class AgentState(TypedDict):
#     messages: Annotated[list[dict], "The messages in the conversation"]


# async def acall_model(state: AgentState, config: dict) -> AgentState:
#     model = get_model(
#         config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
#     )

#     agent_runnable = create_react_agent(
#         model,
#         tools=[code_generator],
#         response_format=ExpressionResponse,
#     )

#     response = await agent_runnable.ainvoke(state["messages"], config)
#     return {"messages": state["messages"] + [response]}


# # Define the graph
# agent = StateGraph(AgentState)

# # Add the model node
# agent.add_node("model", acall_model)

# # Set the entry point
# agent.set_entry_point("model")

# # Add edge to END
# agent.add_edge("model", END)

# # Compile the graph
# edp_coder = agent.compile(
#     checkpointer=MemorySaver(),
# )
