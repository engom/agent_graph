import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

_ = load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

agent = get_agent(DEFAULT_AGENT)


# Find me a recipe for chocolate chip cookies
async def main() -> None:
    inputs = {
        "messages": [
            (
                "user",
                "Create a column Race based on the values of different race-related variables by combining columns ",
            )
        ]
    }
    result = await agent.ainvoke(
        inputs,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    result["messages"][-1].pretty_print()

    # Draw the agent graph as png
    # export CFLAGS="-I/usr/include/graphviz"
    # export LDFLAGS="-L/usr/lib/graphviz"
    # uv add pygraphviz
    # pip install pygraphviz

    agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
