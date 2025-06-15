""" 
type
Annotated - provides additional context without affecting the type itself
Sequence - To automatically handle the state updates for seuquences such as by adding new  message to a chat history

message
BaseMessage - The base class for all messages in LangGraph, which can be extended to create custom message types
ToolMessage - Passes data back to LLM after it calls a tool such as the content and the tool_call_id
SystemMessage - A message that provides instructions or context to the LLM, such as the system prompt

add_messages is a reducer Function
rule that controls how updates from nodes are combined with the existing state.
Tell us how to merge new data into the current state.

without a reducer, the state would be replaced with the new data, losing previous information.

NON usiamo append come nella lezione precedente perché sarebbero troppi
"""
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from utility import save_graph_image

load_dotenv()

# agent state definition


class AgentState(TypedDict):
    # preserve the history of messages, appending new ones
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: Annotated[Sequence[ToolMessage], add_messages]


@tool
def add_numbers(a: int, b: int) -> int:
    """
    A simple tool that adds two numbers.
    """
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """
    A simple tool that multiply two numbers.
    """
    return a * b


@tool
def subtract_numbers(a: int, b: int) -> int:
    """
    A simple tool that subtract two numbers
    """
    return a - b


tools = [add_numbers, multiply_numbers, subtract_numbers]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# Create a node that uses the LLM to process messages


def model_call(state: AgentState) -> AgentState:
    """
    This node will solve the request you input
    """

    # define the system prompt
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. You can use tools to answer questions. "
                "If you need to use a tool, please call it with the correct parameters."
    )
    # Add the system prompt to the messages and the query from the user
    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


# conditional node edge
def should_continue(state: AgentState) -> bool:
    """
    Check if we continue the loop or not.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# define the graph

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


# save image graph function
# save_graph_image(app, "reactAgent")


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [
    ("user", "Add 540 + 12 and multiply the result by 6. Also tell me a joke please")]}

print_stream(app.stream(inputs, stream_mode="values"))
