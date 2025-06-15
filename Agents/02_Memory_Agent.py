from IPython.display import Image, display
import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# Define the state graph


class AgentState(TypedDict):
    # sono datatype in langgraph.
    messages: List[Union[HumanMessage, AIMessage]]


# inizialize model
llm = ChatOpenAI(model="gpt-4o")


# create node

def process(state: AgentState) -> AgentState:
    """
    This node will solve the request you input
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print("Current State: ", state["messages"])

    return state

# Define the state graph


graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# ---- Save the graph to a file ---

# print the graph
# display(Image(agent.get_graph().draw_mermaid_png()))  # Display the graph image

# save the graph as a PNG file
# png_bytes = agent.get_graph().draw_mermaid_png()

# Scrive il contenuto bytes in un file .png
# with open("image/test.png", "wb") as f:
# f.write(png_bytes)
# --------

# History of messages

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    # Add user input to the conversation history
    conversation_history.append(HumanMessage(content=user_input))

    # Invoke the agent with the current conversation history, not just the last message
    # the agent is the compiled graph
    # and the invoke method will run the graph
    # and return the result
    result = agent.invoke({"messages": conversation_history})

    # Update the conversation history with the AI's response
    conversation_history = result["messages"]

    # Get the next user input
    user_input = input("Enter: ")


# ---- Save the conversation history to a file ( a duplicated version of the state) ----
# for prototying we can save the conversation history to a text file,
# in productiont we can save it to a database or a vector store
# Configurable header for the conversation history file; update as needed for production use
CONVERSATION_HISTORY_HEADER = "Conversation History:\n"

with open("conversation_history.txt", "w", encoding="utf-8") as f:
    f.write(CONVERSATION_HISTORY_HEADER)

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation history.\n")

# Print the conversation history
print("Conversation saved to conversation_history.txt")
