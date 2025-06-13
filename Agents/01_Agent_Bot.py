from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# define the state graph


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenAI(model="gpt-4o")

# define the node


def process(state: AgentState) -> AgentState:
    """
    Process the messages in the state and return the updated state."""
    # process the messages in the state
    response = llm.invoke(state["messages"])
    print(f"\nResponse: {response.content}")
    return state


# create the state graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")
