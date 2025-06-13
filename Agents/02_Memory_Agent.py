import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# Define the state graph


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
