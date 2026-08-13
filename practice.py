# import os
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from langchain_groq import ChatGroq
# load_dotenv()
# from typing import TypedDict,Annotated
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langgraph.graph.message import add_messages
# from langchain_core.messages import HumanMessage

# llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.4)
# class TravelState(TypedDict):
#     days: str
#     destination:str
#     user_query:str
#     budget:str
#     temperature: str
#     booking:str

# def planner_node(state:TravelState):
#     request= state["user_query"]
#     prompt = f"""
#     Extract the following information from the travel request.
#     Return ONLY JSON.
#     Fields:
#     destination days budget Request:{request}"""

#     response= llm.invoke(prompt)
#     return {"destination":state["destination"], "budget":state["budget"], "no_of_days":state["days"]}

# def weather_node(state:TravelState):
#     weath= state["temperature"]
#     destina= state["destination"]
#     prompt=f"""
#     Provide a short weather overview for travelers visiting
#     {state['destination']}.
#     Give:
#     - Temperature range
#     - Best time for sightseeing
#     - Packing suggestions
#     Keep it concise.
#     """
#     response= llm.invoke(prompt)
#     return {"temper": state["temperature"]}

# # NODE 3 - HOTEL AGENT
# def hotel_node(state: TravelState):
#     prompt = f"""
#     Suggest 5 hotels in {state['destination']}.
#     Budget:
#     {state['budget']}
#     Return:
#     Hotel Name
#     Approx Cost
#     Short Description
#     """
#     response = llm.invoke(prompt)
#     return {
#     "hotels": response.content
#     }

# graph= StateGraph(TravelState)
# app= graph.compile()

# #creating the nodes

# graph.add_node("planner", planner_node)
# graph.add_node("weather", weather_node)
# graph.add_node("hotel",hotel_node)

# #connecting to edges

# graph.add_edge(START, "planner")
# graph.add_edge("planner", "weather")
# graph.add_edge("weather", "hotel")
# graph.add_edge("hotel", END)

# app.invoke({})


#middle ware 
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,SystemMessage
load_dotenv()

# llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

# agent= create_agent(
#     model=llm,
#     checkpointer=InMemorySaver(),
#     middleware=[
#         SummarizationMiddleware(
#             model=llm,
#             trigger=("messages", 10),
#             keep=("messages", 4)
#         )
#     ]
# )
# config= {"configurable": {"thread_id":"test_1"}}

# #alternative test data
# questions= ["what is 2+1", "what is 10/5", "what is 2*2", "what is 2**2","what is 5/5", ]

# for q in questions:
#     response=  agent.invoke({"messages":[HumanMessage(content=q)]},config)
#     print(f"messages: {response}")
#     print(f"messages: {len(response['messages'])}")

# Build a chatbot with langgraph(Graph API)
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware,HumanInTheLoopMiddleware
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing import TypedDict,Annotated
from langchain_tavily import TavilySearch
load_dotenv()

llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)

tool= TavilySearch(max_results=2)

def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b
def weather(city:str,temperature:int,description:str):
    """you are an tourist guide help us to check weather"""
    return {city} and {temperature}
tools= [tool,multiply,weather]

llm_with_tool= llm.bind_tools(tools)

class State:
    messages: Annotated[list,add_messages]

def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

#stategraph
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition

builder= StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"tool_calling_llm" )
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools","tool_calling_llm")

graph= builder.compile()

response= graph.invoke({"messages":"what is 2*3 and trending news in karimnagar and what is the name of the prime minister of india what is the weather condition in '{hyderabad}'"})

for m in response["messages"]:
    print(m.pretty_print())

















