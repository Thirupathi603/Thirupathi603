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

# class State(TypedDict):

#     messages: Annotated[list,add_messages]
#     #messages have list type the add_messages function 
#     #in the annotated defines how this state key should be updated
#     #in this case,it appends the messages to the list ,rather than overwriting them
# llm= ChatGroq(model="llama-3.3-70b-versatile")
# def chatbot_node(state:State):
#     """you are an ai chatbot """
#     return {"messages": llm.invoke(state["messages"])}


# graph_builder= StateGraph(State)

# # add nodes
# graph_builder.add_node("llm_bot", chatbot_node)

# # add edges
# graph_builder.add_edge(START, "llm_bot")
# graph_builder.add_edge("llm_bot", END)

# graph=graph_builder.compile()

# response= graph.invoke({"messages":"hii"})
# print(response["messages"][-1].content)

llm= ChatGroq(model="llama-3.3-70b-versatile")
tool= TavilySearch(max_results=2)
tool.invoke("what is the trending news in ai")

#custom function

def multiply(a:int ,b:int)->int:
    """Multiply a and b Args: a(int): first int b(int) 
    second int returns int:output int"""

tools= [tool,multiply]

llm_with_tools= llm.bind_tools(tools)

print(llm_with_tools)

class State(TypedDict):

    messages: Annotated[list,add_messages]
    #messages have list type the add_messages function 
    #in the annotated defines how this state key should be updated
    #in this case,it appends the messages to the list ,rather than overwriting them
llm= ChatGroq(model="llama-3.3-70b-versatile")

#stategraph
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition

def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])] }


builder= StateGraph(State)
builder.add_node("tools_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

#add edges

builder.add_edge(START,"tools_calling_llm")


builder.add_conditional_edges("tools_calling_llm",
#if the latest message(result) from assistant is a toolcall ->tools condition routes to tools
#if the latest message(result) from assistant is not a toolcall ->tools condition routes to end
                              tools_condition
)
builder.add_edge("tools",END)

#compile the graph
graph=builder.compile()

response=graph.invoke({"messages":"what is 2*3 and trending news in karimnagar"
""})
# print(response["messages"][-1])
for m in response["messages"]:
    print(m.pretty_print())













