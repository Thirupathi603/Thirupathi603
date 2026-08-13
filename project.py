import os

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
#lets create the states first

class pipelinestate(TypedDict):
    raw_input: str
    edited_text:str
    script_text: str
    final_output:str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

#creating multiple nodes

def editor_node(state:pipelinestate)-> dict:
    """step1: cleans up grammer, remove typos,and refines the tone"""
    prompt= ("you are an expert copyeditor. clean up the following raw text."
    "fix the grammatical errors,spelling mistakes and smooth out the transistion flow."
    "while keeping the core message intact. return only the edited text. \n\n"
    f"Text:\n{state['raw_input']}"
    )
    response=llm.invoke(prompt)
    return {"edited_text":response.content.strip()}

def script_node(state:pipelinestate)-> dict:
    """step2: format the clean text into an engaging video scrpt style."""
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")
    prompt = ("You are a charismatic YouTube content creator. Take this edited text and transform "
    "it into a highly engaging, punchy, conversational video script hook. Make it sound "
    "like a real person speaking passionately. Return only the script content.\n\n"

    f"Edited Text:\n{state['edited_text']}")
    response=llm.invoke(prompt)
    return {"script_text":response.content.strip()}

def translator_node(state:pipelinestate)->dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")
    prompt = (
    "You are an expert content localizer for the Indian market. Take the following script "
    "and convert it into natural, flowing 'Telglish'. Do not simply translate it sentence-by-sentence "
    "or repeat information. Alternating comfortably between Telugu and English phrases in the way "
    "an intellectual tech educator would speak naturally on a live stream. Keep the energy high. "

    "Return only the final Hinglish text.\n\n"

    f"Script:\n{state['script_text']}")
    response=llm.invoke(prompt)
    return {"final_output":response.content.strip()}


#now your state,nodes are ready and now its time to crate graph
#and for creating the graph you have to connect these nodes and for that you have
#to use the edges
#edges are very important to create the workflows

from langgraph.graph import StateGraph, START, END
#create the graph
graph= StateGraph(pipelinestate)

#now add the nodes to the graph
graph.add_node("editor",editor_node)
graph.add_node("script",script_node)
graph.add_node("translator",translator_node)

#Add edges (Sequential -one after another)

graph.add_edge(START, "editor")
graph.add_edge("editor","script")
graph.add_edge("script","translator")
graph.add_edge("translator",END)

#compile the graph

app= graph.compile()
result=app.invoke({"raw_input":"Ai agents are the future of tech they can think,plan and act on their own"})

#output
print(result["final_output"])