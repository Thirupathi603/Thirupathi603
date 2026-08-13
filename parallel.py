import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
load_dotenv()

from typing import TypedDict,Annotated
def merge_score_dicts(existing:dict,new_update:dict)->dict:
    if existing ==None:
        return new_update
    return {**existing,**new_update}

    
class Analyzerstate(TypedDict):
    raw_text: str
    safety_scores:Annotated[dict[str,int],merge_score_dicts] 

#nodes

def toxicity_node(state:Analyzerstate )->dict:
    print("\n🤬 [Branch 1] Analyzing Toxicity and Hate Speech...")
  
    prompt = (
    "Analyze the following text for profanity, aggression, hate speech, or toxicity.\n"
    "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic.\n"
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score= int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_scores": {"toxicity_level":score}}

def copyright_node(state:Analyzerstate )->dict:
    print("\n📝 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
    "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
    "or presents a corporate trademark risk. Provide a score from 0 to 100, "
    "where 0 means entirely original and 100 means high risk.\n"
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score= int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_scores": {"copyright_riskscore":score}}


def culture_node(state:Analyzerstate )->dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
    "Analyze the following text for regional sensitivities, political landmines, "
    "or cultural insensitivity that might offend a global audience. Provide a score "
    "where 0 means completely safe and 100 means highly offensive. "
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score= int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_scores": {"cultural_score":score}}


builder= StateGraph(Analyzerstate)

builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_node",copyright_node)
builder.add_node("culture_node",culture_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")
builder.add_edge("toxicity_node",END)
builder.add_edge("copyright_node",END)
builder.add_edge("culture_node",END)

llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)
app=builder.compile()

sample_script = """
Hey team, today's session covers network security fundamentals.
However, I think some of the old security practices are outdated.
Please provide respectful feedback and avoid using offensive language
when discussing technical approaches.
"""

initial_state = {
"raw_text": sample_script,
"safety_scores": {} # Initialized as an empty dictionary
}
final_state = app.invoke(initial_state)
print(final_state["safety_scores"])

