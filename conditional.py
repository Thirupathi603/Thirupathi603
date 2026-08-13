import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
load_dotenv()
from typing import TypedDict,Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
#Build a Rag pipeline
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def build_retriever(pdf_path: str):
    loader= PyPDFLoader(pdf_path)
    document= loader.load()
    text_splitter= RecursiveCharacterTextSplitter(chunk_size= 800,chunk_overlap=200)
    chunks= text_splitter.split_documents(documents= document)
    vector_score= FAISS.from_documents(chunks,embeddings)
    return vector_score.as_retriever(search_kwargs={"k":3})
academic_retriever= build_retriever("C:\\Users\\008257\\Downloads\\academy-trust-handbook-2025-effective-from-1-september-2025.371297086.pdf")
fee_retriever= build_retriever("C:\\Users\\008257\\Downloads\\Fees Structure 2024-25.pdf")

#model 
llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.4)
# step 2 create a State(blueprint)
class State(TypedDict):
    programme: str
    messages: Annotated[list,add_messages]
    query: str
    retrieved_context:str

# step3 nodes preparation
def classifier_node(state: State) ->dict:
    """look at the latest user message and decide which path to take"""
    last_message= state["messages"][-1].content
    prompt= (
    "Classify the following student query into exactly one category: "
    "'academic', 'fee', or 'general'.\n\n"

    "Use 'academic' for questions about attendance, exams, grading, credits, "
    "promotion, course structure, summer training, or degree requirements.\n"

    "Use 'fee' for questions about tuition, payment, refund, late charges, "
    "scholarships, or any money-related topic.\n"

    "Use 'general' for greetings, casual talk, or anything not related to "
    "the college rules or fee.\n\n"

    f"Query: {last_message}\n\n"

    "Return only one word: academic, fee, or general."
)
    
    response= llm.invoke(prompt)
    category = response.content.strip().lower()
    if "academic" in category:
        category="academic"
    elif "fee" in category:
        category="fee"
    else:
        category="general"
    return {"query":category}

def academic_rag_node(state:State)->dict:
    """Retrieves relevant chunks from the academics handbook"""
    query= state["messages"][-1].content
    docs=academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state:State)->dict:
    """Retrieves relevant chunks from the fee handbook"""
    query= state["messages"][-1].content
    docs=fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}
def general_node(state:State)-> dict:
    """Answers directly using the llm's own knowledge no retrieval needed"""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student."
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
        f"You are a college assistant helping a {programme} student. "
        f"Use the following context from the official college documents to answer "
        f"the question accurately. If the context mentions specific figures for "
        f"different programmes, highlight the one relevant to {programme} if possible."
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Give a clear, friendly, and precise answer."
    )

    response = llm.invoke(prompt)
    return {"messages":[("ai",response.content.strip())]}

# step4 router function

def router_query(state:State):
    if state["query"]=="academic":
        return "academic_rag"
    elif state["query"]=="fee":
        return "fee_rag"
    else:
        return "general"
    
graph= StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

#edges
graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier",router_query)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response",END)


app= graph.compile()

print(f"\n great you are set as an {"student_programe"} student")
student_programe = input("Enter your programme: ")
while True:
    user_query= input("you: ")
    if user_query.lower() in ["exit","quit"]:
        break
    result = app.invoke({
    "programme": student_programe,
    "messages": [HumanMessage(content=user_query)]})
    print("AI:", result["messages"][-1].content)


