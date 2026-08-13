from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

data=PyPDFLoader("C:\\Users\\008257\\Downloads\\EmployeeHandbook.pdf")
docs= data.load()

splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks= splitter.split_documents(docs)

embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db= Chroma.from_documents(documents=chunks,embedding=embedding,persist_directory="chr_db")

print("db created successfully")

retriever= vector_db.as_retriever(search_type="mmr",search_kwargs={"k":6, "fetch_k":20,"lambda_mult":0.5})

# query = "what is Maternity Leave"

# # Retrieve Top Chunks
# retrieved_docs=retriever.invoke(query)

# # for r in response:
# #     print(r.page_content)

# # Convert Documents into Context String
# context = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize LLM

llm= ChatMistralAI(model_name="mistral-small-latest")

#continues chat bot

while True:
    query= input("you: ")
    if query=="0":
        break
    docs= retriever.invoke(query)
    context= "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Answer only from the provided context.

    Context:
    {context}

    Question:
    {query}
    """
    response=llm.invoke(prompt)
    
    print("\nBot:", response.content)
