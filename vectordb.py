from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers import MultiQueryRetriever



from dotenv import load_dotenv

load_dotenv()

doc= [Document(page_content= "python is widely used in artificial intelligence.",metadata={"source":"Ai_book"}),
    Document(page_content="pandas is widely used in python.",metadata={"source":"Datascience_book"}),
    Document(page_content= "neural networks widely used in deeplearning.",metadata={"source":"dl_book"})]


# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectordb= Chroma.from_documents(documents=doc , embedding=embeddings, persist_directory="chroma-db")


retriever= vectordb.as_retriever(search_type= "mmr",search_kwargs={"k":1})  #Maximum Marginal Relevance diversed documents

llm= ChatMistralAI(model_name="mistral-small-latest")

result= retriever.invoke("what is used in deep learning")


# result=vectordb.similarity_search("what is used in deep",k=1)


for r in result:
    print(r)
