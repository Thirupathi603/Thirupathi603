from langchain_openai import embeddings 
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

messages=["what is ai","what is datascience"]

vector=llm.embed_documents(messages)
print(vector)





