from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model= HuggingFaceEmbeddings()

vector_store= Chroma(persist_directory=="chroma-db", embedding_function==embedding_model)

retriever= vector_store.as_retriever(search_type="mmr",search_kwargs= {"k":2,"fetch_k":3 ,"lambda_mult":0.5})

lllm= ChatMistralAI("mistral-small-latest")









