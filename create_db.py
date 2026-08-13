#load the pdf
#split into chunks
#create an embeddings
#store it in chroma db
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

loader= PyPDFLoader("C:\\Users\\008257\\Downloads\\Appointment Letter (1)33 (1).pdf")
docs= loader.load()
splitter= RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap= 20)

chunks= splitter.split_documents(docs)

embeddings= HuggingFaceEmbeddings()
vectorstore= Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory="chroma_db")
