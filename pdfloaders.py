from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

data= PyPDFLoader("C:\\Users\\008257\\Downloads\\Appointment Letter (1)33 (1).pdf")
docs= data.load()
splitter= TokenTextSplitter(chunk_size=50,chunk_overlap=10)

chunks= splitter.split_documents(docs)

print(chunks[0].page_content)









