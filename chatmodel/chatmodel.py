# from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

llm= ChatMistralAI(model_name="mistral-small")

response=llm.invoke("what is ai")
print(response.content)


