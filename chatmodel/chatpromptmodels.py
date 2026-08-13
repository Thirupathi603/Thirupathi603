# from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

llm= ChatMistralAI(model_name="mistral-small")

messages=[]
while True:
    prompt= input("you:")
    messages.append(prompt)
    if prompt=="0":
        break
    response=llm.invoke(messages)
    messages.append(response.content)
    print("bot:",response.content)
print(messages)




