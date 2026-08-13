import os
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
llm= ChatMistralAI(model_name="mistral-small")
choice= int(input("tell me your choice:"))
if choice==1:
    mode="you are an funny ai agent"
elif choice==2:ia
    mode= "you are an sad ai agent"
elif choice==3:
    mode="you are an angry ai agent"
messages=[SystemMessage(content=mode)]
while True:
    prompt= input("you:")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    response=llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("bot:",response.content)
print(messages)