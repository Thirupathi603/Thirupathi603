from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print
load_dotenv()

#creating a tool
@tool
def get_text_length(text:str) ->int:
    """return the length of given text"""
    return len(text)
tools= {"get_text_length": get_text_length}
llm= ChatMistralAI(model_name="mistral-small-latest")

#binding the tools

llm_with_tool= llm.bind_tools([get_text_length])

message= []
prompt= input("you: ")
query= HumanMessage(prompt)

message.append(query) 
# print(message)   #Human message
result= llm_with_tool.invoke(message)
# print(result)  #Ai message
message.append(result)

if result.tool_calls:
    tool_name= result.tool_calls[0]['name']
    tool_message= tools[tool_name].invoke(result.tool_calls[0])
    # print(tool_message)
    message.append(tool_message)
    # print(message)
res= llm_with_tool.invoke(message)
print(res.content)