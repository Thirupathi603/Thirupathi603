from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from rich import print
load_dotenv()

#creating a tool
@tool
def get_text_length(text:str) ->int:
    """return the length of given text"""
    return len(text)

llm= ChatMistralAI(model_name="mistral-small-latest")

#binding the tools

llm_bind_tool= llm.bind_tools([get_text_length])


result= llm_bind_tool.invoke("return the length of the characters in the given text: 'hi how are you")

print(result.tool_calls[0])

print(get_text_length.invoke({'text': 'hi how are you'}))





