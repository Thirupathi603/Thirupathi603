from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import  TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,ToolMessage,SystemMessage
from langchain.tools import tool
from rich import print
from langchain.agents import create_agent
load_dotenv()
import os
import requests

@tool
def get_weather(city: str) ->str:
    """get the weather of the city"""
    api_key =os.getenv("OPENWEATHER_API_KEY")

    url=f"https://api.openweathermap.org/data/2.5/weather" f"?q={city}&appid={api_key}&units=metric"

    response= requests.get(url)
    data= response.json()
    print(data)
    
    return f"""
    City: {data['name']}
    Temperature: {data['main']['temp']}°C

    Humidity: {data['main']['humidity']}%

    Weather: {data['weather'][0]['description']}"""
print(get_weather.invoke("hyderabad"))

@tool
def get_news(city:str) ->str:
    """get the updated news from the city"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response=client.search(query="latest news",
                        search_depth="basic",max_results=5) 
    news = []


    for item in response["results"]:
        news.append(

        f"Title: {item['title']}\n" f"URL: {item['url']}\n")

        return "\n\n".join(news)
    
llm= ChatMistralAI(model_name="mistral-small-latest")
agent=create_agent(llm, tools=[get_news,get_weather],system_prompt="you are an helful assistant")

# result=agent.invoke({"messages":[{"role":"user","content":"what is the weather condition and latest news in hyderabad"}]})

while True:
    user= input("you :")
    if user.lower==exit:
        break
    result=agent.invoke({"messages":[{"role":"user","content":user}]})
    print(result)
    
