from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

search_tool= TavilySearchResults(max_results=5)

llm= ChatMistralAI(model_name="mistral-small-latest")

prompt=  ChatPromptTemplate.from_template("""you are an ai assistant summarizes the news into the bullet points{news} """)  
parser= StrOutputParser()
chain= prompt | llm | parser
news_result=search_tool.run("latest news of karimnagar telangana")
result=chain.invoke({"news": news_result})
print(result)
