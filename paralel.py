from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model= ChatMistralAI(model_name="mistral-small-latest")
#prompt--model--parser

short_prom= ChatPromptTemplate.from_template("explain {topic} in 1-2 lines")
detailed_prom= ChatPromptTemplate.from_template("explain {topic} in simple words")

parser= StrOutputParser()

chain= RunnableParallel({"short":short_prom| model | parser,
         "detailed": detailed_prom | model | parser })

result=chain.invoke("topic:machine learning")

print(result["short"])
print(result["detailed"])


