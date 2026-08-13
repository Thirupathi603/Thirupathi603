from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


#prompt template
prompt=ChatPromptTemplate.from_template("explain {topic} in simple words")

#model

model= ChatMistralAI(model_name="mistral-small-latest")

# output parser

parser=StrOutputParser()

chain= prompt | model | parser

result=chain.invoke("topic:golkonda fort ")
print(result)



