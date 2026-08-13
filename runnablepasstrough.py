from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm= ChatMistralAI(model_name="mistral-small-latest")
#codeprompt |llm | strout | explainprompt | llm | strout 

code_prom= ChatPromptTemplate.from_messages([("system","you are a code generator"),
("human","{topic}")])
explain_prom= ChatPromptTemplate.from_messages([("system","you are an helpfull ai assistant who explains code in simple terms"),
("human","explain the following code in simple words: \n{code}")])

parser=  StrOutputParser()

seq= code_prom | llm | parser
seq2= RunnableParallel({"code":RunnablePassthrough(),"explaination":explain_prom | llm | parser })

chain= seq | seq2

result=chain.invoke({"topic": "explain palindrome code"})

print(result["code"])
print(result["explaination"])                     
              
