#middle ware 
# In Agentic AI, middleware is a layer that sits between the user, agent, tools,
# and LLM and controls how information flows through the system.


from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware,HumanInTheLoopMiddleware
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.types import Command
load_dotenv()

llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

# agent= create_agent(
#     model=llm,
#     checkpointer=InMemorySaver(),
#     middleware=[
#         SummarizationMiddleware(
#             model=llm,
#             trigger=("messages", 10),
#             keep=("messages", 4)
#         )
#     ]
# )
# config= {"configurable": {"thread_id":"test_1"}}

# #alternative test data
# questions= ["what is 2+1", "what is 10/5", "what is 2*2", "what is 2**2","what is 5/5", ]

# for q in questions:
#     response=  agent.invoke({"messages":[HumanMessage(content=q)]},config)
#     print(f"messages: {response}")
#     print(f"messages: {len(response['messages'])}")

from langgraph.types import interrupt

# HumanIntheloop middleware


def read_email_tool(email_id:str):
    """read the email id"""
    return f" email content for id :{email_id}"

def send_email_tool(recipent:str,subject:str, body:str):
    """mock function to send an email"""
    return f"email sent to {recipent} with '{subject}'"

llm= create_agent(
    model= llm,
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": True,"read_email_tool": False})
]
)

config= {"configurable": {"thread_id": "test-"}}

# step1 request

response = llm.invoke(
    {
        "messages": [
            HumanMessage(
                content='send an email to vthiru133@gmail.com with subject "hello" and body "How are you?"'
            )
        ]
    },
    config=config
)

# Step 2: Approve
if "__interrupt__" in response:
    print("⏸️ Paused! Approving...")

    result = llm.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "edit"}
                ]
            }
        ),
        config=config
    )

print(f"✅ Result: {result['messages'][-1].content}")