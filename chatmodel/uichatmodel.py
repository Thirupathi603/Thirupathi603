import os
from dotenv import load_dotenv
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize model
llm = ChatMistralAI(model_name="mistral-small")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="you are a angry agent model")
    ]

# Title
st.title("Angry Chatbot")

# Display messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.write("You:", msg.content)
    elif isinstance(msg, AIMessage):
        st.write("Bot:", msg.content)

# Input box
prompt = st.text_input("You:")

if st.button("Send"):
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))

        response = llm.invoke(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=response.content))

        st.write("Bot:", response.content)