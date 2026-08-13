
import streamlit as st
from hr import get_answer

st.title("📘 Employee Handbook Chatbot")

query = st.text_input("Ask a question")

if query:

    answer = get_answer(query)

    st.write(answer)
