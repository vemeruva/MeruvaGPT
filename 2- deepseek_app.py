from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import os

st.title("Meruva GPT using DeepSeek-R1")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Try setting base_url directly or use environment variable
base_url = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
model = OllamaLLM(model="deepseek-r1", base_url=base_url)

chain = prompt | model

question = st.text_input("Enter your question here")

if question:
    try:
        formatted_prompt = prompt.format(question=question)
        response = chain.invoke(formatted_prompt)
        st.write(response)
    except Exception as e:
        st.write(f"Error: {e}")
        # Optionally, add more error details for debugging
        import traceback
        st.code(traceback.format_exc())
