from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import os
import logging
import httpx
import socket

# Enable debug logging for httpx
logging.basicConfig(level=logging.DEBUG)

st.title("Meruva GPT using DeepSeek-R1")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

base_url = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
model = OllamaLLM(model="deepseek-r1", base_url=base_url)

chain = prompt | model

question = st.text_input("Enter your question here")

if question:
    try:
        # Try to resolve the Ollama server URL
        socket.gethostbyname(base_url.split('://')[-1].split(':')[0])
        
        formatted_prompt = prompt.format(question=question)
        response = chain.invoke(formatted_prompt)
        st.write(response)
    except socket.gaierror:
        st.write("Unable to resolve Ollama server URL")
    except httpx.ConnectError as e:
        st.write(f"Connection error: {e}")
    except Exception as e:
        st.write(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
