from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os

# Load OpenAI key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the app title
st.title("Meruva GPT using OpenAI")

# Define the prompt template
template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model
model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)

# Create the chain
chain = prompt | model

# Input from user
question = st.text_input("Enter your question here")

if question:
    try:
        formatted_prompt = prompt.format(question=question)
        response = chain.invoke(formatted_prompt)
        st.write(response.content)
    except Exception as e:
        st.write(f"Error: {e}")
