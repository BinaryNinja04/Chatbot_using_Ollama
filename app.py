import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] ="A Chatbot With OLLAMA"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant, please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A chatbot with Llama")

## Drop down to select various OpenAI models
engine=st.sidebar.selectbox("Select an LLM model",["llama3.2"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main Interface
st.write("Go ahead and ask something...")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")