import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

##Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With GROQ"

##Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant . Please response to user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    model = ChatGroq(
        groq_api_key=api_key, 
        model_name=llm, 
        temperature=temperature, 
        max_tokens=max_tokens
    )
    output_parser=StrOutputParser()
    chain=prompt|model|output_parser
    answer=chain.invoke({'question':question})
    return answer

st.title("QA Chatbot with GROQ")

##Sidebar for settings
st.sidebar.title("Settings")
api_key = os.getenv("GROQ_API_KEY")


##Drop down to select various model
llm=st.sidebar.selectbox("Select a model",["llama-3.1-8b-instant","llama-3.3-70b-versatile","openai/gpt-oss-120b"])


##Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)##value is default value

##Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide query")