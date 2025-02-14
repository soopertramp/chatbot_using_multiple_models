import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os

from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"

# Streamlit App Title
st.set_page_config(page_title="Enhanced Q&A Chatbot With Ollama", layout="centered")
st.title("💬 Q&A Chatbot With Ollama")

# Sidebar for settings
with st.sidebar:

    st.header("🔧 Settings")

    # Select OpenAI model
    engine = st.sidebar.selectbox("Select model", ["Mistral by MistralAI", "Gemma2 by Google", "llama3.2 by MetaAI"])

    st.markdown("---")
    st.write("**About:** This chatbot uses Open Source modes via Ollama.")
    st.markdown("---")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display Chat History
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.text(text)

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:

    # Append user message to chat history
    st.session_state["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Setup Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user queries."),
            ("user", "Question: {question}")
        ]
    )

    def generate_response(question):
        llm=Ollama(model=engine)
        output_parser=StrOutputParser()
        chain=prompt|llm|output_parser
        answer=chain.invoke({'question':question})
        return answer

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = generate_response(user_input)
                st.markdown(response)
                st.session_state["messages"].append(("assistant", response))
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

elif not user_input:
    st.write("Please provide the user input")