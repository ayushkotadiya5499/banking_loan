import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Why Loan Rejected? 🤖", layout="wide")
st.title("🤖 Why was my loan rejected?")
st.markdown("Ask the AI assistant why your loan might not be approved based on your financial profile.")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Please check your `.env` file.")
    st.stop()

# Initialize LangChain LLM
llm = ChatOpenAI(openai_api_key=api_key, temperature=0.5, model_name="gpt-4")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Optional system prompt to guide the LLM
system_msg = """
You are a financial assistant helping users understand why their loan application might be denied.
Base your answers on credit score, missed payments, income, number of open/closed accounts, etc.
Use polite and helpful tone, and avoid giving financial advice.
"""

st.markdown("---")
st.subheader("💬 Chat with AI")

# User input box
user_input = st.text_input("Ask a question like: 'Why was my loan rejected?'", key="loan_chat_input")

if user_input:
    with st.spinner("Thinking... 🤔"):
        # Modify input if needed to ensure it follows our system goal
        prompt = f"{system_msg}\n\nUser: {user_input}"
        response = conversation.predict(input=prompt)
    st.success("AI Response:")
    st.markdown(response)
