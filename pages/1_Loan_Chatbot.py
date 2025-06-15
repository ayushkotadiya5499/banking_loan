# pages/1_💬_Loan_Chatbot.py
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

st.title("💬 Ask Why Loan Was Rejected")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Please set your OpenAI API key in the .env file.")
else:
    llm = OpenAI(openai_api_key=api_key, temperature=0.7)
    template = PromptTemplate(
        input_variables=["issue"],
        template="Explain in simple terms why the following loan application may have been rejected: {issue}"
    )
    chain = LLMChain(llm=llm, prompt=template)

    user_input = st.text_area("Describe your loan application issue:")
    if st.button("Ask LLM"):
        if user_input:
            response = chain.run(user_input)
            st.success(response)
        else:
            st.warning("Please enter your loan issue first.")
