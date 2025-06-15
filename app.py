import streamlit as st
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# LangChain for chatbot
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Sidebar Navigation
st.set_page_config(page_title="Loan App", layout="wide")
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Loan Risk Predictor", "Chat Bot"])

# ---------- Page 1: Loan Risk Predictor ----------
if page == "Loan Risk Predictor":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Loan Risk Prediction Web App</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Predict loan eligibility using credit and customer profile</h4><br>", unsafe_allow_html=True)

    # Load model
    with open("bank_model", "rb") as f:
        model = pickle.load(f)

    index_to_label = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
    label_meaning = {
        "P1": "✅ Safe to give loan",
        "P2": "⚠️ Medium risk",
        "P3": "⚠️ Medium-High risk",
        "P4": "❌ Dangerous to give loan"
    }

    # Input Form
    with st.form("input_form"):
        st.subheader("Enter Customer Details")
        col1, col2 = st.columns(2)

        with col1:
            Tot_Active_TL = st.number_input("Total Active TL", 0)
            Total_TL_opened_L6M = st.number_input("Total TL opened (Last 6M)", 0)
            Tot_TL_closed_L6M = st.number_input("TL closed (Last 6M)", 0)
            Tot_TL_closed_L12M = st.number_input("TL closed (Last 12M)", 0)
            Tot_Missed_Pmnt = st.number_input("Missed Payments", 0)
            CC_TL = st.number_input("Credit Card TL", 0)
            Consumer_TL = st.number_input("Consumer TL", 0)
            Gold_TL = st.number_input("Gold TL", 0)
            Home_TL = st.number_input("Home Loan TL", 0)
            PL_TL = st.number_input("Personal Loan TL", 0)
            Other_TL = st.number_input("Other TL", 0)
            Age_Oldest_TL = st.number_input("Oldest TL Age (months)", 0)
            Age_Newest_TL = st.number_input("Newest TL Age (months)", 0)
            time_since_recent_payment = st.number_input("Months Since Last Payment", 0)
            time_since_recent_deliquency = st.number_input("Months Since Last Delinquency", 0)
            num_times_delinquent = st.number_input("Times Delinquent", 0)
            max_recent_level_of_deliq = st.number_input("Max Recent Delinquency Level", 0)
            num_deliq_6_12mts = st.number_input("Delinquencies (6–12M)", 0)

        with col2:
            max_deliq_6mts = st.number_input("Max Delinquency (6M)", 0)
            max_deliq_12mts = st.number_input("Max Delinquency (12M)", 0)
            num_times_60p_dpd = st.number_input("Times 60+ DPD", 0)
            num_std = st.number_input("Standard TL Count", 0)
            num_std_6mts = st.number_input("Standard TLs (Last 6M)", 0)
            num_sub = st.number_input("Substandard TL Count", 0)
            num_sub_12mts = st.number_input("Substandard TLs (12M)", 0)
            num_dbt = st.number_input("Doubtful TLs", 0)
            num_lss = st.number_input("Loss TLs", 0)
            recent_level_of_deliq = st.number_input("Recent Delinquency Level", 0)
            time_since_recent_enq = st.number_input("Months Since Last Enquiry", 0)
            AGE = st.number_input("Age", 18, 100)
            NETMONTHLYINCOME = st.number_input("Net Monthly Income", 0)
            Time_With_Curr_Empr = st.number_input("Time with Current Employer (months)", 0)
            Credit_Score = st.number_input("Credit Score (300-900)", 300, 900)
            EDUCATION = st.number_input("Education (encoded)", 0)

            marital_status = st.radio("Marital Status", ["Married", "Single"], horizontal=True)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            CC_Flag = st.checkbox("Has Credit Card")
            PL_Flag = st.checkbox("Has Personal Loan")
            HL_Flag = st.checkbox("Has Home Loan")
            GL_Flag = st.checkbox("Has Gold Loan")
            last_enq = st.selectbox("Last Product Enquiry", ["AL", "CC", "ConsumerLoan", "HL", "PL", "others"])
            first_enq = st.selectbox("First Product Enquiry", ["AL", "CC", "ConsumerLoan", "HL", "PL", "others"])

        submitted = st.form_submit_button("🚀 Predict Risk")

    if submitted:
        try:
            encoded = {
                "MARITALSTATUS_Married": marital_status == "Married",
                "MARITALSTATUS_Single": marital_status == "Single",
                "GENDER_F": gender == "Female",
                "GENDER_M": gender == "Male",
                "last_prod_enq2_AL": last_enq == "AL",
                "last_prod_enq2_CC": last_enq == "CC",
                "last_prod_enq2_ConsumerLoan": last_enq == "ConsumerLoan",
                "last_prod_enq2_HL": last_enq == "HL",
                "last_prod_enq2_PL": last_enq == "PL",
                "last_prod_enq2_others": last_enq == "others",
                "first_prod_enq2_AL": first_enq == "AL",
                "first_prod_enq2_CC": first_enq == "CC",
                "first_prod_enq2_ConsumerLoan": first_enq == "ConsumerLoan",
                "first_prod_enq2_HL": first_enq == "HL",
                "first_prod_enq2_PL": first_enq == "PL",
                "first_prod_enq2_others": first_enq == "others"
            }

            input_data = {
                "Tot_Active_TL": Tot_Active_TL,
                "Total_TL_opened_L6M": Total_TL_opened_L6M,
                "Tot_TL_closed_L6M": Tot_TL_closed_L6M,
                "Tot_TL_closed_L12M": Tot_TL_closed_L12M,
                "Tot_Missed_Pmnt": Tot_Missed_Pmnt,
                "CC_TL": CC_TL,
                "Consumer_TL": Consumer_TL,
                "Gold_TL": Gold_TL,
                "Home_TL": Home_TL,
                "PL_TL": PL_TL,
                "Other_TL": Other_TL,
                "Age_Oldest_TL": Age_Oldest_TL,
                "Age_Newest_TL": Age_Newest_TL,
                "time_since_recent_payment": time_since_recent_payment,
                "time_since_recent_deliquency": time_since_recent_deliquency,
                "num_times_delinquent": num_times_delinquent,
                "max_recent_level_of_deliq": max_recent_level_of_deliq,
                "num_deliq_6_12mts": num_deliq_6_12mts,
                "max_deliq_6mts": max_deliq_6mts,
                "max_deliq_12mts": max_deliq_12mts,
                "num_times_60p_dpd": num_times_60p_dpd,
                "num_std": num_std,
                "num_std_6mts": num_std_6mts,
                "num_sub": num_sub,
                "num_sub_12mts": num_sub_12mts,
                "num_dbt": num_dbt,
                "num_lss": num_lss,
                "recent_level_of_deliq": recent_level_of_deliq,
                "time_since_recent_enq": time_since_recent_enq,
                "AGE": AGE,
                "NETMONTHLYINCOME": NETMONTHLYINCOME,
                "Time_With_Curr_Empr": Time_With_Curr_Empr,
                "CC_Flag": int(CC_Flag),
                "PL_Flag": int(PL_Flag),
                "HL_Flag": int(HL_Flag),
                "GL_Flag": int(GL_Flag),
                "Credit_Score": Credit_Score,
                "EDUCATION": EDUCATION
            }

            input_data.update(encoded)
            input_df = pd.DataFrame([input_data])
            proba = model.predict_proba(input_df)[0]
            predicted_class_index = int(np.argmax(proba))
            predicted_label = index_to_label[predicted_class_index]
            predicted_meaning = label_meaning[predicted_label]

            st.success(f"Predicted Risk: **{predicted_label}** - {predicted_meaning}")
            st.balloons()

            st.markdown("---")
            st.markdown("### 🔍 Prediction Probabilities")
            for idx, prob in enumerate(proba):
                label = index_to_label.get(idx, f"Unknown-{idx}")
                meaning = label_meaning.get(label, "Unknown")
                st.info(f"**{label} ({meaning})**: `{prob:.4f}`")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ---------- Page 2: Chat Bot ----------
elif page == "Chat Bot":
    st.title("🤖 Why was my loan rejected?")
    st.markdown("Ask the AI assistant why your loan might not be approved based on your financial profile.")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API key not found. Please check your `.env` file.")
        st.stop()

    # LangChain setup
    llm = ChatOpenAI(api_key=api_key, temperature=0.5, model="gpt-4")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    system_msg = """
    You are a financial assistant helping users understand why their loan application might be denied.
    Base your answers on credit score, missed payments, income, number of open/closed accounts, etc.
    Use a polite and helpful tone, and avoid giving specific financial advice.
    """

    st.markdown("---")
    st.subheader("💬 Chat with AI")

    user_input = st.text_input("Ask a question like: 'Why was my loan rejected?'", key="loan_chat_input")

    if user_input:
        with st.spinner("Thinking... 🤔"):
            prompt = f"{system_msg}\n\nUser: {user_input}"
            response = conversation.predict(input=prompt)
        st.success("AI Response:")
        st.markdown(response)
