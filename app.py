import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Load the model
model_path = 'C:/Users/gchar/Major_Project/logistic_regression_model.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Model file not found.")
    st.stop()

def run():
    st.set_page_config(page_title="Loan Prediction", page_icon="🏦", layout="wide")

    # Sidebar for specific inputs
    st.sidebar.title("Loan Application Form 📋")

    # Sidebar fields
    account_no = st.sidebar.text_input('Account Number 📑', placeholder='Enter your account number')
    fn = st.sidebar.text_input('Full Name 👤', placeholder='Enter your full name')

    # Gender
    gen_display = ('Female 👩', 'Male 👨')
    gen_options = list(range(len(gen_display)))
    gen = st.sidebar.selectbox("Gender 🌟", gen_options, format_func=lambda x: gen_display[x])

    # Marital Status
    mar_display = ('No 💔', 'Yes 💍')
    mar_options = list(range(len(mar_display)))
    mar = st.sidebar.selectbox("Marital Status 💑", mar_options, format_func=lambda x: mar_display[x])

    # Number of Dependents
    dep_display = ('No 👶', 'One 👶', 'Two 👶👶', 'More than Two 👨‍👩‍👧‍👦')
    dep_options = list(range(len(dep_display)))
    dep = st.sidebar.selectbox("Dependents 👨‍👩‍👧", dep_options, format_func=lambda x: dep_display[x])

    # Main panel for remaining inputs and results
    st.title("Loan Prediction Form 🏦")

    # Remaining inputs
    edu_display = ('Not Graduate 🎓', 'Graduate 🎓')
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox("Education 🎓", edu_options, format_func=lambda x: edu_display[x])

    emp_display = ('Job 💼', 'Business 🏢')
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox("Employment Status 💼", emp_options, format_func=lambda x: emp_display[x])

    prop_display = ('Rural 🌾', 'Semi-Urban 🌇', 'Urban 🏙️')
    prop_options = list(range(len(prop_display)))
    prop = st.selectbox("Property Area 🏡", prop_options, format_func=lambda x: prop_display[x])

    cred_display = ('Between 300 to 500 📉', 'Above 500 📈')
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox("Credit Score 💳", cred_options, format_func=lambda x: cred_display[x])

    mon_income = st.number_input("Applicant's Monthly Income 💵", value=0)
    co_mon_income = st.number_input("Co-Applicant's Monthly Income 💴", value=0)
    loan_amt = st.number_input("Loan Amount 💰", value=0)

    dur_display = ['2 Months ⏳', '6 Months 🕑', '8 Months 🕑', '1 Year 🗓️', '16 Months 🗓️']
    dur_options = range(len(dur_display))
    dur = st.selectbox("Loan Duration ⏱️", dur_options, format_func=lambda x: dur_display[x])

    if st.button("Submit 📝"):
        duration = [60, 180, 240, 360, 480][dur]
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        prediction = model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))

        if ans == 0:
            st.error(
                f"Hello {fn}! 👋\n"
                f"Account Number: {account_no} 🔢\n"
                f"According to our calculations, you will not get the loan from the bank. 🚫"
            )
        else:
            st.success(
                f"Hello {fn}! 🎉\n"
                f"Account Number: {account_no} 🔢\n"
                f"Congratulations!! You will get the loan from the bank. ✅"
            )

run()
