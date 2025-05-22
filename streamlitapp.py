import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- LOGIN CONFIGURATION ---
USERNAME = "media@firsteconomy.com"
PASSWORD = "Pixel_098"

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Login Dashboard", layout="wide")

# --- SESSION STATE TO TRACK LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN PAGE ---
def login_page():
    st.markdown("""
        <style>
            .main {
                background-color: #f0f2f6;
                padding-top: 100px;
                text-align: center;
            }
            .stTextInput > div > div > input {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## \U0001F4C8 Media Dashboard Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

# --- DASHBOARD PAGE ---
def dashboard():
    st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F4CA Media Performance Dashboard")
    st.markdown("Welcome, **media@firsteconomy.com**!")

    # --- GENERATE RANDOM DATA ---
    np.random.seed(42)
    df1 = pd.DataFrame({
        'Campaign': [f'Campaign {i}' for i in range(1, 11)],
        'Clicks': np.random.randint(100, 1000, 10),
        'Impressions': np.random.randint(1000, 10000, 10),
        'Conversions': np.random.randint(10, 100, 10),
    })

    df2 = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=10),
        'Spend': np.random.uniform(100, 1000, 10),
        'Revenue': np.random.uniform(200, 2000, 10),
    })

    # --- SHOW TABLES ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Campaign Data")
        st.dataframe(df1, use_container_width=True)
    with col2:
        st.subheader("Financial Data")
        st.dataframe(df2, use_container_width=True)

    # --- PLOTS ---
    chart1 = alt.Chart(df1).mark_bar().encode(
        x='Campaign',
        y='Clicks',
        color=alt.Color('Campaign', legend=None)
    ).properties(title='Clicks by Campaign')

    chart2 = alt.Chart(df2).mark_line(point=True).encode(
        x='Date',
        y='Revenue',
        tooltip=['Date', 'Revenue']
    ).properties(title='Revenue Over Time')

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)

# --- MAIN ---
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
