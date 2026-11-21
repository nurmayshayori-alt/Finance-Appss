import streamlit as st
import pandas as pd
import plotly.express as px
import os
from groq import Groq
from dotenv import load_dotenv

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Budget vs. Actuals AI",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# 2. LOAD GROQ API KEY
# =====================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY is missing! Set it in Streamlit Secrets or .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-20b"

# =====================================================
# 3. STREAMLIT APP UI
# =====================================================
st.title("📊 Budget vs. Actuals AI – Variance Analysis & Commentary")
st.write("Upload your Budget vs. Actuals file and get AI-driven financial insights!")

uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        required_columns = ["Category", "Budget", "Actual"]

        if not all(col in df.columns for col in required_columns):
            st.error("⚠️ The uploaded file must contain 'Category', 'Budget', and 'Actual' columns!")
            st.stop()

        df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
        df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")

        # Hitung Variance & Variance %
        df["Variance"] = df["Actual"] - df["Budget"]
        df["Variance %"] = (df["Variance"] / df["Budget"].replace(0, float("nan"))) * 100

        # Tampilkan Data Preview
        st.subheader("📊 Data Preview with Variance Calculation")
        st.dataframe(df, use_container_width=True)

        # KPI Metrics
        total_budget = df["Budget"].sum()
        total_actual = df["Actual"].sum()
        total_variance = total_actual - total_budget

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Budget", f"{total_budget:,.2f}")
        col2.metric("Total Actual", f"{total_actual:,.2f}")
        col3.metric("Total Variance", f"{total_variance:,.2f}")

        # Visualisasi Plotly
        st.subheader("📈 Budget vs. Actual Variance Analysis")
        fig_bar = px.bar(
            df,
            x="Category",
            y="Variance",
            color="Variance",
            title="📊 Variance by Category",
            text_auto=".2s",
            color_continuous_scale=["red", "yellow", "green"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = px.line(
            df,
            x="Category",
            y=["Budget", "Actual"],
            markers=True,
            title="📉 Budget vs. Actual Performance",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # AI Section
        st.subheader("🤖 AI-Powered Variance Analysis")
        with st.spinner("AI is analyzing your variance data..."):
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI financial analyst providing variance analysis insights on budget vs. actuals."
                    },
                    {
                        "role": "user",
                        "content": f"Here is the budget vs. actual variance summary:\n{df.to_string(index=False)}\nWhat are the key insights and recommendations?"
                    }
                ],
                model=MODEL_NAME,
            )
            st.write(response.choices[0].message.content)

        # AI Chat
        st.subheader("🗣️ Chat with AI About Variance Analysis")
        user_query = st.text_input("🔍 Ask the AI about your variance data:")
        if user_query:
            with st.spinner("AI is preparing your answer..."):
                chat_response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI financial analyst helping users understand their budget vs. actual variance analysis."
                        },
                        {
                            "role": "user",
                            "content": f"Variance Data:\n{df.to_string(index=False)}\nQuestion: {user_query}"
                        }
                    ],
                    model=MODEL_NAME,
                )
                st.write(chat_response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error processing file: {e}")
