# app.py
import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CSS STYLING ----------------
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: #ffffff;
    }

    /* Center title */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Styled text area */
    .stTextArea textarea {
        background-color: #f5f5f5;
        color: #000000;
        font-size: 18px;
    }

    /* Styled buttons */
    div.stButton > button:first-child {
        background-color: #ff4b5c;
        color: white;
        font-size: 18px;
        height: 50px;
        width: 100%;
        border-radius: 10px;
    }

    /* Sentiment outputs */
    .stSuccess {
        background-color: #00c853 !important;
        color: white !important;
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }

    .stError {
        background-color: #d50000 !important;
        color: white !important;
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }

    .stWarning {
        background-color: #ffab00 !important;
        color: white !important;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- APP TITLE ----------------
st.markdown('<div class="title">🐦 Twitter Sentiment Analysis</div>', unsafe_allow_html=True)
st.write("Classify tweets as **Positive** or **Negative** using ML 🚀")

# ---------------- USER INPUT ----------------
tweet = st.text_area("✍️ Enter your tweet:")

# ---------------- PREDICTION ----------------
if st.button("Predict Sentiment"):
    if tweet.strip():
        X_input = vectorizer.transform([tweet])
        prediction = model.predict(X_input)[0]

        if prediction == 0:
            st.error("🚨 Negative Tweet")
        else:
            st.success("✅ Positive Tweet")
    else:
        st.warning("Please enter a tweet first!")
