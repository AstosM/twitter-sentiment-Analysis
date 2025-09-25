# 🐦 Twitter Sentiment Analysis Web App

This is a **Twitter Sentiment Analysis** web application built using **Python**, **Scikit-learn**, and **Streamlit**.  
The app allows users to enter any tweet and predicts whether the sentiment is **Positive** or **Negative**.  

---

## 🔹 Features

- Classifies tweets as **Positive** ✅ or **Negative** 🚨
- Uses a **Logistic Regression model** trained on the Sentiment140 dataset
- Preprocessing includes:
  - Lowercasing
  - Removing special characters
  - Stopwords removal
  - Stemming
- Stylish and interactive **Streamlit UI** with gradient background and colored outputs

---

## 🔹 How It Works

1. **Input Tweet**: Enter a tweet into the text area.
2. **Predict**: Click the "Predict Sentiment" button.
3. **Result**: The app shows whether the tweet is positive or negative.

The model was trained locally using the **Sentiment140 Kaggle dataset**, and the trained files (`sentiment_model.pkl` and `vectorizer.pkl`) are included in the repository.  

---

