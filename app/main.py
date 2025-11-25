import streamlit as st
import joblib

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


st.title("Spam Detector App")
st.write("Enter an email text below to check if it's spam or not.")

user_input = st.text_area("Email content", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.error("🚨 This email is **SPAM**!")
        else:
            st.success("✅ This email is **NOT spam** (Ham).")
