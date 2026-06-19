import streamlit as st
import pickle

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 SMS Spam Detector")
st.write(
    "This application uses Machine Learning and TF-IDF vectorization "
    "to classify messages as Spam or Not Spam."
)
st.write("Enter a message and check whether it is Spam or Not Spam.")

message = st.text_area("Type your message here")

if st.button("Predict"):
    if message.strip():
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message.")