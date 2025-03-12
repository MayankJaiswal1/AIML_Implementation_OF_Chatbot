import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL handling for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = "intents.json"
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .chat-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: #555;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Chatbot - NLP Implementation")
st.markdown("Welcome! Ask me anything, and I'll respond intelligently.")

# Sidebar menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    chat_log = []
    user_input = st.text_input("You:", "", key="user_input")
    if user_input:
        response = chatbot(user_input)
        st.success(f"Chatbot: {response}")
        chat_log.append((user_input, response))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

elif choice == "Conversation History":
    st.header("Conversation History")
    if os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    else:
        st.write("No conversation history found.")

elif choice == "About":
    st.write("The goal of this project is to create a chatbot that understands and responds to user input using NLP and Logistic Regression.")
    st.subheader("Project Overview:")
    st.write("""
    - Uses NLP techniques and Logistic Regression for intent recognition.
    - Built with Streamlit for an interactive chatbot interface.
    - Supports real-time user interaction with dynamic responses.
    """)
    
    st.subheader("Dataset:")
    st.write("""
    - Contains labeled intents and entities.
    - Helps classify user inputs into predefined categories.
    """)
    
    st.subheader("Conclusion:")
    st.write("This chatbot can be expanded with more data and advanced AI techniques for improved interaction.")

st.markdown('<div class="footer">Developed by Mayank 🚀</div>', unsafe_allow_html=True)
