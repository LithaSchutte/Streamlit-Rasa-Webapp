import requests
import streamlit as st
import pyttsx3  # Import the pyttsx3 library for TTS

# Define Rasa server URL
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

st.set_page_config(layout="centered")

st.title("Rasa Test with Voice")

# Initialize TTS engine
engine = pyttsx3.init()

# Set properties for the TTS engine (optional)
engine.setProperty('rate', 150)  # Speed of speech (optional)
engine.setProperty('volume', 1)  # Volume (range: 0.0 to 1.0)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello and welcome to the app.\n"
                    "I can assist you in the following ways:\n"
                    "- Provide insights into global health data such as the average, minimum or maximum\n"
                    "- Compare health indicators of different countries.\n"
                    "- Explore health trends over time. \n"
                    "- Explore the correlations between different health indicators\n"
                    "\nPlease tell me which of these tasks I can assist you with. Or did you have something else in mind?"}
    ]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Function to send a message to Rasa
def send_message_to_rasa(message):
    payload = {"sender": "streamlit_user", "message": message}
    response = requests.post(RASA_SERVER_URL, json=payload)
    response.raise_for_status()
    return response.json()


# Function to speak the response
def speak_response(response_text):
    engine.say(response_text)
    engine.runAndWait()


# User input
if user_input := st.chat_input("Type your message here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send user input to Rasa and display bot responses
    try:
        bot_responses = send_message_to_rasa(user_input)
        response_text = "\n".join(response.get("text", "") for response in bot_responses)

        # Make the bot speak the response
        speak_response(response_text)

    except Exception as e:
        response_text = f"Error connecting to Rasa server: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)
