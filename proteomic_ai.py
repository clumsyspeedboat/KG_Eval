import streamlit as st
import configparser
import openai
from openai import OpenAI

# Load environment variables from the .ini file
config = configparser.ConfigParser()
config.read('.ini')

# Access the API key from the environment
OPENAI_API_KEY = config.get('api_key', 'OPENAI_API_KEY')

# Initialize the OpenAI client
openai.api_key = OPENAI_API_KEY

st.title('Proteomic-AI')

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

