# app.py

import streamlit as st
import configparser
import logging
import base64
from pathlib import Path

from agents.centroid_protex_agent import CentroidProtexAgent

# -----------------------------------------
# Page and Layout Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Protex",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------
# Load and Apply CSS
# -----------------------------------------
def load_css(css_file_path):
    """
    Load external CSS file and inject into the Streamlit app.

    Args:
        css_file_path (str): Path to the CSS file.
    """
    try:
        with open(css_file_path) as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {css_file_path}")

css_path = Path('assets/css/styles.css')
load_css(css_path)

# -----------------------------------------
# Top Navigation Bar with Logo
# -----------------------------------------
def load_logo(logo_path):
    """
    Load and display the company logo in the header.

    Args:
        logo_path (str): Path to the logo image.
    """
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f'''
            <div class="main-header">
                <img src="data:image/png;base64,{encoded_string}" alt="Logo">
                <h1>Protex - Enhanced B2B Knowledge Assistant</h1>
            </div>
            ''',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Logo file not found at {logo_path}")

logo_image_path = Path('assets/images/logo.png')
load_logo(logo_image_path)

# -----------------------------------------
# Logging Configuration
# -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Load Config
# -----------------------------------------
config = configparser.ConfigParser()
config_path = Path('config.ini')
if not config_path.is_file():
    st.error("Configuration file 'config.ini' not found.")
    logger.error("Configuration file 'config.ini' not found.")
    st.stop()

config.read(config_path)

try:
    OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
    DEFAULT_NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
    DEFAULT_NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
    DEFAULT_NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("Configuration error in 'config.ini'. Please check the file.")
    st.stop()

# -----------------------------------------
# Session State Defaults
# -----------------------------------------
if 'neo4j_connected' not in st.session_state:
    st.session_state['neo4j_connected'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'centroid_protex_agent' not in st.session_state:
    st.session_state['centroid_protex_agent'] = None

if 'remember_credentials' not in st.session_state:
    st.session_state['remember_credentials'] = False

if 'saved_uri' not in st.session_state:
    st.session_state['saved_uri'] = DEFAULT_NEO4J_URI

if 'saved_user' not in st.session_state:
    st.session_state['saved_user'] = DEFAULT_NEO4J_USER

if 'saved_password' not in st.session_state:
    st.session_state['saved_password'] = DEFAULT_NEO4J_PASSWORD

# -----------------------------------------
# Sidebar - Database Configuration
# -----------------------------------------
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.header("Database Configuration")

with st.sidebar.expander("Show/Hide Configuration", expanded=not st.session_state['neo4j_connected']):
    st.session_state['saved_uri'] = st.text_input(
        "Neo4j URI",
        value=st.session_state['saved_uri'],
        label_visibility='visible'
    )
    st.session_state['saved_user'] = st.text_input(
        "Neo4j User",
        value=st.session_state['saved_user'],
        label_visibility='visible'
    )
    st.session_state['saved_password'] = st.text_input(
        "Neo4j Password",
        type="password",
        value=st.session_state['saved_password'],
        label_visibility='visible'
    )
    st.session_state['remember_credentials'] = st.checkbox(
        "Remember these credentials for this session",
        value=st.session_state['remember_credentials']
    )

    connect_clicked = st.button("Connect to Neo4j")
    if connect_clicked:
        try:
            if not st.session_state['centroid_protex_agent']:
                agent = CentroidProtexAgent(config_path='config.ini')
                st.session_state['centroid_protex_agent'] = agent
            else:
                agent = st.session_state['centroid_protex_agent']
                logger.info("CentroidProtexAgent retrieved from session state.")

            # Update agent's Neo4j credentials
            agent.update_neo4j_connection(
                uri=st.session_state['saved_uri'],
                user=st.session_state['saved_user'],
                password=st.session_state['saved_password']
            )

            if agent.neo4j_helper.is_connected():
                st.success("Successfully connected to Neo4j database.")
                st.session_state['neo4j_connected'] = True
                if st.session_state['remember_credentials']:
                    # Credentials are already stored in session_state
                    pass
            else:
                st.error("Failed to connect to Neo4j. Please check your connection settings.")
                st.session_state['neo4j_connected'] = False
        except Exception as e:
            st.error("An error occurred while trying to connect to Neo4j.")
            logger.error(f"Neo4j Connection Error: {e}")
            st.session_state['neo4j_connected'] = False

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# Main Content Area
# -----------------------------------------
with st.container():
    # Display connection status
    if st.session_state['neo4j_connected']:
        st.markdown("<p class='status-info'>Connected to Neo4j database ✔</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-info'>Not connected to Neo4j database. Please configure on the left and click 'Connect'</p>", unsafe_allow_html=True)

    # Input Area
    st.markdown("""
    <div class="input-container">
        <p>Enter your query here:</p>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "",
        key="user_query_text",
        placeholder="Type your query...",
        label_visibility='hidden'
    )
    send_button = st.button("Send", key="submit_button")

    if send_button:
        if not st.session_state['neo4j_connected']:
            st.warning("Please connect to the database first.")
        else:
            if user_query.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                logger.info(f"User Query: {user_query}")

                try:
                    agent = st.session_state['centroid_protex_agent']
                    response = agent.process_query(user_query)
                except Exception as e:
                    st.error("An error occurred while processing your query.")
                    logger.error(f"Processing Error: {e}")
                    response = {
                        'response': "An error occurred while processing your query.",
                        'results': None,
                        'analysis': None
                    }

                if response.get('response'):
                    st.session_state.chat_history.append({"role": "assistant", "content": response['response']})
            else:
                st.warning("Please enter a query.")

    # Reset Conversation Button
    reset_clicked = st.button("🧹 Reset Conversation")
    if reset_clicked:
        st.session_state.chat_history = []
        logger.info("Conversation has been reset.")
        st.experimental_rerun()

# -----------------------------------------
# Chat History Display
# -----------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Chat History")

with st.container():
    chat_history_container = st.container()
    with chat_history_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                message_html = f'''
                <div class="chat-container">
                    <div class="chat-message user">
                        <div class="message-avatar">You</div>
                        <div class="message-content">{chat["content"]}</div>
                    </div>
                </div>
                '''
                st.markdown(message_html, unsafe_allow_html=True)
            elif chat["role"] == "assistant":
                content = chat["content"]
                # The content may contain Markdown tables and formatted text
                # We'll render it as markdown within the message content
                message_html = f'''
                <div class="chat-container">
                    <div class="chat-message assistant">
                        <div class="message-avatar">AI</div>
                        <div class="message-content">{content}</div>
                    </div>
                </div>
                '''
                st.markdown(message_html, unsafe_allow_html=True)

    # Make the chat history scrollable
    st.markdown(
        """
        <style>
            .chat-container {
                max-height: 600px;
                overflow-y: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
