# app.py

import streamlit as st
import configparser
import logging
import base64
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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

# Remove the large st.markdown CSS block and replace with minimal required styles
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Top Navigation Bar with Logo
# -----------------------------------------
def load_logo_in_sidebar(logo_path):
    """Load and display the logo in the sidebar."""
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.sidebar.markdown(
            f'''
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{encoded_string}" alt="Logo">
                <h1>Protex</h1>
            </div>
            ''',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.sidebar.error(f"Logo file not found at {logo_path}")

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

if 'agent' not in st.session_state:
    st.session_state['agent'] = None

if 'remember_credentials' not in st.session_state:
    st.session_state['remember_credentials'] = False

if 'saved_uri' not in st.session_state:
    st.session_state['saved_uri'] = DEFAULT_NEO4J_URI

if 'saved_user' not in st.session_state:
    st.session_state['saved_user'] = DEFAULT_NEO4J_USER

if 'saved_password' not in st.session_state:
    st.session_state['saved_password'] = DEFAULT_NEO4J_PASSWORD


# -----------------------------------------
# Sidebar - AI Model Configuration
# -----------------------------------------
def configure_ai_model():
    """Configure AI model settings in sidebar"""

    # First load the logo
    logo_image_path = Path('assets/images/logo.png')
    load_logo_in_sidebar(logo_image_path)
    
    st.sidebar.markdown("<h3 class='sidebar-header' style='color: #000000;'>AI Model Selection</h3>", unsafe_allow_html=True)
    
    # Model selection with custom styling
    model_choice = st.sidebar.radio(
        "",
        ["Llama (Free)", "OpenAI (API Key Required)"],
        key="model_choice",
        label_visibility="collapsed"
    )

    config = configparser.ConfigParser()
    config_path = Path("config.ini")
    
    if model_choice == "OpenAI (API Key Required)":
        # Check for existing OpenAI configuration
        if config_path.exists():
            config.read(config_path)
            current_key = config.get('openai', 'OPENAI_API_KEY', fallback='')
        else:
            current_key = ''

        # OpenAI configuration
        openai_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=current_key,
            type="password",
            key="openai_key"
        )
        
        openai_model = st.sidebar.selectbox(
            "OpenAI Model",
            ["gpt-4", "gpt-3.5-turbo"],
            key="openai_model"
        )

        if st.sidebar.button("Save OpenAI Configuration"):
            try:
                if not config_path.exists():
                    st.sidebar.error("config.ini not found")
                    return
                
                # Read existing config or create new sections
                config.read(config_path)
                if not config.has_section('neo4j') or not config.has_section('llama'):
                    st.sidebar.error("Missing required sections in config.ini")
                    return

                config['openai'] = {
                    'OPENAI_API_KEY': openai_key
                }
                config['model_settings'] = {
                    'DEFAULT_MODEL': 'openai'
                }

                with open(config_path, 'w') as f:
                    config.write(f)
                st.sidebar.success("OpenAI configuration saved!")
                
                # Reinitialize agent with new configuration
                st.session_state.agent = CentroidProtexAgent()
                
            except Exception as e:
                st.sidebar.error(f"Error saving configuration: {e}")
    else:
        # Using Llama
        if config_path.exists():
            config.read(config_path)
            config['model_settings'] = {'DEFAULT_MODEL': 'llama'}
            with open(config_path, 'w') as f:
                config.write(f)
            
            # Reinitialize agent with Llama configuration
            st.session_state.agent = CentroidProtexAgent()
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
# -----------------------------------------
# Sidebar - Database Configuration
# -----------------------------------------
def configure_database():
    """Configure database connection in sidebar"""
    
    st.sidebar.markdown("<h3 class='sidebar-header' style='color: #000000;'>Database Configuration</h3>", unsafe_allow_html=True)

    # Remove expander and show fields directly
    st.session_state['saved_uri'] = st.sidebar.text_input(
        "Neo4j URI",
        value=st.session_state['saved_uri'],
        label_visibility='visible'
    )
    st.session_state['saved_user'] = st.sidebar.text_input(
        "Neo4j User",
        value=st.session_state['saved_user'],
        label_visibility='visible'
    )
    st.session_state['saved_password'] = st.sidebar.text_input(
        "Neo4j Password",
        type="password",
        value=st.session_state['saved_password'],
        label_visibility='visible'
    )
    st.session_state['remember_credentials'] = st.sidebar.checkbox(
        "Remember for this session",
        value=st.session_state['remember_credentials']
    )

    connect_clicked = st.sidebar.button("Connect")
    if connect_clicked:
        try:
            # Initialize agent if not exists
            if not st.session_state.get('agent'):
                st.session_state['agent'] = CentroidProtexAgent(config_path='config.ini')
                logger.info("New CentroidProtexAgent created")
            
            agent = st.session_state['agent']
            
            # Update agent's Neo4j credentials
            agent.update_neo4j_connection(
                uri=st.session_state['saved_uri'],
                user=st.session_state['saved_user'],
                password=st.session_state['saved_password']
            )

            if agent.neo4j_helper.is_connected():
                st.sidebar.success("Successfully connected to Neo4j database.")
                st.session_state['neo4j_connected'] = True
            else:
                st.sidebar.error("Failed to connect to Neo4j. Please check your connection settings.")
                st.session_state['neo4j_connected'] = False
        except Exception as e:
            st.sidebar.error(f"An error occurred while trying to connect to Neo4j: {str(e)}")
            logger.error(f"Neo4j Connection Error: {e}")
            st.session_state['neo4j_connected'] = False

    # Move connection status to sidebar
    if st.session_state['neo4j_connected']:
        st.sidebar.markdown("""
            <div class='connection-status'>
                <span class='status-badge connected'>✓ Connected</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
            <div class='connection-status'>
                <span class='status-badge disconnected'>✕ Not Connected</span>
            </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# Add retry logic for timeouts
# -----------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_query_with_retry(agent, query):
    """Process query with retry logic for timeouts"""
    try:
        response = agent.process_query(query)
        if not response or not response.get('response'):
            raise ValueError("Empty response received")
        return response
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise

# -----------------------------------------
# Main Function
# -----------------------------------------
def main():
    # Configure sidebar
    configure_ai_model()
    configure_database()
    
    # Main content area
    with st.container():
        # Query input area
        user_query = st.text_input(
            "",
            key="query_input",
            placeholder="Ask me anything about the data...",
            label_visibility='hidden',
            on_change=lambda: st.session_state.update({'send_clicked': True})
        )
        
        col1, col2, col3 = st.columns([3,1,1])
        with col2:
            send_clicked = st.button("Send", key="send_btn", use_container_width=True) or st.session_state.get('send_clicked', False)
            if send_clicked:
                st.session_state.send_clicked = False  # Reset for next use
        with col3:
            reset_button = st.button("🧹 Reset", key="reset_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Process query
        if send_clicked:
            if not st.session_state['neo4j_connected']:
                st.warning("Please connect to the database first.")
            elif user_query.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                logger.info(f"User Query: {user_query}")

                try:
                    agent = st.session_state.get('agent')
                    if agent is None:
                        raise ValueError("Agent not initialized. Please connect to the database first.")
                        
                    with st.spinner('Processing your query... This may take a moment.'):
                        response = process_query_with_retry(agent, user_query)
                        if response and response.get('response'):
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response['response']}
                            )
                        else:
                            st.error("No response received from agent.")
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {str(e)}")
                    logger.error(f"Processing Error: {e}")
            else:
                st.warning("Please enter a query.")

        # Reset conversation
        if reset_button:
            st.session_state.chat_history = []
            logger.info("Conversation has been reset.")
            st.experimental_rerun()

        # Display chat history
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

if __name__ == "__main__":
    main()
