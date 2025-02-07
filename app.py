# app.py

import streamlit as st
import configparser
import logging
logging.getLogger().setLevel(logging.DEBUG)
import base64
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.neo4j_agent import Neo4jAgent
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
def load_api_config():
    """Load API configuration securely"""
    config = configparser.ConfigParser()
    config_path = Path('config.ini')
    config_exists = config_path.is_file()
    
    if config_exists:
        config.read(config_path)
        return config, True
    return config, False

config = configparser.ConfigParser()
config_path = Path('config.ini')
if not config_path.is_file():
    st.error("Configuration file 'config.ini' not found.")
    logger.error("Configuration file 'config.ini' not found.")
    st.stop()

config.read(config_path)

try:
    OPENAI_API_KEY = config.get('openai', 'openai_api_key')
    DEFAULT_NEO4J_URI = config.get('neo4j', 'neo4j_uri')
    DEFAULT_NEO4J_USER = config.get('neo4j', 'neo4j_user')
    DEFAULT_NEO4J_PASSWORD = config.get('neo4j', 'neo4j_password')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("Configuration error in 'config.ini'. Please check the file.")
    st.stop()

# -----------------------------------------
# Session State Defaults
# -----------------------------------------
if 'neo4j_connected' not in st.session_state:
    st.session_state['neo4j_connected'] = False

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''

if 'api_url' not in st.session_state:
    st.session_state['api_url'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'neo4j_agent' not in st.session_state:
    st.session_state['neo4j_agent'] = None

if 'llm_agent' not in st.session_state:
    st.session_state['llm_agent'] = None

if 'remember_credentials' not in st.session_state:
    st.session_state['remember_credentials'] = False

if 'saved_uri' not in st.session_state:
    st.session_state['saved_uri'] = DEFAULT_NEO4J_URI

if 'saved_user' not in st.session_state:
    st.session_state['saved_user'] = DEFAULT_NEO4J_USER

if 'saved_password' not in st.session_state:
    st.session_state['saved_password'] = DEFAULT_NEO4J_PASSWORD

if 'model_type' not in st.session_state:
    st.session_state['model_type'] = 'llama'

# -----------------------------------------
# Sidebar - AI Model Configuration
# -----------------------------------------
def configure_ai_model():
    """Configure AI model settings in sidebar"""
    logo_image_path = Path('assets/images/logo.png')
    load_logo_in_sidebar(logo_image_path)
    
    st.sidebar.markdown("<p style='margin-bottom: 0.3rem; font-weight: 600; font-size: 1rem;'>Model Selection</p>", unsafe_allow_html=True)
    
    # Get current config before setting model type
    config, config_exists = load_api_config()
    
    # Update model selection to properly set case and update config
    model_selection = st.sidebar.radio(
        "",
        options=["Llama", "OpenAI"],
        index=0 if st.session_state.get('model_type', 'llama').lower() == 'llama' else 1,
        key="model_select",
        label_visibility="collapsed"
    )
    
    # Update model type in session state and config
    st.session_state['model_type'] = model_selection.lower()
    
    # Update config with current model selection
    if 'model_settings' not in config:
        config.add_section('model_settings')
    config['model_settings']['default_model'] = st.session_state['model_type']
    
    # Re-initialize LLM agent when model type changes
    if st.session_state.get('llm_agent') and st.session_state.get('previous_model_type', '') != st.session_state['model_type']:
        st.session_state['llm_agent'] = CentroidProtexAgent(config=config)
        st.session_state['previous_model_type'] = st.session_state['model_type']
        if st.session_state.get('neo4j_agent'):
            st.session_state['llm_agent'].set_neo4j_agent(st.session_state['neo4j_agent'])

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

    # Toggle between Connect and Disconnect button
    if st.session_state.get('neo4j_connected', False):
        if st.sidebar.button("Disconnect"):
            try:
                st.session_state['neo4j_agent'].close()
            except Exception as e:
                st.sidebar.error(f"Error disconnecting: {str(e)}")
            st.session_state['neo4j_connected'] = False
            st.sidebar.info("Disconnected from Neo4j.")
    else:
        if st.sidebar.button("Connect"):
            try:
                # Initialize or update Neo4jAgent
                if not st.session_state.get('neo4j_agent'):
                    # Create Neo4jAgent first
                    st.session_state['neo4j_agent'] = Neo4jAgent(
                        st.session_state['saved_uri'],
                        st.session_state['saved_user'],
                        st.session_state['saved_password']
                    )
                    
                    # Create CentroidProtexAgent with selected model config
                    model_type = st.session_state.get('model_type', 'llama')
                    config_updates = {}
                    if model_type == 'openai':
                        config_updates['openai_api_key'] = st.session_state['api_key']
                    else:
                        config_updates['llama_api_url'] = st.session_state['api_url']
                        config_updates['llama_api_key'] = st.session_state['api_key']
                    
                    # Update config with current settings
                    for key, value in config_updates.items():
                        section = key.split('_')[0]
                        if section not in config:
                            config.add_section(section)
                        config[section][key] = value

                    # Create single instance of CentroidProtexAgent
                    st.session_state['llm_agent'] = CentroidProtexAgent(config_path='config.ini')
                    st.session_state['llm_agent'].set_neo4j_agent(st.session_state['neo4j_agent'])
                    
                else:
                    st.session_state['neo4j_agent'].update_connection(
                        st.session_state['saved_uri'],
                        st.session_state['saved_user'],
                        st.session_state['saved_password']
                    )

                if st.session_state['neo4j_agent'].initialize():
                    st.session_state['neo4j_connected'] = True
                else:
                    st.sidebar.error("Failed to connect to Neo4j.")
                    st.session_state['neo4j_connected'] = False
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")
                st.session_state['neo4j_connected'] = False

    # Display connection status
    status_class = 'connected' if st.session_state['neo4j_connected'] else 'disconnected'
    status_symbol = '✓' if st.session_state['neo4j_connected'] else '✕'
    status_text = 'Connected' if st.session_state['neo4j_connected'] else 'Not Connected'
    
    st.sidebar.markdown(f"""
        <div class='connection-status'>
            <span class='status-badge {status_class}'>{status_symbol} {status_text}</span>
        </div>
    """, unsafe_allow_html=True)
# -----------------------------------------
# Add retry logic for timeouts
# -----------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def process_query_with_retry(neo4j_agent, llm_agent, query):
    """Process query with chain of thought and proper error handling"""
    logging.debug(f"\n[DEBUG] ========= Starting new query processing =========")
    logging.debug(f"[DEBUG] Query: {query}")
    
    if not neo4j_agent or not llm_agent:
        logging.debug("[DEBUG] Agents not initialized")
        raise ValueError("Agents not properly initialized")

    try:
        # Find matching query
        logging.debug("[DEBUG] Calling find_matching_query")
        query_match = neo4j_agent.find_matching_query(query)
        logging.debug(f"[DEBUG] find_matching_query returned type: {type(query_match)}")
        logging.debug(f"[DEBUG] find_matching_query returned content: {query_match}")

        # --- Added conversion if query_match is a string ---
        if isinstance(query_match, str):
            logging.debug("[DEBUG] Converting string query_match to dict")
            try:
                query_match = json.loads(query_match)
                logging.debug("[DEBUG] Successfully converted query_match to dict")
            except Exception as e:
                logging.debug(f"[DEBUG] Failed to parse query_match as JSON: {str(e)}")
                query_match = {}
                
        logging.debug(f"[DEBUG] Final query_match type: {type(query_match)}")
        logging.debug(f"[DEBUG] Final query_match content: {query_match}")
        
        # --- Debug: Log type and content of query_match ---
        logging.debug(f"[DEBUG] query_match type before conversion: {type(query_match)}; content: {query_match}")
        # --- Added conversion if query_match is a string ---
        if isinstance(query_match, str):
            try:
                import json
                query_match = json.loads(query_match)
            except Exception:
                query_match = {}
            logging.debug(f"[DEBUG] query_match converted to dict: {query_match}")
        # ----------------------------------------------------
        if not query_match or not isinstance(query_match, dict):
            return {
                'response': "I couldn't find a matching query pattern for your question.",
                'error': None,
                'chain_of_thought': {
                    'query_understanding': "No matching query pattern found",
                    'matching_score': 0,
                    'cypher_query': None
                }
            }

        # Safely extract query information
        matched_query = query_match.get('matched_query', {})
        if not isinstance(matched_query, dict):
            matched_query = {}
            
        cypher_query = matched_query.get('query', '')
        if not cypher_query:
            return {
                'response': "Invalid query pattern found.",
                'error': 'InvalidQueryPattern',
                'chain_of_thought': None
            }
        
        # Execute the query
        results = neo4j_agent.execute_query(cypher_query)
        
        # Format results with chain of thought
        llm_response = llm_agent.format_results(
            user_query=query,
            query_matching=query_match,
            results=results or []
        )
        # --- Added type check to guard against string output ---
        if isinstance(llm_response, str):
            llm_response = {"formatted_response": llm_response}
        # ----------------------------------------------------------

        # Ensure we have a formatted_response
        formatted_response = llm_response.get('formatted_response', 'No response generated')
        
        # Create chain of thought details
        chain_of_thought = {
            'query_understanding': matched_query.get('description', 'No description available'),
            'matching_score': query_match.get('match_score', 0),
            'cypher_query': cypher_query,
            'considered_matches': query_match.get('considered_matches', [])
        }

        return {
            'response': formatted_response,
            'details': llm_response,
            'error': None,
            'chain_of_thought': chain_of_thought
        }
            
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        return {
            'response': f"An error occurred while processing your query: {str(e)}",
            'error': str(e),
            'chain_of_thought': None
        }

def main():
    # Configure sidebar
    configure_ai_model()
    configure_database()
    
    # Main content area
    with st.container():
        # Query input area - Remove on_change trigger
        user_query = st.text_input(
            "",
            key="query_input",
            placeholder="Ask me anything about the data...",
            label_visibility='hidden'
        )
        
        col1, col2, col3 = st.columns([3,1,1])
        with col2:
            # Simplify send_clicked logic to only use button state
            send_clicked = st.button("Send", key="send_btn", use_container_width=True)
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
                    neo4j_agent = st.session_state.get('neo4j_agent')
                    llm_agent = st.session_state.get('llm_agent')
                    if not neo4j_agent or not llm_agent:
                        raise ValueError("Agents not initialized. Please connect to the database first.")
                        
                    with st.spinner('Processing your query... This may take a moment.'):
                        response = process_query_with_retry(neo4j_agent, llm_agent, user_query)
                        
                        # Display chain of thought
                        if response.get('chain_of_thought'):
                            st.write("### Query Understanding")
                            st.write("Original query:", user_query)
                            st.write("Best matching query:", response['chain_of_thought']['query_understanding'])
                            st.write("Match score:", response['chain_of_thought']['matching_score'])
                            
                            if response['chain_of_thought'].get('cypher_query'):
                                st.write("### Cypher Query")
                                st.code(response['chain_of_thought']['cypher_query'], language='cypher')
                            
                            if response['chain_of_thought'].get('considered_matches'):
                                st.write("### Alternative Matches Considered")
                                for match in response['chain_of_thought']['considered_matches']:
                                    st.write(f"- {match.get('query_name', 'Unknown')} (Score: {match.get('score', 0)})")
                        
                        if response and response.get('response'):
                            st.write("### Results")
                            st.write(response['response'])
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response['response']}
                            )
                        else:
                            st.error("No response received from agents.")
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
