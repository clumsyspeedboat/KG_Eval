# proteomic_ai.py

import streamlit as st
import configparser
import logging
import pandas as pd
import re
import os
import base64

from pathlib import Path

# Import custom modules
from utils.neo4j_helper import Neo4jHelper
from utils.openai_api import OpenAIChat
from utils.context_loader import load_context
from utils.query_matcher import find_best_query_match

# -----------------------------
# Function Definitions
# -----------------------------

def load_css(file_path):
    """
    Load CSS from a file.

    Args:
        file_path (str): Path to the CSS file.

    Returns:
        str: CSS content or empty string if file not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            css = f.read()
        return css
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")
        logger.error(f"CSS file not found: {file_path}")
        return ""
    except Exception as e:
        st.error(f"Error loading CSS file: {file_path}")
        logger.error(f"Error loading CSS file: {file_path} | Exception: {e}")
        return ""

def load_html(file_path):
    """
    Load HTML from a file.

    Args:
        file_path (str): Path to the HTML file.

    Returns:
        str: HTML content or empty string if file not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return html
    except FileNotFoundError:
        st.error(f"HTML file not found: {file_path}")
        logger.error(f"HTML file not found: {file_path}")
        return ""
    except Exception as e:
        st.error(f"Error loading HTML file: {file_path}")
        logger.error(f"Error loading HTML file: {file_path} | Exception: {e}")
        return ""

def encode_image_to_base64(image_path):
    """
    Encode an image to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image or empty string if file not found.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        st.error(f"Logo image file not found: {image_path}")
        logger.error(f"Logo image file not found: {image_path}")
        return ""
    except Exception as e:
        st.error(f"Error encoding logo image: {image_path}")
        logger.error(f"Error encoding logo image: {image_path} | Exception: {e}")
        return ""

def embed_logo_in_html(html_content, logo_base64):
    """
    Embed the base64-encoded logo into the HTML content.

    Args:
        html_content (str): Original HTML content with a placeholder for the logo.
        logo_base64 (str): Base64-encoded string of the logo image.

    Returns:
        str: Modified HTML content with the embedded logo.
    """
    if logo_base64:
        data_uri = f"data:image/png;base64,{logo_base64}"
        # Replace the img tag's src with the data URI
        modified_html = html_content.replace('<img id="protex-logo" alt="Protex Logo">', f'<img src="{data_uri}" alt="Protex Logo">')
        return modified_html
    else:
        # If logo not found or failed to encode, return original HTML without the logo
        logger.warning("Logo Base64 string is empty. Returning HTML without the logo.")
        return html_content

def extract_cypher_query(response):
    """
    Extract the Cypher query enclosed in triple backticks from the assistant's response.

    Args:
        response (str): The assistant's response.

    Returns:
        str or None: The extracted Cypher query or None if not found.
    """
    pattern = r'```cypher\s*(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def extract_disease_from_query(user_query):
    """
    Extracts the disease name from the user's query.

    Args:
        user_query (str): The user's natural language query.

    Returns:
        str or None: The extracted disease name or None if not found.
    """
    pattern = r'(?:have|with)\s+([\w\s\'-]+)[\?\.]?'
    match = re.search(pattern, user_query, re.IGNORECASE)
    if match:
        disease = match.group(1).strip()
        return disease
    else:
        return None

@st.cache_resource
def load_cached_context(folder_path, as_dict=False):
    """
    Load context and predefined queries with caching.

    Args:
        folder_path (str): Path to the folder containing context or queries.
        as_dict (bool): Whether to load queries as a dictionary.

    Returns:
        Any: Loaded context or queries.
    """
    return load_context(folder_path, as_dict)

# -----------------------------
# Set Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Protex",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Load configuration from config.ini
# -----------------------------
config = configparser.ConfigParser()
config_path = Path('config.ini')
if not config_path.is_file():
    st.error("Configuration file 'config.ini' not found.")
    logger.error("Configuration file 'config.ini' not found.")
    st.stop()

config.read(config_path)

# Access OpenAI API key
try:
    OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("OpenAI API key is not configured properly in 'config.ini'.")
    st.stop()

# -----------------------------
# Initialize session state variables
# -----------------------------
if 'neo4j_connected' not in st.session_state:
    st.session_state['neo4j_connected'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# -----------------------------
# Load CSS and HTML
# -----------------------------
css_path = Path('assets/css/styles.css')
css = load_css(css_path)
if css:
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

header_html_path = Path('assets/html/header.html')
logo_image_path = Path('assets/images/logo.png')
header_html = load_html(header_html_path)
if header_html:
    logo_base64 = encode_image_to_base64(logo_image_path)
    modified_header_html = embed_logo_in_html(header_html, logo_base64)
    st.markdown(modified_header_html, unsafe_allow_html=True)

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2>Settings</h2>
    """, unsafe_allow_html=True)

    # Allow user to input Neo4j host and port
    neo4j_host = st.text_input("Neo4j Host", value="localhost")
    neo4j_port = st.number_input("Neo4j Port", min_value=1, max_value=65535, value=7687)
    NEO4J_URI = f"bolt://{neo4j_host}:{int(neo4j_port)}"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = st.text_input("Neo4j Password", type="password")

    # Connect button
    if st.button("Connect to Neo4j"):
        if NEO4J_PASSWORD:
            try:
                # Attempt to connect to Neo4j
                neo4j_helper = Neo4jHelper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                if neo4j_helper.is_connected():
                    st.success("Successfully connected to Neo4j database.")
                    st.session_state['neo4j_connected'] = True
                    # Store the helper in session state
                    st.session_state['neo4j_helper'] = neo4j_helper
                else:
                    st.error("Failed to connect to Neo4j. Please check your connection settings.")
                    st.session_state['neo4j_connected'] = False
            except Exception as e:
                st.error("An error occurred while trying to connect to Neo4j.")
                logger.error(f"Neo4j Connection Error: {e}")
                st.session_state['neo4j_connected'] = False
        else:
            st.warning("Please enter the Neo4j password.")

    # Response temperature for OpenAI API
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, 0.05)

    # Reset conversation button
    if st.button("🧹 Reset Conversation"):
        st.session_state.chat_history = []
        logger.info("Conversation has been reset.")
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Main Application Logic
# -----------------------------
# Check if connected to Neo4j before proceeding
if st.session_state['neo4j_connected']:
    # Initialize OpenAI Chat helper
    openai_chat = OpenAIChat(api_key=OPENAI_API_KEY)

    # Load context and predefined queries with caching
    context_folder = "./resources/context_files/"
    queries_folder = "./resources/queries/"
    context = load_cached_context(context_folder)
    predefined_queries = load_cached_context(queries_folder, as_dict=True)

    # Input area
    with st.container():
        st.markdown("""
        <div class="input-container">
            <p style='text-align: center; font-size: 1.1em;'>
                <strong>Protex</strong> - Connect to the database and get your proteomic questions answered!
            </p>
            <div class="query-input">
        """, unsafe_allow_html=True)

        user_query = st.text_input("Enter your query here:", key="user_query")
        if st.button("Send", key="submit_button"):
            if user_query.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                logger.info(f"User Query: {user_query}")

                # Find best query match
                try:
                    best_match = find_best_query_match(user_query, predefined_queries)
                except Exception as e:
                    st.error("An error occurred while matching your query.")
                    logger.error(f"Query Matching Error: {e}")
                    best_match = None

                if best_match:
                    if 'query_template' in best_match:
                        # Extract the disease name from the user's query
                        disease = extract_disease_from_query(user_query)
                        if disease:
                            try:
                                cypher_query = best_match['query_template'].format(disease=disease)
                                logger.info(f"Using matched query: {best_match['description']}")
                            except Exception as e:
                                st.error("Error formatting the Cypher query.")
                                logger.error(f"Cypher Query Formatting Error: {e}")
                                cypher_query = None
                        else:
                            st.warning("Please specify the disease.")
                            st.session_state.chat_history.append({"role": "assistant", "content": "Please specify the disease you are interested in."})
                            cypher_query = None
                    else:
                        # Use the matched query
                        cypher_query = best_match['query']
                        logger.info(f"Using matched query: {best_match['description']}")
                        assistant_response = f"Here is the Cypher query based on your request:\n```cypher\n{cypher_query}\n```"
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                else:
                    # Generate a new query using OpenAI
                    logger.info("No good match found. Generating query using OpenAI.")

                    # Updated system prompt
                    system_prompt = (
                        "You are an AI assistant that translates user queries into Cypher queries for a Neo4j database. "
                        "The database contains information about proteins, patients, diseases, and treatments. "
                        "When the user's query is related to this database, generate an appropriate Cypher query and enclose it within triple backticks ``` with 'cypher' specified (e.g., ```cypher\nYOUR_QUERY\n```). "
                        "If the user's query is not related to this database, politely inform them that the database only contains information about proteins, patients, diseases, and treatments, and ask them to provide a relevant query. "
                        "Examples of user queries and corresponding Cypher queries:\n"
                        "- User: 'Show me all proteins involved in remission.'\n"
                        "- Cypher: ```cypher\nMATCH (p:Protein)-[:INVOLVED_IN]->(bp:BiologicalProcess {name: 'Remission'}) RETURN p```\n"
                        "- User: 'How many patients have Crohn's disease?'\n"
                        "- Cypher: ```cypher\nMATCH (pat:Patient) WHERE pat.disease = 'Crohn disease' RETURN COUNT(pat) AS NumberOfPatients```\n"
                        "Be organized in your response; use well-formatted tables when presenting data. You may add a brief opinion at the end if appropriate. "
                        "Do not include any explanations before or after the Cypher query."
                    )

                    try:
                        assistant_response = openai_chat.generate_query(system_prompt, user_query)
                        logger.info("Received assistant's response from OpenAI.")
                    except Exception as e:
                        st.error("Failed to generate a response due to an API error.")
                        logger.error(f"OpenAI API Error: {e}")
                        st.stop()

                    # Try to extract the Cypher query from the assistant's response
                    cypher_query = extract_cypher_query(assistant_response)
                    if cypher_query:
                        st.session_state.chat_history.append({"role": "assistant", "content": f"```cypher\n{cypher_query}\n```"})
                        logger.info(f"Generated Cypher Query: {cypher_query}")
                    else:
                        # The assistant's response does not contain a Cypher query
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        logger.warning("No Cypher query found in the assistant's response.")
                        cypher_query = None

                # Run the Cypher query if available
                if cypher_query:
                    # Retrieve neo4j_helper from session_state
                    neo4j_helper = st.session_state.get('neo4j_helper')
                    if neo4j_helper:
                        try:
                            results = neo4j_helper.run_query(cypher_query)
                            logger.info(f"Cypher Query Executed: {cypher_query}")
                        except Exception as e:
                            st.error("An error occurred while executing the query.")
                            logger.error(f"Cypher Query Execution Error: {e}")
                            st.session_state.chat_history.append({"role": "assistant", "content": "An error occurred while executing the query."})
                            results = None

                        if results is not None:
                            if len(results) > 0:
                                # Handle patient count queries
                                if 'NumberOfPatients' in results[0]:
                                    num_patients = results[0]['NumberOfPatients']
                                    disease = disease if 'disease' in locals() else 'the specified disease'
                                    result_summary = f"There are {num_patients} patients diagnosed with {disease}."
                                    st.session_state.chat_history.append({"role": "assistant", "content": result_summary})
                                    logger.info(f"Result Summary: {result_summary}")
                                else:
                                    # For other types of results
                                    df = pd.DataFrame(results)
                                    # Display results as a table
                                    st.table(df)
                                    logger.info("Displayed query results as a table.")

                                    # Provide download option
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Results as CSV",
                                        data=csv,
                                        file_name='query_results.csv',
                                        mime='text/csv',
                                    )
                                    logger.info("Provided CSV download option.")

                                    # Generate a summary of the results
                                    try:
                                        result_summary = openai_chat.summarize_results(results)
                                        st.session_state.chat_history.append({"role": "assistant", "content": result_summary})
                                        logger.info(f"Result Summary: {result_summary}")
                                    except Exception as e:
                                        st.error("Failed to generate a summary of the results.")
                                        logger.error(f"Summary Generation Error: {e}")
                            else:
                                st.warning("No results found for your query.")
                                st.session_state.chat_history.append({"role": "assistant", "content": "No results found for your query."})
                                logger.info("No results found for the query.")
                    else:
                        st.error("Neo4j helper is not available. Please reconnect to the database.")
                        logger.error("Neo4j helper is not available in session state.")
                else:
                    # Cypher query is None, error already handled
                    pass
            else:
                st.warning("Please enter a query.")

    # Display chat history
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Chat History")
    with st.container():
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
                message_html = f'''
                <div class="chat-container">
                    <div class="chat-message assistant">
                        <div class="message-avatar">AI</div>
                        <div class="message-content">{chat["content"]}</div>
                    </div>
                </div>
                '''
                st.markdown(message_html, unsafe_allow_html=True)
else:
    # Display prompt to connect to Neo4j
    st.markdown("""
    <p style='text-align: center; font-size: 1.1em; color: #34495E;'>
    Please connect to the Neo4j database using the settings in the sidebar to start using the assistant.
    </p>
    """, unsafe_allow_html=True)