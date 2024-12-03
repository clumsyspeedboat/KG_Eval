# app.py

import streamlit as st
import configparser
import logging
import pandas as pd
import re
import os
import base64
import json
import time

from pathlib import Path

# Import custom modules
from utils.neo4j_helper import Neo4jHelper
from utils.context_loader import load_context
from utils.query_matcher import find_best_query_match_enhanced, find_best_query_match_rapidfuzz, find_best_query_match_with_tags
from utils.prompt_loader import load_agent_prompts  # Importing the prompt loader

# Import agents
from agents.ontology_protex_agent import OntologyProtexAgent
from agents.inference_protex_agent import InferenceProtexAgent

# -----------------------------
# Function Definitions
# -----------------------------

def load_css(css_file_path):
    """
    Load CSS from a file.
    """
    try:
        with open(css_file_path) as f:
            css = f.read()
        return css
    except FileNotFoundError:
        st.warning(f"CSS file not found at {css_file_path}")
        return ""

def load_html(html_file_path):
    """
    Load HTML from a file.
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return html
    except FileNotFoundError:
        st.warning(f"HTML file not found at {html_file_path}")
        return ""

def encode_image_to_base64(image_path):
    """
    Encode image to base64.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded
    except FileNotFoundError:
        st.warning(f"Image file not found at {image_path}")
        return ""

def embed_logo_in_html(html_content, logo_base64):
    """
    Embed logo into HTML content.
    """
    return html_content.replace("{{logo}}", f"data:image/png;base64,{logo_base64}")

def extract_cypher_query(response):
    """
    Extract the Cypher query from the assistant's response.
    """
    match = re.search(r'```cypher\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if match:
        cypher_query = match.group(1).strip()
        return cypher_query
    else:
        logger.warning("No Cypher query found in the assistant's response.")
        return ""

def extract_disease_from_query(user_query):
    """
    Extract the disease name from the user's natural language query.
    """
    # Simple regex to find disease names (assuming they follow "disease" keyword)
    match = re.search(r'disease\s+([A-Za-z\s]+)', user_query, re.IGNORECASE)
    if match:
        disease = match.group(1).strip()
        logger.info(f"Extracted disease: {disease}")
        return disease
    else:
        logger.warning("No disease name found in the query.")
        return ""

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

# Access OpenAI API key and Neo4j credentials
try:
    OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
    NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
    NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
    NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("Configuration error in 'config.ini'. Please check the file.")
    st.stop()

# -----------------------------
# Initialize session state variables
# -----------------------------
if 'neo4j_connected' not in st.session_state:
    st.session_state['neo4j_connected'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'ontology_protex_agent' not in st.session_state:
    st.session_state['ontology_protex_agent'] = None

if 'inference_protex_agent' not in st.session_state:
    st.session_state['inference_protex_agent'] = None

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
    NEO4J_URI_DYNAMIC = f"bolt://{neo4j_host}:{int(neo4j_port)}"
    NEO4J_USER_DYNAMIC = st.text_input("Neo4j User", value="neo4j")
    NEO4J_PASSWORD_DYNAMIC = st.text_input("Neo4j Password", type="password")

    # Connect button
    if st.button("Connect to Neo4j"):
        if NEO4J_PASSWORD_DYNAMIC:
            try:
                # Attempt to connect to Neo4j
                neo4j_helper = Neo4jHelper(NEO4J_URI_DYNAMIC, NEO4J_USER_DYNAMIC, NEO4J_PASSWORD_DYNAMIC)
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
# Initialize OntologyProtexAgent
# -----------------------------
def initialize_ontology_agent(prompts: dict):
    try:
        neo4j_helper = st.session_state.get('neo4j_helper')
        if not neo4j_helper:
            logger.error("Neo4jHelper is not available in session state.")
            st.error("Neo4jHelper is not available. Please connect to Neo4j first.")
            return None

        openai_api_key = OPENAI_API_KEY  # Ensure this is loaded from config

        resources_path = "./resources/"  # Adjust if your resources are in a different path

        ontology_agent = OntologyProtexAgent(
            neo4j_helper=neo4j_helper,
            openai_api_key=openai_api_key,
            resources_path=resources_path,
            prompts=prompts
        )

        # Extract and form the ontology
        schema = ontology_agent.extract_schema()
        additional_resources = ontology_agent.load_additional_resources()
        ontology = ontology_agent.form_ontology(
            schema=schema,
            additional_resources=additional_resources
        )

        # Save the ontology
        ontology_agent.save_ontology("configs/ontology.json")

        # Store the agent in session state
        st.session_state['ontology_protex_agent'] = ontology_agent
        logger.info("OntologyProtexAgent initialized successfully.")
        return ontology_agent

    except Exception as e:
        logger.error(f"Failed to initialize OntologyProtexAgent: {e}")
        st.error("Failed to initialize OntologyProtexAgent.")
        return None

# -----------------------------
# Initialize InferenceProtexAgent
# -----------------------------
def initialize_inference_agent(openai_api_key: str, ontology_summary_path: str):
    try:
        inference_agent = InferenceProtexAgent(
            openai_api_key=openai_api_key,
            ontology_summary_path=ontology_summary_path
        )
        st.session_state['inference_protex_agent'] = inference_agent
        logger.info("InferenceProtexAgent initialized successfully.")
        return inference_agent
    except Exception as e:
        logger.error(f"Failed to initialize InferenceProtexAgent: {e}")
        st.error("Failed to initialize InferenceProtexAgent.")
        return None

# -----------------------------
# Load Agent Prompts
# -----------------------------
def load_prompts():
    prompts_path = Path('configs/agent_prompts.json')
    try:
        prompts = load_agent_prompts(str(prompts_path))
        return prompts
    except Exception as e:
        st.error("Failed to load agent prompts.")
        logger.error(f"Failed to load agent prompts: {e}")
        st.stop()

# -----------------------------
# Main Application Logic
# -----------------------------
# Check if connected to Neo4j before proceeding
if st.session_state['neo4j_connected']:
    # Load agent prompts
    prompts = load_prompts()

    # Initialize OntologyProtexAgent if not already initialized
    if not st.session_state['ontology_protex_agent']:
        ontology_agent = initialize_ontology_agent(prompts)
        if not ontology_agent:
            st.stop()
    else:
        ontology_agent = st.session_state['ontology_protex_agent']
        logger.info("OntologyProtexAgent retrieved from session state.")

    # Initialize InferenceProtexAgent if not already initialized
    if not st.session_state.get('inference_protex_agent'):
        ontology_summary_path = "configs/ontology_summary.json"
        inference_agent = initialize_inference_agent(OPENAI_API_KEY, ontology_summary_path)
        if not inference_agent:
            st.stop()
    else:
        inference_agent = st.session_state['inference_protex_agent']
        logger.info("InferenceProtexAgent retrieved from session state.")

    # Load context and predefined queries with caching
    context_folder = "./resources/context_files/"
    queries_folder = "./resources/queries/"
    context = load_context(context_folder)
    predefined_queries = load_context(queries_folder, as_dict=True)

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

                # Find best query match using enhanced matching
                try:
                    # Choose the matching function based on your preference
                    # Option 1: Enhanced similarity-based matching
                    best_match = find_best_query_match_enhanced(user_query, predefined_queries)
                    
                    # Option 2: RapidFuzz matching (ensure you have RapidFuzz installed and configured)
                    # best_match = find_best_query_match_rapidfuzz(user_query, predefined_queries, threshold=40)
                    
                    # Option 3: Tag-based matching
                    # best_match = find_best_query_match_with_tags(user_query, predefined_queries)
                    
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
                    # Use InferenceProtexAgent to generate Cypher query
                    logger.info("No good match found. Using InferenceProtexAgent to generate Cypher query.")
                    cypher_query = inference_agent.process_user_query(user_query)
                    if inference_agent.validate_cypher_query(cypher_query):
                        st.session_state.chat_history.append({"role": "assistant", "content": f"```cypher\n{cypher_query}\n```"})
                        logger.info(f"Generated Cypher Query: {cypher_query}")
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": cypher_query})
                        cypher_query = None

                # Run the Cypher query if available
                if cypher_query and inference_agent.validate_cypher_query(cypher_query):
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
                                        summary = inference_agent.generate_summary(
                                            "Summarize the following query results in detail, highlighting key insights and trends.",
                                            json.dumps(results, indent=2)
                                        )
                                        st.session_state.chat_history.append({"role": "assistant", "content": summary})
                                        logger.info(f"Result Summary: {summary}")
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
                    # Cypher query is None or invalid, error already handled
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
