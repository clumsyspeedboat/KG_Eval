# proteomic_ai.py

import streamlit as st
import configparser
import logging
import pandas as pd
import json
import re

from pathlib import Path

# Import custom modules
from openai_api import OpenAIChat
from neo4j_helper import Neo4jHelper

def load_json_context_files(folder_path):
    """
    Load and combine contents from all JSON files in the specified folder.

    Args:
        folder_path (str or Path): Path to the folder containing JSON files.

    Returns:
        str: Combined content extracted from all JSON files.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"The folder {folder_path} does not exist or is not a directory.")

    combined_context = ""
    for file_path in folder.glob("*.json"):
        try:
            # Use 'utf-8-sig' encoding to handle BOM
            with file_path.open('r', encoding='utf-8-sig') as f:
                data = json.load(f)
                # Extract specific fields; adjust keys based on your JSON structure
                title = data.get("title", "No Title")
                content = data.get("content", "")
                combined_context += f"### {title}\n{content}\n\n"
                logger.debug(f"Loaded and processed: {file_path.name}")
        except json.JSONDecodeError as jde:
            logger.error(f"JSON decode error in {file_path.name}: {jde}")
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}")

    return combined_context

# 1. Set Streamlit page configuration
st.set_page_config(
    page_title="🧬 Proteomic AI Chat Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 3. Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# 4. Access OpenAI API key and Neo4j credentials
try:
    OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
    OPENAI_API_BASE = config.get('openai', 'OPENAI_API_BASE', fallback="https://api.openai.com/v1")
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("OpenAI API key is not configured properly.")
    st.stop()

try:
    NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
    NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
    NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Neo4j configuration error: {e}")
    st.error("Neo4j credentials are not configured properly.")
    st.stop()

# 5. Initialize Neo4j helper with caching
@st.cache_resource
def get_neo4j_helper(uri, user, password):
    try:
        helper = Neo4jHelper(uri, user, password)
        return helper
    except Exception as e:
        logger.error(f"Failed to initialize Neo4jHelper: {e}")
        st.error("Failed to connect to Neo4j database.")
        st.stop()

neo4j_helper = get_neo4j_helper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 6. Initialize OpenAI Chat helper
openai_chat = OpenAIChat(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, model="gpt-4")

# 7. Load context for the system prompt
context = load_json_context_files("./context_files/")
system_prompt = (
    "You are an AI assistant connected to a Neo4j database containing information about proteomics, diseases, drugs, and other biomedical data. "
    "Your job is to translate natural language queries into Cypher queries for the database, execute them, and provide answers based on the query results and the provided context.\n\n"
    "When presenting Cypher queries, use code blocks with ```cypher syntax.\n"
    "Always ensure that your answers are derived from the query results and the context.\n"
    f"### Provided Context:\n{context}"
)

# 8. Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [{"role": "system", "content": system_prompt}]

# 9. Sidebar settings
st.sidebar.header("Settings")
temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.7, 0.05)

# 10. Title and input
st.title("🧬 Proteomic AI Chat Assistant")
st.markdown("""
Welcome to the **Proteomic AI Chat Assistant**! Ask any question about proteomics, and I'll translate it into a Cypher query, fetch the results from our database, and provide you with an informative answer.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Enter your query")
    user_query = st.text_input("Enter your query here:", key="user_query")

    # Submit button logic
    if st.button("Submit", key="submit_button"):
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.rerun()
        else:
            st.warning("Please enter a query.")

with col2:
    st.markdown("### Chat History")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "system":
                continue
            elif chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            elif chat["role"] == "assistant":
                st.markdown(f"**Assistant:** {chat['content']}")

# 11. Process the latest user message
if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-1]["role"] == "user":
    user_message = st.session_state.chat_history[-1]["content"]
    logger.debug(f"Processing user message: {user_message}")

    # Get the conversation history for context
    conversation = st.session_state.chat_history

    # Translate to Cypher and get the answer using the OpenAIChat class
    with st.spinner("Generating response..."):
        try:
            assistant_response = openai_chat.generate_response(conversation, temperature=temperature)
        except Exception as e:
            st.error(f"An error occurred while communicating with OpenAI: {e}")
            assistant_response = None

    if assistant_response:
        # Append assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Extract Cypher query from the assistant's response
        cypher_match = re.search(r'```cypher\n(.*?)\n```', assistant_response, re.DOTALL)
        if cypher_match:
            cypher_query = cypher_match.group(1)
            logger.debug(f"Extracted Cypher Query: {cypher_query}")

            # Run Cypher query
            with st.spinner("Running the Cypher query..."):
                results = neo4j_helper.run_query(cypher_query)
                if results is None:
                    st.error("Failed to execute the Cypher query.")
                    logger.error("Failed to execute the Cypher query.")
                else:
                    logger.debug(f"Query Results: {results}")

                    if len(results) > 0:
                        # Convert results to DataFrame
                        df = pd.DataFrame(results)

                        # Generate a summary or detailed answer based on the results
                        with st.spinner("Generating answer based on query results..."):
                            try:
                                result_summary = openai_chat.generate_result_summary(conversation, results, temperature=temperature)
                            except Exception as e:
                                st.error(f"An error occurred while summarizing the results: {e}")
                                result_summary = None

                            if result_summary:
                                st.session_state.chat_history.append({"role": "assistant", "content": result_summary})
                            else:
                                st.error("Failed to generate a summary of the query results.")

                        # Display DataFrame
                        st.dataframe(df)

                        # Provide download option
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name='query_results.csv',
                            mime='text/csv',
                        )
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": "No results found for your query."})
                        logger.warning("No results returned from the Cypher query.")
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "Failed to extract Cypher query from the response."})
            logger.error("Failed to extract Cypher query from the assistant's response.")

        st.rerun()
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "Failed to generate a response."})
        logger.error("Failed to generate a response from the assistant.")
        st.rerun()

# 12. Reset conversation button in the sidebar
if st.sidebar.button("🧹 Reset Conversation"):
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
    logger.info("Conversation has been reset.")
    st.rerun()
