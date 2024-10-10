# proteomic_ai.py

import streamlit as st
import configparser
import logging
import pandas as pd
import json

from openai_api import chat_gpt
from neo4j_helper import Neo4jHelper

from pathlib import Path

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
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract specific fields; adjust keys based on your JSON structure
                title = data.get("title", "No Title")
                content = data.get("content", "")
                combined_context += f"### {title}\n{content}\n\n"
                print(f"Loaded and processed: {file_path.name}")
        except json.JSONDecodeError as jde:
            print(f"JSON decode error in {file_path.name}: {jde}")
        except Exception as e:
            print(f"Failed to read {file_path.name}: {e}")

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
    OPENAI_API_KEY = config.get('api_keys', 'OPENAI_API_KEY')
    OPENAI_API_BASE = config.get('api_keys', 'OPENAI_API_BASE', fallback="https://api.openai.com/v1")
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
    return Neo4jHelper(uri, user, password)

neo4j_helper = get_neo4j_helper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 6. Sidebar settings
st.sidebar.header("Settings")
temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.7, 0.05)

# 7. Title and input
st.title("🧬 Proteomic AI Chat Assistant")
st.markdown("""
Welcome to the **Proteomic AI Chat Assistant**! Ask any question about proteomics, and I'll translate it into a Cypher query and fetch the results from our database.
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
    if 'chat_history' not in st.session_state:
        context = load_json_context_files("/context_files/")
        system_prompt = (
            "You are an AI assistant connected to a Neo4j database. Your job is to translate natural language queries "
            "into Cypher queries for the database. Always assume that you are connected to the database. "
            "Generate Cypher queries for Neo4j database containing information about proteomics, diseases, drugs, and other biomedical data.\n\n"
            
            "Primarily you should answer from the following context and the results of the query."
             f"### Provided Context:\n{context}"
            
            "Present the query within a code block using the ```cypher syntax."
            "Run the queries too on your connected database and display the result."
            "You can answer from external context too."
        )
        st.session_state['chat_history'] = [{"role": "system", "content": system_prompt}]

    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "system":
                continue
            elif chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            elif chat["role"] == "assistant":
                st.markdown(f"**Assistant:** {chat['content']}")

# 8. Process the latest user message
if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-1]["role"] == "user":
    user_message = st.session_state.chat_history[-1]["content"]

    # Translate to Cypher using chat_gpt with adjustable temperature
    with st.spinner("Translating your query..."):
        cypher_query = chat_gpt(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            model="gpt-4",
            prompt=user_message,
            temperature=temperature  
        )

    if cypher_query:
        st.session_state.chat_history.append({"role": "assistant", "content": f"Translated Cypher Query:\n```cypher\n{cypher_query}\n```"})

        # Run Cypher query
        with st.spinner("Running the Cypher query..."):
            results = neo4j_helper.run_query(cypher_query)

        if results is not None and len(results) > 0:
            df = pd.DataFrame(results)
            if "count(d)" in df.columns:
                count_value = df["count(d)"].values[0]
                st.session_state.chat_history.append({"role": "assistant", "content": f"There are {count_value} drugs in the database."})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "Query Results:"})
                st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='query_results.csv',
                mime='text/csv',
            )
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "No results found or failed to run the Cypher query."})

        st.rerun()
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "Failed to translate the query into a valid Cypher command."})
        st.rerun()

# 9. Reset conversation button in the sidebar
if st.sidebar.button("🧹 Reset Conversation"):
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt}
    ]
    st.rerun()
