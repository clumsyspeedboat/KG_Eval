# proteomic_ai.py

import streamlit as st  # Import Streamlit first
import configparser
import logging
import pandas as pd
import json

from openai_api import chat_gpt
from neo4j_helper import Neo4jHelper

# 1. Set Streamlit page configuration
st.set_page_config(
    page_title="🧬 Proteomic AI Chat Assistant",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. Configure Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 3. Load configuration from config.ini or use Streamlit secrets
config = configparser.ConfigParser()
config.read('config.ini')  # Ensure the correct path to your config.ini

# 4. Access OpenAI API key
try:
    OPENAI_API_KEY = config.get('api_keys', 'OPENAI_API_KEY')
    OPENAI_API_BASE = config.get('api_keys', 'OPENAI_API_BASE', fallback="https://api.openai.com/v1")
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Configuration Error: {e}")
    st.error("OpenAI API key is not configured properly.")
    st.stop()

# 5. Access Neo4j credentials
try:
    NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
    NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
    NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logger.error(f"Neo4j configuration error: {e}")
    st.error("Neo4j credentials are not configured properly.")
    st.stop()

# 6. Initialize Neo4j helper with caching to persist connections across reruns
@st.cache_resource
def get_neo4j_helper(uri, user, password):
    return Neo4jHelper(uri, user, password)

neo4j_helper = get_neo4j_helper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 7. Retrieve and store the database schema
@st.cache_data(show_spinner=False)
def fetch_and_store_schema(_helper: Neo4jHelper):
    schema = _helper.get_schema()
    print("Fetched schema:", schema)  # Debugging line
    if schema:
        # Convert schema to JSON for easy inclusion in prompts
        schema_json = json.dumps(schema, indent=2)
        return schema_json
    else:
        return "Failed to retrieve the database schema."

schema_context = fetch_and_store_schema(neo4j_helper)

# 8. Streamlit App Content
st.title("🧬 Proteomic AI Chat Assistant")
st.markdown("""
Welcome to the **Proteomic AI Chat Assistant**! Ask any question about proteomics, and I'll translate it into a Cypher query and fetch the results from our database.
""")

# 9. Initialize session state for chat history
if 'chat_history' not in st.session_state:
    # Include schema context in the system prompt
    system_prompt = (
        "You are an AI assistant specialized in translating natural language queries into Cypher queries for a Neo4j database. "
        "Use the provided schema to generate accurate Cypher queries based on user requests.\n\n"
        "For each user request, generate only the Cypher query without any explanations or additional text. "
        "Present the query within a code block using the ```cypher syntax."
    )
    st.session_state['chat_history'] = [
        {"role": "system", "content": system_prompt}
    ]

# 10. Display chat history in the main page within a scrollable container
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "system":
            continue  # Skip system messages
        elif chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        elif chat["role"] == "assistant":
            st.markdown(f"**Assistant:** {chat['content']}")

# 11. Input area for user query
user_query = st.text_input("Enter your query here:", key="user_query")

# 12. Submit button logic
if st.button("Submit", key="submit_button"):
    if user_query:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.rerun()  # Replace st.experimental_rerun with st.rerun
    else:
        st.warning("Please enter a query.")

# 13. Process the latest user message
if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-1]["role"] == "user":
    user_message = st.session_state.chat_history[-1]["content"]
    print("User message:", user_message)  # Debugging line

    # Translate to Cypher using the chat_gpt function
    with st.spinner("Translating your query..."):
        cypher_query = chat_gpt(
            api_key=OPENAI_API_KEY, 
            base_url=OPENAI_API_BASE, 
            model="gpt-4",  # Specify the model
            prompt=user_message
        )
        print("Translated Cypher Query:", cypher_query)  # Debugging line

    if cypher_query and cypher_query.lower().startswith(("match", "create", "return", "delete", "set", "merge", "with", "call", "unwind")):
        # Append assistant's translated Cypher query to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": f"Translated Cypher Query:\n```cypher\n{cypher_query}\n```"})

        # Run Cypher query
        with st.spinner("Running the Cypher query..."):
            results = neo4j_helper.run_query(cypher_query)
            print("Query Results:", results)  # Debugging line

        if results is not None and len(results) > 0:
            # Format results into a DataFrame for better readability
            df = pd.DataFrame(results)
            st.session_state.chat_history.append({"role": "assistant", "content": f"Query Results:\n{df.to_html(index=False, escape=False)}"})
            st.markdown("**Assistant:** Query Results:")
            st.write(df)

            # Provide a download button for CSV
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

# 14. Display chat history in sidebar
st.sidebar.header("🗨️ Chat History")
for chat in st.session_state.chat_history:
    if chat["role"] == "system":
        continue  # Skip system messages
    elif chat["role"] == "user":
        st.sidebar.markdown(f"**You:** {chat['content']}")
    elif chat["role"] == "assistant":
        st.sidebar.markdown(f"**Assistant:** {chat['content']}")

# 15. Add a Reset Conversation Button in Sidebar
if st.sidebar.button("🧹 Reset Conversation"):
    # Re-initialize the chat history with updated system prompt
    system_prompt = (
        "You are an AI assistant specialized in translating natural language queries into Cypher queries for a Neo4j database. "
        "Use the provided schema to generate accurate Cypher queries based on user requests.\n\n"
        "For each user request, generate only the Cypher query without any explanations or additional text. "
        "Present the query within a code block using the ```cypher syntax."
    )
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt}
    ]
    st.rerun()
