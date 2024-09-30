# test_schema.py

from neo4j_helper import Neo4jHelper
import configparser
import json

def test_get_schema():
    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    try:
        NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
        NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
        NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Configuration Error: {e}")
        return

    # Initialize Neo4jHelper
    neo4j_helper = Neo4jHelper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Fetch schema
    schema = neo4j_helper.get_schema()
    if schema:
        print("Database Schema:")
        print(json.dumps(schema, indent=2))
    else:
        print("Failed to retrieve the database schema.")

    # Close the connection
    neo4j_helper.close()

if __name__ == "__main__":
    test_get_schema()
