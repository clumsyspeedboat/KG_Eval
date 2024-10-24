# utils/neo4j_helper.py

from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jHelper:
    def __init__(self, uri, user, password):
        """
        Initialize the Neo4jHelper class with connection details.

        Args:
            uri (str): The URI of the Neo4j database.
            user (str): The username for authentication.
            password (str): The password for authentication.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e
        
    def is_connected(self):
        """Check if the Neo4j connection is established."""
        return self.is_connected

    def close(self):
        """
        Close the connection to the Neo4j database.
        """
        if self.driver:
            self.driver.close()

    def run_query(self, cypher_query):
        """
        Execute a Cypher query against the Neo4j database.

        Args:
            cypher_query (str): The Cypher query to execute.

        Returns:
            list or None: The results of the query or None if an error occurs.
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                logger.debug(f"Query returned {len(records)} records.")
                return records
        except Exception as e:
            logger.error(f"Failed to run query: {e}")
            return None
