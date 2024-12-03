# utils/neo4j_helper.py

from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jHelper:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the Neo4j driver.

        Args:
            uri (str): Neo4j URI.
            user (str): Username.
            password (str): Password.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j driver initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            raise

    def is_connected(self) -> bool:
        """
        Check if the connection to Neo4j is successful.

        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j.")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def run_query(self, query: str, parameters: dict = None) -> list:
        """
        Run a Cypher query and return the results.

        Args:
            query (str): Cypher query.
            parameters (dict, optional): Parameters for the query.

        Returns:
            list: List of result records.
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                logger.info(f"Executed query: {query}")
                return records
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
