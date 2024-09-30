# neo4j_helper.py

from neo4j import GraphDatabase
import logging
import json

logger = logging.getLogger(__name__)

class Neo4jHelper:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e

    def close(self):
        self.driver.close()

    def run_query(self, cypher_query):
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = result.data()
                logger.debug(f"Query Results: {records}")
                return records
        except Exception as e:
            logger.error(f"Failed to run query: {e}")
            return None

    def get_schema(self):
        """
        Retrieves the database schema using the CALL db.schema.visualization() command.

        Returns:
            dict: The schema visualization as a dictionary.
        """
        cypher_query = "CALL db.schema.visualization()"
        schema = self.run_query(cypher_query)
        if schema:
            # Process schema if necessary
            return schema
        else:
            logger.error("Failed to retrieve database schema.")
            return {}
