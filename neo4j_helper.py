# neo4j_helper.py

from neo4j import GraphDatabase
import logging

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
        return self.run_query(cypher_query)

    def get_node_names(self):
        """
        Retrieves all unique node labels and their associated node_names properties.
        Returns:
            dict: A dictionary where the keys are node labels and the values are lists of node_names.
        """
        cypher_query = """
        MATCH (n)
        WITH DISTINCT labels(n) AS node_labels
        UNWIND node_labels AS label
        RETURN DISTINCT label, COLLECT(DISTINCT n.node_names) AS node_names
        """
        result = self.run_query(cypher_query)
        if result:
            # Convert the result to a dictionary of {label: [node_names]}
            node_names_dict = {row['label']: row['node_names'] for row in result}
            logger.debug(f"Node names fetched: {node_names_dict}")
            return node_names_dict
        else:
            logger.error("Failed to retrieve node names.")
            return {}
