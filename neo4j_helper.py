# neo4j_helper.py

from neo4j import GraphDatabase, exceptions
import logging

logger = logging.getLogger(__name__)

class Neo4jHelper:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j.")
        except exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service is unavailable: {e}")
            raise e
        except exceptions.AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, cypher_query):
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                logger.debug(f"Query returned {len(records)} records.")
                return records
        except exceptions.CypherSyntaxError as e:
            logger.error(f"Cypher syntax error: {e}")
            return None
        except exceptions.ClientError as e:
            logger.error(f"Client error: {e}")
            return None
        except exceptions.TransientError as e:
            logger.error(f"Transient error: {e}")
            return None
        except exceptions.DatabaseError as e:
            logger.error(f"Database error: {e}")
            return None
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

    def get_node_labels(self):
        """
        Retrieves all unique node labels in the database.
        Returns:
            list: A list of node labels.
        """
        cypher_query = "MATCH (n) RETURN DISTINCT labels(n) AS labels"
        result = self.run_query(cypher_query)
        if result:
            labels = [label for row in result for label in row['labels']]
            logger.debug(f"Node labels fetched: {labels}")
            return labels
        else:
            logger.error("Failed to retrieve node labels.")
            return []
