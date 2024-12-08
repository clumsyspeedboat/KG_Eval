# test.py

import configparser
import logging
import json
from pathlib import Path
from neo4j import GraphDatabase, basic_auth
import time
import psutil
import os
from collections import defaultdict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

def load_config(config_path: str):
    config = configparser.ConfigParser()
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    config.read(config_path)
    return config

class Neo4jHelper:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

def extract_property_keys(neo4j_helper, logger):
    property_keys_set = set()
    try:
        query = """
        MATCH (n)
        UNWIND keys(n) AS prop
        RETURN DISTINCT prop
        UNION
        MATCH ()-[r]->()
        UNWIND keys(r) AS prop
        RETURN DISTINCT prop
        """
        start_time = time.time()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()
        memory_before = process.memory_info().rss

        results = neo4j_helper.run_query(query)
        for record in results:
            property_keys_set.add(record['prop'])

        end_time = time.time()
        cpu_times_after = process.cpu_times()
        memory_after = process.memory_info().rss
        logger.info(f"Extracted {len(property_keys_set)} unique property keys.")
        logger.info(f"Time taken to extract property keys: {end_time - start_time} seconds")
        logger.info(f"CPU time used: {cpu_times_after.user - cpu_times_before.user} seconds")
        logger.info(f"Memory used: {(memory_after - memory_before) / (1024 * 1024)} MB")
    except Exception as e:
        logger.error(f"Error extracting property keys: {e}")
    return list(property_keys_set)

def extract_nodes(neo4j_helper, property_key_to_index, logger):
    nodes = {}
    batch_size = 100000  # Adjust as needed
    last_id = -1  # Start with -1 to include all IDs
    total_nodes = 0
    try:
        start_time = time.time()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()
        memory_before = process.memory_info().rss

        while True:
            query = """
            MATCH (n)
            WHERE ID(n) > $last_id
            RETURN ID(n) AS node_id, labels(n) AS labels, keys(n) AS props
            ORDER BY ID(n)
            LIMIT $batch_size
            """
            parameters = {'last_id': last_id, 'batch_size': batch_size}
            results = neo4j_helper.run_query(query, parameters)
            if not results:
                break
            for record in results:
                node_id = record['node_id']
                labels = record['labels']
                props = record['props']
                prop_indices = [property_key_to_index[p] for p in props if p in property_key_to_index]
                nodes[node_id] = {
                    'labels': labels,
                    'property_indices': prop_indices
                }
                last_id = node_id
                total_nodes += 1
                if total_nodes % 10000 == 0:
                    logger.info(f"Processed {total_nodes} nodes.")

        end_time = time.time()
        cpu_times_after = process.cpu_times()
        memory_after = process.memory_info().rss
        logger.info(f"Extracted {total_nodes} nodes.")
        logger.info(f"Time taken to extract nodes: {end_time - start_time} seconds")
        logger.info(f"CPU time used: {cpu_times_after.user - cpu_times_before.user} seconds")
        logger.info(f"Memory used: {(memory_after - memory_before) / (1024 * 1024)} MB")
    except Exception as e:
        logger.error(f"Error extracting nodes: {e}")
    return nodes

def extract_relationships(neo4j_helper, property_key_to_index, logger):
    relationships = []
    batch_size = 100000  # Adjust as needed
    last_rel_id = -1
    total_rels = 0
    try:
        start_time = time.time()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()
        memory_before = process.memory_info().rss

        while True:
            query = """
            MATCH ()-[r]->()
            WHERE ID(r) > $last_rel_id
            RETURN ID(r) AS rel_id, ID(startNode(r)) AS start_id, ID(endNode(r)) AS end_id, type(r) AS type, keys(r) AS props
            ORDER BY ID(r)
            LIMIT $batch_size
            """
            parameters = {'last_rel_id': last_rel_id, 'batch_size': batch_size}
            results = neo4j_helper.run_query(query, parameters)
            if not results:
                break
            for record in results:
                rel_id = record['rel_id']
                start_id = record['start_id']
                end_id = record['end_id']
                rel_type = record['type']
                props = record['props']
                prop_indices = [property_key_to_index[p] for p in props if p in property_key_to_index]
                relationships.append({
                    'start_id': start_id,
                    'end_id': end_id,
                    'type': rel_type,
                    'property_indices': prop_indices
                })
                last_rel_id = rel_id
                total_rels += 1
                if total_rels % 10000 == 0:
                    logger.info(f"Processed {total_rels} relationships.")

        end_time = time.time()
        cpu_times_after = process.cpu_times()
        memory_after = process.memory_info().rss
        logger.info(f"Extracted {total_rels} relationships.")
        logger.info(f"Time taken to extract relationships: {end_time - start_time} seconds")
        logger.info(f"CPU time used: {cpu_times_after.user - cpu_times_before.user} seconds")
        logger.info(f"Memory used: {(memory_after - memory_before) / (1024 * 1024)} MB")
    except Exception as e:
        logger.error(f"Error extracting relationships: {e}")
    return relationships

def build_compact_ontology(property_keys, nodes, relationships, logger):
    try:
        start_time = time.time()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()
        memory_before = process.memory_info().rss

        property_list = property_keys

        # Build node_dict
        node_dict = {}
        node_properties_set = {}
        for node_id, data in nodes.items():
            prop_indices = data['property_indices']
            prop_tuple = tuple(sorted(prop_indices))
            if prop_tuple in node_properties_set:
                node_dict[str(node_id)] = node_properties_set[prop_tuple]
            else:
                node_dict[str(node_id)] = prop_indices
                node_properties_set[prop_tuple] = prop_indices

        # Build edge_dict
        edge_dict = {}
        for rel in relationships:
            rel_type = rel['type']
            prop_indices = rel['property_indices']
            if rel_type not in edge_dict:
                edge_dict[rel_type] = prop_indices

        # Build knowledge_matrix with edge counts using defaultdict
        knowledge_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for rel in relationships:
            start_id = str(rel['start_id'])
            end_id = str(rel['end_id'])
            rel_type = rel['type']
            knowledge_matrix[start_id][end_id][rel_type] += 1

        # Format knowledge_matrix
        formatted_knowledge_matrix = {
            start_id: {
                end_id: [[count, edge_type] for edge_type, count in edge_types.items()]
                for end_id, edge_types in end_dict.items()
            }
            for start_id, end_dict in knowledge_matrix.items()
        }

        # Convert node IDs to labels
        node_id_to_label = {str(node_id): data['labels'] for node_id, data in nodes.items()}

        compact_ontology = {
            'property_list': property_list,
            'node_dict': node_dict,
            'edge_dict': edge_dict,
            'knowledge_matrix': formatted_knowledge_matrix,
            'node_id_to_label': node_id_to_label
        }

        end_time = time.time()
        cpu_times_after = process.cpu_times()
        memory_after = process.memory_info().rss
        logger.info("Compact ontology built successfully.")
        logger.info(f"Time taken to build compact ontology: {end_time - start_time} seconds")
        logger.info(f"CPU time used: {cpu_times_after.user - cpu_times_before.user} seconds")
        logger.info(f"Memory used: {(memory_after - memory_before) / (1024 * 1024)} MB")
        return compact_ontology
    except Exception as e:
        logger.error(f"Error building compact ontology: {e}")
        return None

def main():
    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    memory_before = process.memory_info().rss
    start_time = time.time()

    logger = setup_logging()

    # Load configuration
    try:
        config = load_config('config.ini')
        NEO4J_URI = config.get('neo4j', 'NEO4J_URI')
        NEO4J_USER = config.get('neo4j', 'NEO4J_USER')
        NEO4J_PASSWORD = config.get('neo4j', 'NEO4J_PASSWORD')
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize Neo4jHelper
    try:
        neo4j_helper = Neo4jHelper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j_helper.run_query("RETURN 1 AS num")
        logger.info("Successfully connected to Neo4j database.")
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        return

    # Extract all unique property keys
    property_keys = extract_property_keys(neo4j_helper, logger)
    property_key_to_index = {k: i for i, k in enumerate(property_keys)}

    # Extract nodes
    nodes = extract_nodes(neo4j_helper, property_key_to_index, logger)

    # Extract relationships
    relationships = extract_relationships(neo4j_helper, property_key_to_index, logger)

    # Build ontology
    ontology = {
        'property_keys': property_keys,
        'nodes': nodes,
        'relationships': relationships
    }

    # Save ontology to configs/ontology.json
    try:
        ontology_path = Path('configs/ontology.json')
        ontology_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ontology_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, separators=(',', ':'), indent=None)
        ontology_size = os.path.getsize(ontology_path) / (1024 * 1024)
        logger.info(f"Ontology successfully saved to '{ontology_path.resolve()}'.")
        logger.info(f"Ontology file size: {ontology_size} MB")
    except Exception as e:
        logger.error(f"Error saving ontology to '{ontology_path}': {e}")

    # Build compact ontology
    compact_ontology = build_compact_ontology(property_keys, nodes, relationships, logger)

    # Save compact ontology to configs/compact_ontology.json
    try:
        compact_ontology_path = Path('configs/compact_ontology.json')
        with open(compact_ontology_path, 'w', encoding='utf-8') as f:
            json.dump(compact_ontology, f, separators=(',', ':'), indent=None)
        compact_ontology_size = os.path.getsize(compact_ontology_path) / (1024 * 1024)
        logger.info(f"Compact ontology successfully saved to '{compact_ontology_path.resolve()}'.")
        logger.info(f"Compact ontology file size: {compact_ontology_size} MB")
    except Exception as e:
        logger.error(f"Error saving compact ontology to '{compact_ontology_path}': {e}")

    # Close Neo4j connection
    neo4j_helper.close()
    logger.info("Neo4j connection closed.")

    end_time = time.time()
    cpu_times_after = process.cpu_times()
    memory_after = process.memory_info().rss
    execution_time = end_time - start_time
    cpu_time = cpu_times_after.user - cpu_times_before.user
    memory_used = (memory_after - memory_before) / (1024 * 1024)

    logger.info(f"Total execution time: {execution_time} seconds")
    logger.info(f"Total CPU time used: {cpu_time} seconds")
    logger.info(f"Total memory used: {memory_used} MB")

if __name__ == "__main__":
    main()
