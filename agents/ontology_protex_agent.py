# agents/ontology_protex_agent.py

import logging
import json
import openai
from pathlib import Path
from collections import defaultdict
import time

# Import specific OpenAI exceptions
from openai import RateLimitError, APIError, APIConnectionError

logger = logging.getLogger(__name__)

class OntologyProtexAgent:
    def __init__(self, neo4j_helper, openai_api_key: str, resources_path: str, prompts: dict, openai_model: str = "gpt-4"):
        """
        Initialize the OntologyProtexAgent.

        Args:
            neo4j_helper: Helper to interact with Neo4j.
            openai_api_key (str): OpenAI API key.
            resources_path (str): Path to the resources folder.
            prompts (dict): Dictionary containing agent prompts.
            openai_model (str): OpenAI model to use.
        """
        self.neo4j_helper = neo4j_helper
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.openai_model = openai_model
        self.resources_path = Path(resources_path)
        self.ontology = {}
        self.system_prompt = prompts.get("OntologyProtexAgent", {}).get("system_prompt", "")
        self.user_prompt = prompts.get("OntologyProtexAgent", {}).get("user_prompt", "")

    def extract_schema(self):
        """
        Extract the schema (relationships and their properties, node properties) from Neo4j.
        """
        try:
            # Extract Relationships
            query_rels = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            rels_result = self.neo4j_helper.run_query(query_rels)
            relationships = set()
            for record in rels_result:
                rel_type = record['relationshipType']
                relationships.add(rel_type)
            logger.info(f"Extracted Relationships: {relationships}")

            # Extract Relationship Property Keys
            rel_props = {}
            for rel in relationships:
                query_rel_props = f"MATCH ()-[r:{rel}]-() RETURN keys(r) AS keys LIMIT 100"
                props_result = self.neo4j_helper.run_query(query_rel_props)
                props = set()
                for record in props_result:
                    keys = record['keys']
                    for key in keys:
                        props.add(key)
                rel_props[rel] = list(props)
            logger.info(f"Extracted Relationship Properties: {rel_props}")

            # Extract Node Property Keys
            query_nodes = "CALL db.labels() YIELD label RETURN label"
            nodes_result = self.neo4j_helper.run_query(query_nodes)
            node_props = defaultdict(list)
            for record in nodes_result:
                node = record['label']
                query_node_props = f"MATCH (n:{node}) RETURN keys(n) AS keys LIMIT 100"
                props_result = self.neo4j_helper.run_query(query_node_props)
                props = set()
                for prop_record in props_result:
                    keys = prop_record['keys']
                    for key in keys:
                        props.add(key)
                node_props[node].extend(list(props))
            logger.info(f"Extracted Node Properties: {dict(node_props)}")

            # Perform Hierarchical Grouping
            grouped_rels = self.group_relationships(relationships)
            grouped_node_props = self.group_properties(node_props)
            grouped_rel_props = self.group_properties(rel_props)

            return {
                'relationships': grouped_rels,
                'node_properties': grouped_node_props,
                'relationship_properties': grouped_rel_props
            }
        except Exception as e:
            logger.error(f"Error extracting schema from Neo4j: {e}")
            raise

    def group_relationships(self, relationships):
        """
        Group relationships based on common prefixes.
        """
        grouped = defaultdict(list)
        for rel in relationships:
            if '_' in rel:
                prefix, suffix = rel.split('_', 1)
                grouped[prefix].append(suffix)
            else:
                grouped['OTHER'].append(rel)
        logger.info(f"Grouped Relationships: {dict(grouped)}")
        return dict(grouped)

    def group_properties(self, properties_dict):
        """
        Group properties based on common prefixes.
        """
        grouped = defaultdict(list)
        for key, props in properties_dict.items():
            for prop in props:
                if '_' in prop:
                    prefix, suffix = prop.split('_', 1)
                    grouped[prefix].append(suffix)
                else:
                    grouped['OTHER'].append(prop)
        logger.info(f"Grouped Properties: {dict(grouped)}")
        return dict(grouped)

    def load_additional_resources(self):
        """
        Load any additional resources if necessary.
        """
        # Implement loading of additional resources if applicable
        return {}

    def form_ontology(self, schema: dict, additional_resources: dict):
        """
        Combine schema with additional resources to form the ontology.

        Args:
            schema (dict): Extracted schema with 'relationships', 'relationship_properties', and 'node_properties'.
            additional_resources (dict): Additional relationships and other resources.

        Returns:
            dict: The combined ontology.
        """
        try:
            ontology_rels = defaultdict(list, schema['relationships'])

            # Add additional relationships with grouping
            additional_rels = additional_resources.get('relationships', [])
            for rel in additional_rels:
                if '_' in rel:
                    prefix, suffix = rel.split('_', 1)
                    ontology_rels[prefix].append(suffix)
                else:
                    ontology_rels['OTHER'].append(rel)

            # Combine property keys
            node_properties = schema.get('node_properties', {})
            relationship_properties = schema.get('relationship_properties', {})

            grouped_node_props = self.group_properties(node_properties)
            grouped_rel_props = self.group_properties(relationship_properties)

            self.ontology = {
                'relationships': dict(ontology_rels),
                'node_properties': grouped_node_props,
                'relationship_properties': grouped_rel_props
            }
            logger.info(f"Formed Ontology: {self.ontology}")

            # Save the ontology before generating summary
            self.save_ontology("configs/ontology.json")

            # Summarize ontology
            self.summarize_and_save_ontology()

            return self.ontology
        except Exception as e:
            logger.error(f"Error forming ontology: {e}")
            raise

    def save_ontology(self, path: str):
        """
        Save the full ontology to a JSON file.

        Args:
            path (str): Path to save the ontology JSON.
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.ontology, f, indent=4)
            logger.info(f"Ontology saved to {path}.")
        except Exception as e:
            logger.error(f"Error saving ontology to {path}: {e}")
            raise

    def summarize_and_save_ontology(self):
        """
        Summarize the ontology and save both the full ontology and the summary as a JSON file in the configs directory.
        """
        try:
            summary_prompt = (
                "Provide an extensive and detailed summary of the following ontology data. "
                "Include comprehensive descriptions of each relationship group and elaborating on the nature of their properties. "
                "Detail the property keys associated with each relationship property group and node property group. "
                "Highlight key entities and explain how they interconnect within the system."
            )
            ontology_text = f"Relationships: {json.dumps(self.ontology['relationships'], indent=2)}\n\n" \
                           f"Relationship Properties: {json.dumps(self.ontology['relationship_properties'], indent=2)}\n\n" \
                           f"Node Properties: {json.dumps(self.ontology['node_properties'], indent=2)}"
            summary = self.generate_summary(summary_prompt, ontology_text)

            summary_data = {
                'ontology': self.ontology,
                'summary': summary
            }

            summary_path = Path('configs') / 'ontology_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4)
            logger.info(f"Ontology and summary saved to {summary_path}.")
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded while generating summary: {e}")
            # Depending on your application's needs, you might choose to save the ontology without the summary
        except APIError as e:
            logger.error(f"API error while generating summary: {e}")
        except APIConnectionError as e:
            logger.error(f"API connection error while generating summary: {e}")
        except InvalidRequestError as e:
            logger.error(f"Invalid request error while generating summary: {e}")
        except Exception as e:
            logger.error(f"Error summarizing and saving ontology: {e}")

    def generate_summary(self, prompt: str, text: str) -> str:
        """
        Generate a summary of the provided text using OpenAI's API with retry mechanism.

        Args:
            prompt (str): Prompt for the summarization task.
            text (str): Text to be summarized.

        Returns:
            str: Summary of the text.
        """
        try:
            full_prompt = f"{prompt}\n\n{text}"
            retries = 3
            backoff_factor = 1  # Increased backoff to handle rate limits better
            for attempt in range(retries):
                try:
                    response = openai.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "system", "content": "You are a detailed and thorough summarizer."},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=500  # Adjust as needed
                    )
                    summary = response.choices[0].message.content
                    return summary
                except RateLimitError as e:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning(f"Rate limit exceeded. Retrying in {wait} seconds...")
                    time.sleep(wait)
                except APIConnectionError as e:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning(f"API connection error: {e}. Retrying in {wait} seconds...")
                    time.sleep(wait)
                except InvalidRequestError as e:
                    logger.error(f"Invalid request: {e}. Check your prompt and parameters.")
                    break  # Do not retry on invalid requests
                except APIError as e:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning(f"API error: {e}. Retrying in {wait} seconds...")
                    time.sleep(wait)
            # After retries
            logger.error("Exceeded maximum retry attempts for OpenAI API.")
            raise Exception("Rate limit exceeded or API error. Please try again later.")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
