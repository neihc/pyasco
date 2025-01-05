from typing import Optional, List, Dict, Any
from neo4j import GraphDatabase
import logging
from ..logger_config import setup_logger

class GraphDB:
    """Class for handling Neo4j database operations"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize GraphDB with optional immediate connection
        Args:
            uri (str): Neo4j database URI
            username (str): Database username
            password (str): Database password
        """
        self.logger = setup_logger('graphdb', 'graphdb_verbose.log', verbose=True)
        self.driver = None
        self.indexes = {}  # Track created indexes
        if uri and username and password:
            self.configure(uri, username, password)
    
    def configure(self, uri: str, username: str, password: str):
        """Configure the Neo4j driver with the given credentials"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """
        Execute a Cypher query against Neo4j database.
        Args:
            query (str): The Cypher query to execute
            parameters (dict): Optional parameters for the query
        Returns:
            list: List of records returned by the query
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not configured. Call configure first.")
        
        try:
            # Log query details
            self.logger.debug("=" * 80)
            self.logger.debug("GRAPH DB QUERY")
            self.logger.debug(f"Query: {query}")
            if parameters:
                self.logger.debug(f"Parameters: {parameters}")

            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = [dict(record) for record in result]
                
                # Log response details
                self.logger.debug("=" * 80)
                self.logger.debug("GRAPH DB RESPONSE")
                self.logger.debug(f"Records: {records}")
                
                return records
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict:
        """
        Create a new node in the graph database.
        Args:
            label (str): Label for the node
            properties (dict): Properties of the node
        Returns:
            dict: Created node properties
        """
        query = f"CREATE (n:{label} $props) RETURN n"
        result = self.execute_query(query, {"props": properties})
        return result[0]['n'] if result else None

    def create_relationship(self, from_node_id: int, to_node_id: int,
                          relationship_type: str, properties: Dict[str, Any] = None) -> Dict:
        """
        Create a relationship between two nodes.
        Args:
            from_node_id (int): ID of the source node
            to_node_id (int): ID of the target node
            relationship_type (str): Type of relationship
            properties (dict): Optional properties for the relationship
        Returns:
            dict: Created relationship properties
        """
        query = """
        MATCH (a), (b) 
        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
        CREATE (a)-[r:`{}`]->(b)
        SET r = $props
        RETURN r
        """.format(relationship_type)
        
        params = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "props": properties or {}
        }
        
        result = self.execute_query(query, params)
        return result[0]['r'] if result else None

    def _node_to_dict(self, node) -> Dict:
        """Convert a Neo4j Node to a dictionary"""
        if not node:
            return None
        return {
            "id": node.id,
            "labels": list(node.labels),
            "properties": dict(node)
        }

    def create_text_index(self, label: str, properties: List[str]) -> bool:
        """
        Create a full-text index for the specified label and properties
        Args:
            label (str): Node label to index
            properties (list): List of property names to include in the index
        Returns:
            bool: True if index was created successfully
        """
        try:
            index_name = f"{label.lower()}_text_idx"
            if index_name in self.indexes:
                self.logger.info(f"Index {index_name} already exists")
                return True

            query = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON EACH [{', '.join(f'n.{prop}' for prop in properties)}]
            """
            self.execute_query(query)
            self.indexes[index_name] = {'label': label, 'properties': properties}
            self.logger.info(f"Created text index {index_name} for {label} on {properties}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            return False

