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

    def semantic_search(self, query: str, node_labels: Optional[List[str]] = None,
                       limit: int = 5) -> List[Dict]:
        """
        Execute a semantic search query against the graph database
        Args:
            query (str): The semantic search query
            node_labels (list): Optional list of node labels to search within
            limit (int): Maximum number of results to return
        Returns:
            list: List of matching nodes with their properties
        """
        label_filter = ""
        if node_labels:
            label_filter = f"WHERE any(label IN labels(n) WHERE label IN {node_labels})"

        cypher_query = f"""
        MATCH (n)
        {label_filter}
        WITH n, properties(n) as props
        WHERE any(value IN [value IN props WHERE value IS NOT NULL] 
                 WHERE toString(value) CONTAINS $query)
        RETURN n
        LIMIT $limit
        """
        
        return self.execute_query(cypher_query, {
            "query": query,
            "limit": limit
        })
