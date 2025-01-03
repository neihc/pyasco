from typing import Optional, List, Dict, Any
from neo4j import GraphDatabase
import logging
from ..logger_config import setup_logger

# Setup verbose logger for Graph DB interactions
graphdb_logger = setup_logger('graphdb', 'graphdb_verbose.log', verbose=True)

# Initialize driver as None - will be set by configure_driver
driver = None

def configure_driver(uri: str, username: str, password: str):
    """Configure the Neo4j driver with the given credentials"""
    global driver
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        graphdb_logger.info("Successfully connected to Neo4j database")
    except Exception as e:
        graphdb_logger.error(f"Failed to connect to Neo4j: {e}")
        raise

def close_driver():
    """Close the Neo4j driver connection"""
    global driver
    if driver:
        driver.close()
        driver = None

def execute_query(query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
    """
    Execute a Cypher query against Neo4j database.
    Args:
        query (str): The Cypher query to execute
        parameters (dict): Optional parameters for the query
    Returns:
        list: List of records returned by the query
    """
    if not driver:
        raise RuntimeError("Neo4j driver not configured. Call configure_driver first.")
    
    try:
        # Log query details
        graphdb_logger.debug("=" * 80)
        graphdb_logger.debug("GRAPH DB QUERY")
        graphdb_logger.debug(f"Query: {query}")
        if parameters:
            graphdb_logger.debug(f"Parameters: {parameters}")

        with driver.session() as session:
            result = session.run(query, parameters or {})
            records = [dict(record) for record in result]
            
            # Log response details
            graphdb_logger.debug("=" * 80)
            graphdb_logger.debug("GRAPH DB RESPONSE")
            graphdb_logger.debug(f"Records: {records}")
            
            return records
    except Exception as e:
        graphdb_logger.error(f"Query execution failed: {e}")
        raise

def create_node(label: str, properties: Dict[str, Any]) -> Dict:
    """
    Create a new node in the graph database.
    Args:
        label (str): Label for the node
        properties (dict): Properties of the node
    Returns:
        dict: Created node properties
    """
    query = f"CREATE (n:{label} $props) RETURN n"
    result = execute_query(query, {"props": properties})
    return result[0]['n'] if result else None

def create_relationship(from_node_id: int, to_node_id: int, 
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
    WHERE id(a) = $from_id AND id(b) = $to_id
    CREATE (a)-[r:`{}`]->(b)
    SET r = $props
    RETURN r
    """.format(relationship_type)
    
    params = {
        "from_id": from_node_id,
        "to_id": to_node_id,
        "props": properties or {}
    }
    
    result = execute_query(query, params)
    return result[0]['r'] if result else None
