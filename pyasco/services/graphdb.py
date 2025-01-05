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

    def find_similar_nodes(self, query_embedding: List[float], threshold: float = 0.7) -> List[Dict]:
        """
        Find nodes with similar embeddings using cosine similarity
        Args:
            query_embedding (list): The query embedding vector
            threshold (float): Minimum similarity score threshold
        Returns:
            list: List of nodes with their similarity scores
        """
        # Convert query embedding to string format for Cypher
        query_embedding_str = str(query_embedding)
        
        query = """
        MATCH (n)
        WHERE EXISTS(n.embedding)
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT 5
        """
        
        results = self.execute_query(
            query,
            {
                "query_embedding": query_embedding,
                "threshold": threshold
            }
        )
        
        return [
            {
                "node": record["n"],
                "similarity": record["similarity"]
            }
            for record in results
        ]

    def _check_vector_index(self, node_label: str) -> bool:
        """Check if vector index exists for a given node label"""
        try:
            result = self.execute_query("""
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR' 
            AND $label IN labelsOrTypes
            AND 'embedding' IN properties
            RETURN count(*) as count
            """, {"label": node_label})
            
            return result[0]['count'] > 0
        except Exception as e:
            self.logger.error(f"Failed to check vector index: {str(e)}")
            return False

    def ensure_vector_index(self, node_label: str) -> bool:
        """Ensure vector index exists for the specified node label"""
        try:
            if self._check_vector_index(node_label):
                self.logger.info(f"Vector index already exists for {node_label}")
                return True

            self.execute_query(f"""
            CREATE VECTOR INDEX {node_label.lower()}_embeddings IF NOT EXISTS 
            FOR (n:{node_label}) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create vector index: {str(e)}")
            return False

    def get_node_neighbors(self, node_id: int) -> List[Dict]:
        """Get all neighbors of a node"""
        try:
            return self.execute_query("""
            MATCH (n)-[r]-(neighbor)
            WHERE elementId(n) = $node_id
            RETURN neighbor, type(r) as relationship_type
            """, {"node_id": node_id})
        except Exception as e:
            self.logger.error(f"Failed to get node neighbors: {str(e)}")
            return []

    def get_vector_search_results(self, label: str, query_embedding: List[float], 
                                similarity_threshold: float, limit: int = 5) -> List[Dict]:
        """Execute vector similarity search for a specific label"""
        try:
            return self.execute_query(f"""
            CALL db.index.vector.queryNodes($index_name, $k, $query)
            YIELD node, score 
            WHERE score >= $threshold
            RETURN node, score
            ORDER BY score DESC
            """, {
                "index_name": f"{label.lower()}_embeddings",
                "k": limit,
                "query": query_embedding,
                "threshold": similarity_threshold
            })
        except Exception as e:
            self.logger.error(f"Vector search failed for label {label}: {str(e)}")
            return []

    def get_indexed_labels(self) -> List[str]:
        """Get all labels that have vector indexes"""
        try:
            results = self.execute_query("""
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            RETURN distinct labelsOrTypes[0] as label
            """)
            return [result['label'] for result in results]
        except Exception as e:
            self.logger.error(f"Failed to get indexed labels: {str(e)}")
            return []

    def get_schema(self) -> str:
        """Get the actual database schema including nodes, relationships and patterns"""
        try:
            # Get node labels
            nodes_schema = self.execute_query("""
            CALL db.labels() YIELD label
            CALL {
                WITH label
                MATCH (n:`${label}`)
                WITH label, n
                LIMIT 1
                RETURN collect(keys(n)) as properties
            }
            RETURN label as nodeType, properties[0] as properties
            """)
            
            # Get relationship types and their properties
            rels_schema = self.execute_query("""
            CALL db.relationshipTypes() YIELD relationshipType
            CALL {
                WITH relationshipType
                MATCH ()-[r:`${relationshipType}`]->()
                WITH relationshipType, r
                LIMIT 1
                RETURN collect(keys(r)) as properties
            }
            RETURN relationshipType as relationType, properties[0] as properties
            """)
            
            # Get relationship patterns
            rel_patterns = self.execute_query("""
            MATCH (start)-[r]->(end)
            RETURN DISTINCT
                labels(start)[0] as fromLabel,
                type(r) as relType,
                labels(end)[0] as toLabel
            """)
            
            # Format schema as detailed string
            schema = []
            
            # Add nodes section
            schema.append("Nodes:")
            for node in nodes_schema:
                props = node['properties'] or []
                schema.append(f"- Label: {node['nodeType']}")
                schema.append("  Properties: " + ", ".join(props))
            
            # Add relationships section
            schema.append("\nRelationships:")
            for rel in rels_schema:
                props = rel['properties'] or []
                schema.append(f"- Type: {rel['relationType']}")
                schema.append("  Properties: " + ", ".join(props))
            
            # Add relationship patterns section
            schema.append("\nRelationship Patterns:")
            for pattern in rel_patterns:
                schema.append(
                    f"- ({pattern['fromLabel']})-[:{pattern['relType']}]->({pattern['toLabel']})"
                )
            
            return "\n".join(schema)
            
        except Exception as e:
            self.logger.error(f"Failed to get database schema: {str(e)}")
            return "Schema retrieval failed"

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

