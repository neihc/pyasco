from pyasco.services.graphdb import GraphDB
from pyasco.services.llm import LLMService
from pyasco.agent.memory_handler import MemoryHandler
from datetime import datetime
import json

def demonstrate_recall(memory_handler, query: str, similarity_threshold: float = 0.7):
    """Helper function to demonstrate recall with detailed output"""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    results = memory_handler.recall(query, similarity_threshold=similarity_threshold)
    
    print("\nRESULTS:")
    if not results:
        print("No results found")
        return
        
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Node Type: {result.get('n', {}).get('labels', [])}")
        print(f"Properties: {json.dumps(result.get('n', {}).get('properties', {}), indent=2)}")
        if 'score' in result:
            print(f"Relevance Score: {result['score']}")
        print("-" * 40)

def main():
    # Initialize services
    graph_db = GraphDB(
        uri="neo4j+s://d46bbb85.databases.neo4j.io",
        username="neo4j",
        password="HpkGcQzPITyR8bKtRFp1w7QcRXgK0hMoamJq0DyFdXs"  # Replace with actual password
    )
    
    llm_service = LLMService()
    
    # Initialize memory handler with custom schema
    memory_instructions = """
    Additional schema for AI agent memories:
    
    Nodes:
    - Topic
        - name: subject of discussion
        - domain: technical/personal/project
        - complexity: basic/intermediate/advanced
        - last_discussed: timestamp
        - summary: text description
        
    - Skill
        - name: specific capability or knowledge
        - proficiency: beginner/intermediate/expert
        - last_used: timestamp
        - details: text description
        
    - Project
        - name: project identifier
        - status: active/completed/planned
        - start_date: timestamp
        - description: text description
        
    Relationships:
    - REQUIRES_SKILL: connects Topic/Project to required Skills
    - BUILDS_ON: shows prerequisite relationships
    - RELATED_TO: connects similar or related nodes
    - CONTRIBUTES_TO: connects Skills to Projects
    """
    
    memory_handler = MemoryHandler(
        memory_instructions=memory_instructions,
        llm_service=llm_service,
        graph_db=graph_db
    )
    
    # Example memory creation
    # project_memory = memory_handler.remember(
        # "Started working on a new RAG implementation project using Neo4j as the vector store.",
        # context={
            # "timestamp": datetime.now().isoformat(),
            # "project_name": "graph-rag",
            # "status": "active"
        # }
    # )
    
    # Demonstrate different types of recalls
    queries = [
        "What projects are currently active and what skills do they require?",
        "Find any discussions or topics related to RAG or vector databases",
        "What skills have been recently used in our projects?",
        "Show me the progression of topics and skills in the graph database domain",
    ]
    
    for query in queries:
        demonstrate_recall(memory_handler, query)
    
    graph_db.close()

if __name__ == "__main__":
    main()
