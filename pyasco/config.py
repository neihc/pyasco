from dataclasses import dataclass, field
from typing import Dict, Optional, List
import yaml
import os

@dataclass
class DockerConfig:
    use_docker: bool = False
    image: str = "python:3.9-slim"
    mem_limit: str = "512m"
    cpu_count: int = 1
    volumes: Dict = None
    env_file: Optional[str] = None
    bash_command: str = "/bin/bash"
    python_command: str = "python"

@dataclass
class LLMConfig:
    model: str = "meta-llama/llama-3.3-70b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None

@dataclass
class GraphDBConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: Optional[str] = None

@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-ada-002"
    dimensions: int = 1536
    batch_size: int = 100
    cache_dir: Optional[str] = None

@dataclass
class MemoryConfig:
    enabled: bool = False
    instructions: str = """
    Memory Schema:
    - Nodes can be of type: Conversation, CodeSnippet, Command, Error, Result
    - Properties should include: content, timestamp, type, metadata
    - Relationships: FOLLOWED_BY, GENERATED, CAUSED, RELATED_TO
    
    Guidelines:
    1. Store complete context and metadata
    2. Maintain chronological order
    3. Link related items with appropriate relationships
    4. Include error states and outcomes
    """
    graph_db: Optional[GraphDBConfig] = field(default_factory=GraphDBConfig)

@dataclass
class Config:
    docker: DockerConfig = field(default_factory=DockerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    skills_path: str = "skills"
    custom_instructions: Optional[str] = None

class ConfigManager:
    @staticmethod
    def load_from_yaml(file_path: str) -> Config:
        """Load configuration from YAML file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        docker_config = DockerConfig(
            use_docker=yaml_config.get('docker', {}).get('use_docker', False),
            image=yaml_config.get('docker', {}).get('image', "python:3.9-slim"),
            mem_limit=yaml_config.get('docker', {}).get('mem_limit', "512m"),
            cpu_count=yaml_config.get('docker', {}).get('cpu_count', 1),
            volumes=yaml_config.get('docker', {}).get('volumes', {}),
            env_file=yaml_config.get('docker', {}).get('env_file'),
            bash_command=yaml_config.get('docker', {}).get('bash_command', "/bin/bash"),
            python_command=yaml_config.get('docker', {}).get('python_command', "python")
        )
        
        llm_config = LLMConfig(
            model=yaml_config.get('llm', {}).get('model', "meta-llama/llama-3.3-70b-instruct"),
            base_url=yaml_config.get('llm', {}).get('base_url', "https://openrouter.ai/api/v1"),
            api_key=yaml_config.get('llm', {}).get('api_key')
        )

        # Configure memory settings if present
        memory_config = None
        if 'memory' in yaml_config:
            graph_db_config = None
            if 'graph_db' in yaml_config['memory']:
                graph_db_config = GraphDBConfig(
                    uri=yaml_config['memory']['graph_db'].get('uri', "bolt://localhost:7687"),
                    username=yaml_config['memory']['graph_db'].get('username', "neo4j"),
                    password=yaml_config['memory']['graph_db'].get('password')
                )

            memory_config = MemoryConfig(
                enabled=yaml_config['memory'].get('enabled', False),
                instructions=yaml_config['memory'].get('instructions', MemoryConfig.instructions),
                graph_db=graph_db_config
            )

        # Configure embedding settings if present
        embedding_config = None
        if 'embedding' in yaml_config:
            embedding_config = EmbeddingConfig(
                model=yaml_config['embedding'].get('model', "text-embedding-ada-002"),
                dimensions=yaml_config['embedding'].get('dimensions', 1536),
                batch_size=yaml_config['embedding'].get('batch_size', 100),
                cache_dir=yaml_config['embedding'].get('cache_dir')
            )
            
        return Config(
            docker=docker_config,
            llm=llm_config,
            memory=memory_config or MemoryConfig(),
            skills_path=yaml_config.get('skills_path', "skills"),
            custom_instructions=yaml_config.get('custom_instructions')
        )
    
    @staticmethod
    def from_args(args) -> Config:
        """Create configuration from command line arguments"""
        volumes = {}
        if hasattr(args, 'mount') and args.mount:
            for mount in args.mount:
                try:
                    host_path, container_path = mount.split(':')
                    volumes[host_path] = {'bind': container_path, 'mode': 'rw'}
                except ValueError:
                    continue
                    
        docker_config = DockerConfig(
            use_docker=args.use_docker,
            image=args.docker_image,
            mem_limit=args.mem_limit,
            cpu_count=args.cpu_count,
            volumes=volumes,
            env_file=args.env_file if hasattr(args, 'env_file') else None,
            bash_command=args.bash_command if hasattr(args, 'bash_command') else "/bin/bash",
            python_command=args.python_command if hasattr(args, 'python_command') else "python"
        )
        
        llm_config = LLMConfig(
            model=args.model,
            base_url=args.llm_base_url if hasattr(args, 'llm_base_url') else "https://openrouter.ai/api/v1",
            api_key=args.llm_api_key if hasattr(args, 'llm_api_key') else None
        )

        # Configure memory if args present
        memory_config = None
        if hasattr(args, 'memory_enabled'):
            graph_db_config = None
            if hasattr(args, 'graph_db_uri'):
                graph_db_config = GraphDBConfig(
                    uri=args.graph_db_uri,
                    username=args.graph_db_username if hasattr(args, 'graph_db_username') else "neo4j",
                    password=args.graph_db_password if hasattr(args, 'graph_db_password') else None
                )

            memory_config = MemoryConfig(
                enabled=args.memory_enabled,
                instructions=args.memory_instructions if hasattr(args, 'memory_instructions') else MemoryConfig.instructions,
                graph_db=graph_db_config
            )

        # Configure embedding if args present
        embedding_config = None
        if hasattr(args, 'embedding_model'):
            embedding_config = EmbeddingConfig(
                model=args.embedding_model,
                dimensions=args.embedding_dimensions if hasattr(args, 'embedding_dimensions') else 1536,
                batch_size=args.embedding_batch_size if hasattr(args, 'embedding_batch_size') else 100,
                cache_dir=args.embedding_cache_dir if hasattr(args, 'embedding_cache_dir') else None
            )
            
        return Config(
            docker=docker_config,
            llm=llm_config,
            memory=memory_config or MemoryConfig(),
            skills_path=args.skills_path if hasattr(args, 'skills_path') else "skills",
            custom_instructions=args.custom_instructions if hasattr(args, 'custom_instructions') else None
        )
