from dataclasses import dataclass, field
from typing import Dict, Optional, List
import yaml
import os


@dataclass
class LLMConfig:
    model: str = "meta-llama/llama-3.3-70b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None

@dataclass
class MemoryConfig:
    enabled: bool = False

@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    custom_instructions: Optional[str] = None

class ConfigManager:
    @staticmethod
    def load_from_yaml(file_path: str) -> Config:
        """Load configuration from YAML file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        llm_config = LLMConfig(
            model=yaml_config.get('llm', {}).get('model', "meta-llama/llama-3.3-70b-instruct"),
            base_url=yaml_config.get('llm', {}).get('base_url', "https://openrouter.ai/api/v1"),
            api_key=yaml_config.get('llm', {}).get('api_key')
        )

        # Configure memory settings if present
        memory_config = None
        if 'memory' in yaml_config:
            memory_config = MemoryConfig(
                enabled=yaml_config['memory'].get('enabled', False),
            )

        return Config(
            llm=llm_config,
            memory=memory_config or MemoryConfig(),
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
                    
        llm_config = LLMConfig(
            model=args.model,
            base_url=args.llm_base_url if hasattr(args, 'llm_base_url') else "https://openrouter.ai/api/v1",
            api_key=args.llm_api_key if hasattr(args, 'llm_api_key') else None
        )

        # Configure memory if args present
        memory_config = None
        if hasattr(args, 'memory_enabled'):
            memory_config = MemoryConfig(
                enabled=args.memory_enabled,
            )

        return Config(
            llm=llm_config,
            memory=memory_config or MemoryConfig(),
            custom_instructions=args.custom_instructions if hasattr(args, 'custom_instructions') else None
        )
