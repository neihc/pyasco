# PyAsco - Python AI Assistant Console

PyAsco is an AI-powered console application that helps you interact with a smart assistant capable of executing Python code, managing skills, and providing intelligent responses.

## Features

- Interactive console interface with command history
- Code execution in local or Docker environments
- Skill learning and improvement capabilities
- Streaming responses from AI
- Rich markdown formatting for outputs

## Installation

There are two ways to install PyAsco:

### 1. Install from PyPI (Recommended)
```bash
uv pip install pyasco
```

### 2. Install from source
```bash
git clone https://github.com/neihc/pyasco.git
cd pyasco
uv tool install -e .
```

## Usage

After installation, you can run the console application from anywhere:

```bash
pyco
```

### Command Line Options

- `--config`: Path to YAML configuration file
- `--use-docker`: Run code in Docker environment
- `--docker-image`: Specify Docker image (default: python:3.9-slim)
- `--mem-limit`: Docker memory limit (default: 512m)
- `--cpu-count`: Docker CPU count (default: 1)
- `--env-file`: Path to environment file for Docker
- `--mount`: Mount points for Docker (format: host_path:container_path)
- `--model`: LLM model to use (default: meta-llama/llama-3.3-70b-instruct)
- `--skills-path`: Path to skills directory (default: skills)

### Magic Commands

- `%exit`: Quit the console
- `%reset`: Start a new conversation
- `%learn_that_skill`: Convert the current conversation into a reusable skill
- `%improve_that_skill`: Improve an existing skill based on current conversation

### Example Usage

1. Start a basic session:
```bash
pyco
```

2. Run with Docker support:
```bash
pyco --use-docker --docker-image python:3.9-slim
```

3. Use a custom configuration:
```bash
pyco --config my_config.yaml
```

## Configuration

Create a YAML configuration file to customize the behavior:

```yaml
docker:
  use_docker: true
  image: "python:3.9-slim"
  mem_limit: "512m"
  cpu_count: 1
  env_file: ".env"
  mounts:
    - "host_path:container_path"

model: "meta-llama/llama-3.3-70b-instruct"
skills_path: "skills"
```

## License

MIT License
