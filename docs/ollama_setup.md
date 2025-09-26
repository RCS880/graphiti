# Ollama Integration Guide for Graphiti

This guide explains how to use Graphiti with Ollama for fully local LLM-powered knowledge graph extraction.

## Overview

The OllamaClient provides first-class support for Ollama's local language models, handling the specific challenges of structured output generation from models that may include reasoning text, schema echoes, and other formatting issues.

## Features

- **Automatic Detection**: Graphiti automatically uses OllamaClient when Ollama environment variables are detected
- **Robust JSON Sanitization**: Handles thinking blocks, markdown formatting, schema echoes, and malformed JSON
- **Embedding Support**: Native support for Ollama's embedding models
- **Cloud Fallback**: Seamlessly fall back to cloud providers (OpenAI, Moonshot) when needed
- **Metrics Tracking**: Monitor sanitization success rates and performance

## Quick Start

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or via Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 2. Pull Required Models

```bash
# For completions (choose one)
ollama pull deepseek-r1:7b        # Recommended for entity extraction (fast, accurate)
ollama pull llama3:8b              # Lightweight alternative
ollama pull gpt-oss:20b           # For complex reasoning tasks (overkill for basic extraction)

# For embeddings
ollama pull nomic-embed-text       # Recommended embedder
```

### 3. Configure Environment Variables

```bash
# Basic Ollama configuration
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="deepseek-r1:7b"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
export OLLAMA_API_KEY="ollama"  # Any non-empty string

# Optional: Separate embedding endpoint
export OLLAMA_EMBED_BASE_URL="http://localhost:11434/v1"

# Optional: Cloud fallback (Moonshot Kimi)
export KIMI_K2_API_KEY="sk-your-api-key"
export KIMI_K2_MODEL="kimi-k2-0905-preview"
export KIMI_K2_BASE_URL="https://api.moonshot.ai/v1"
```

### 4. Initialize Graphiti

```python
from graphiti_core import Graphiti

# OllamaClient will be auto-detected and used
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Add episodes with local processing
await graphiti.add_episode(
    name="local_extraction",
    episode_body="Alice works with Bob on the DataPipe project.",
    source_description="meeting_notes"
)
```

## Advanced Configuration

### Custom Client Configuration

```python
from graphiti_core.llm_client import OllamaClient, LLMConfig
from graphiti_core import Graphiti

# Create custom Ollama client
config = LLMConfig(
    api_key="ollama",
    model="llama3:70b",
    base_url="http://gpu-server:11434/v1",
    temperature=0.7,
    max_tokens=2000
)

ollama_client = OllamaClient(config=config)

# Use with Graphiti
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=ollama_client
)
```

### Monitoring Sanitization Performance

```python
# Access sanitization metrics
metrics = ollama_client.get_sanitization_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Recoverable errors: {metrics['recoverable']}")
print(f"Unrecoverable errors: {metrics['unrecoverable']}")
```

## JSON Sanitization Pipeline

The OllamaClient implements a robust sanitization pipeline to handle various output formats:

1. **Remove Thinking Blocks**: Strips `<think>...</think>` sections
2. **Extract from Markdown**: Unwraps JSON from ` ```json...``` ` blocks
3. **Remove Schema Echoes**: Filters out repeated JSON schemas
4. **Parse JSON Objects**: Attempts direct parsing and regex extraction
5. **Validate with Pydantic**: Ensures output matches expected model

### Example: Handling Messy Output

Input from Ollama:
```
<think>
I need to extract entities and relationships from this text.
Let me analyze the sentence structure...
</think>

Based on my analysis, here's the extracted information:

```json
{
  "entities": ["Alice", "Bob", "DataPipe"],
  "relationships": ["works_with", "contributes_to"],
  "confidence": 0.95
}
```
```

Output after sanitization:
```json
{
  "entities": ["Alice", "Bob", "DataPipe"],
  "relationships": ["works_with", "contributes_to"],
  "confidence": 0.95
}
```

## Model Recommendations

### For Entity Extraction
- **gpt-oss:20b** - Optimized for graph extraction orchestration, excellent reasoning
- **deepseek-r1:7b** - Good alternative with structured output
- **llama3:8b** - Faster inference, lower resource usage
- **mixtral:8x7b** - Highest quality but resource-intensive

### For Embeddings
- **nomic-embed-text** - 768-dimensional, optimized for semantic search
- **mxbai-embed-large** - 1024-dimensional, higher quality
- **all-minilm** - Lightweight option for resource-constrained environments

## Troubleshooting

### Issue: JSON Parsing Failures

If you're seeing frequent sanitization failures:

1. **Check Model Output**: Some models need better prompting
   ```python
   # The client automatically adds hints for JSON output
   # You can also try different models
   export OLLAMA_MODEL="mixtral:8x7b"
   ```

2. **Enable Debug Logging**:
   ```python
   import logging
   logging.getLogger('graphiti_core.llm_client.ollama_client').setLevel(logging.DEBUG)
   ```

3. **Review Metrics**:
   ```python
   metrics = ollama_client.get_sanitization_metrics()
   if metrics['success_rate'] < 0.8:
       print("Consider switching to a different model")
   ```

### Issue: Slow Performance

1. **Use GPU Acceleration**:
   ```bash
   # Check if GPU is detected
   ollama list
   ```

2. **Adjust Model Size**:
   ```bash
   # Use smaller models for faster inference
   export OLLAMA_MODEL="llama3:7b"
   ```

3. **Configure Timeouts**:
   ```python
   import httpx
   client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
   ollama_client = OllamaClient(client=client)
   ```

### Issue: Empty Embeddings

The client automatically retries once if embeddings are empty. If the issue persists:

```bash
# Try a different embedding model
export OLLAMA_EMBED_MODEL="mxbai-embed-large"

# Or use a separate embedding service
export OLLAMA_EMBED_BASE_URL="http://embedding-server:11434/v1"
```

## Cloud Fallback Strategy

For production reliability, configure automatic fallback to cloud providers:

```python
import os

# Primary: Try Ollama first
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434/v1'

# Fallback: Use Moonshot Kimi if Ollama fails
os.environ['KIMI_K2_API_KEY'] = 'your-api-key'

# The factory will automatically select the appropriate client
from graphiti_core.llm_client import create_llm_client

client = create_llm_client()  # Returns OllamaClient or falls back
```

## Performance Benchmarks

Typical performance with OllamaClient (on M2 Max):

| Model | Extraction Time | Success Rate | Memory Usage |
|-------|-----------------|--------------|--------------|
| deepseek-r1:7b | ~2.5s | 92% | 8GB |
| llama3:8b | ~1.8s | 88% | 10GB |
| mixtral:8x7b | ~5.2s | 95% | 48GB |

## Contributing

If you encounter issues or have improvements for the OllamaClient:

1. Check existing issues: https://github.com/getzep/graphiti/issues
2. Run tests: `pytest tests/llm_client/test_ollama_client.py`
3. Submit PRs with sanitization improvements

## Related Documentation

- [Graphiti Documentation](https://docs.getzep.com/graphiti)
- [Ollama Documentation](https://ollama.com/docs)
- [Neo4j Setup Guide](./neo4j_setup.md)