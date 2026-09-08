# Local RAG Service

Local Retrieval-Augmented Generation stack running private LLMs with search and tool-calling capabilities.

## Architecture

- **Llama.cpp Server**: Runs GGUF models (`Hermes-3-Llama-3.1-8B-GGUF`) with CUDA acceleration and OpenAI-compatible completions endpoint.
- **SearXNG**: Privacy-respecting metasearch engine used for live web search and grounding.
- **API Proxy**: FastAPI middleware handling Hermes function-calling syntax (`<tool_call>`), dispatching web queries and custom tools, and feeding structured responses back to the LLM.

## Usage

```bash
docker compose up -d
```

Services:
- **API Proxy**: `http://localhost:8000`
- **Llama.cpp Server**: `http://localhost:8080`
- **SearXNG**: `http://localhost:3000`
