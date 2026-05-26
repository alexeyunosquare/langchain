# MCP Client Configuration

This document describes how to connect MCP clients to the Agentic RAG server.

## Prerequisites

Set the following environment variables before starting the server:

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key (or LM Studio) | Yes |
| `LM_STUDIO_HOST` | LM Studio host | Default: `localhost` |
| `LM_STUDIO_PORT` | LM Studio port | Default: `1234` |
| `TAVILY_API_KEY` | Tavily web search API key | For web search |
| `CHROMA_PERSIST_DIR` | Persistent vector store dir | For persistence |
| `MCP_API_TOKEN` | Bearer token for auth | Yes |
| `MCP_HOST` | HTTP bind host | Default: `127.0.0.1` |
| `MCP_PORT` | HTTP bind port | Default: `8000` |

## Starting the Server

```bash
# Option 1: Module entry point
MCP_API_TOKEN=your-secret-token python -m mcp_server.server

# Option 2: Console script (after pip install)
MCP_API_TOKEN=your-secret-token agentic-rag-mcp
```

The server starts on `http://127.0.0.1:8000/mcp` by default.

## Claude Desktop Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agentic-rag": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

## Cursor Configuration

Create `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "agentic-rag": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `query` | Answer a question using agentic RAG with self-correction |
| `search_documents` | Search documents without generating an answer |
| `validate_answer` | Validate an answer for hallucinations |
| `add_document` | Ingest a text file into the vector store |
| `get_config` | View current server configuration |
| `reset_conversation` | Clear conversation memory |

## Python Client Example

```python
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="http://localhost:8000/mcp",
    auth="your-secret-token",
)

async with Client(transport) as client:
    result = await client.call_tool("query", {"query": "What is RAG?"})
    print(result)
```