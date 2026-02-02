# RAG Engine

A Retrieval-Augmented Generation engine with support for multiple embedding models.

## Features

- Multiple embedding model support (OpenAI, BGE, Jina)
- PDF document processing with LlamaParse
- Vector-based semantic search
- Easy-to-use query interface

## Quick Start

```python
from rag_engine import rag_engine

# Configure
rag_engine.set_api_keys(openai_key="sk-...", llama_cloud_key="llx-...")

# Load and index
rag_engine.load_pdf("document.pdf")
rag_engine.initialize_embedding_models()
rag_engine.create_indexes()

# Query
results = rag_engine.query("What is the main topic?")
print(results[0].response)
