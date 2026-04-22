"""
Shared Vector DB and Embedder factory for DocForge AI.

Provides singletons for ChromaDB Client and AzureOpenAIEmbeddings to prevent
duplicate initialization overhead and race conditions.

Usage:
    from backend.core.vector import get_embedder, get_chroma_client

    embedder = get_embedder()
    client = get_chroma_client()
"""

import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from backend.core.config import settings

_embedder_instance = None
_chroma_client_instance = None

def get_embedder() -> AzureOpenAIEmbeddings:
    """Return the shared AzureOpenAIEmbeddings singleton."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
            api_version=settings.AZURE_EMB_API_VERSION,
            timeout=60,
            max_retries=2,
        )
    return _embedder_instance

def get_chroma_client():
    """Return the shared persistent ChromaDB client singleton."""
    global _chroma_client_instance
    if _chroma_client_instance is None:
        _chroma_client_instance = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return _chroma_client_instance
