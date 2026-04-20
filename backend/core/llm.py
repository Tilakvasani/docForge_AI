"""
Shared Azure OpenAI LLM factory for DocForge AI.

Provides a single source of truth for all `AzureChatOpenAI` instances used
across the application. The default model is pre-initialized as a module-level
singleton to avoid repeated construction overhead and eliminate race conditions
under concurrent requests.

Usage:
    from backend.core.llm import get_llm

    llm = get_llm()                   # singleton (temp=0.2, max_tokens=3000)
    llm = get_llm(temperature=0.0)    # fresh instance for specialized tasks
"""

from backend.core.config import settings
from langchain_openai import AzureChatOpenAI


_DEFAULT_LLM: AzureChatOpenAI = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_LLM_ENDPOINT,
    api_key=settings.AZURE_OPENAI_LLM_KEY,
    azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
    api_version=settings.AZURE_LLM_API_VERSION,
    temperature=0.2,
    max_tokens=3000,
)


def get_llm(temperature: float = 0.2, max_tokens: int = 3000) -> AzureChatOpenAI:
    """
    Return an `AzureChatOpenAI` instance configured for DocForge AI.

    When called with the default parameters, returns the pre-built module-level
    singleton — safe under all concurrency models with zero construction
    overhead. For any other parameter combination, a fresh instance is
    constructed and returned (useful for one-off tasks like summarization at
    temperature 0.0).

    Args:
        temperature: Sampling temperature passed to the Azure deployment.
                     Lower values produce more deterministic output.
        max_tokens:  Maximum tokens the model may generate per response.

    Returns:
        A configured `AzureChatOpenAI` instance.
    """
    if temperature == 0.2 and max_tokens == 3000:
        return _DEFAULT_LLM

    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version=settings.AZURE_LLM_API_VERSION,
        temperature=temperature,
        max_tokens=max_tokens,
    )