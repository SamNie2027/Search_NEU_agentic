"""
Agent runner module for executing ReAct agents with real LLM and search tools.

Provides functions to run agents with keyword search, semantic search,
and language model integration.
"""
from .agent_system import ReActAgent, AgentConfig
import sys
from pathlib import Path

_lm_module = None

def _get_lm():
    """Lazy import of language_model to avoid loading model at startup."""
    global _lm_module
    if _lm_module is None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from app.llm import language_model as lm
        _lm_module = lm
    return _lm_module

from app.search import tfidf_search as tfidf_mod
from app.search import load_embeddings as load_mod
from app.search import embedding_search as emb_mod


def get_llm():
    """Get the LLM function, loading the model if needed."""
    lm = _get_lm()
    return lm.LLM

def keyword_search(query: str, k: int = 3, bucketLevel: int | None = None, subject: str | None = None):
    """
    Always call the project's TF-IDF `tool_search` implementation.
    This function will raise if `tfidf_mod.tool_search` is not present or if
    the underlying data/DB is not correctly configured.
    """
    if not hasattr(tfidf_mod, "tool_search"):
        raise RuntimeError("app.search.tfidf_search.tool_search is not available")

    payload = tfidf_mod.tool_search(query=query, bucketLevel=bucketLevel, subject=subject, credits=None, k=k)
    return payload


def semantic_search(query: str, k: int = 3):
    """
    Always run the project's embedding-based semantic search using the
    repository's `load_embeddings` and `embedding_search` functions.
    This will raise an exception if the embeddings file is missing or if the
    required packages (sentence-transformers, pandas, numpy) are not installed.
    """
    if not hasattr(load_mod, "load_embeddings"):
        raise RuntimeError("app.search.load_embeddings.load_embeddings is not available")
    if not hasattr(emb_mod, "embedding_search"):
        raise RuntimeError("app.search.embedding_search.embedding_search is not available")

    courses, embeddings = load_mod.load_embeddings()
    hits = emb_mod.embedding_search(query, courses, embeddings, k=k)
    results = [{"id": h.get("id"), "title": h.get("title"), "snippet": h.get("text"), "score": h.get("score")} for h in hits]
    return {"query": query, "results": results}

def run_agent_with_real_llm(question: str, max_steps: int = 6, useFilters: bool = True):
    """
    Run a ReAct agent with the real LLM and search tools.
    
    Args:
        question: The user's search query
        max_steps: Maximum number of agent steps
        useFilters: Whether to allow filter parameters in tool calls
        
    Returns:
        Dictionary with search results, or None if no results found
    """
    tools = {
        "keyword_search": {"fn": keyword_search},
        "semantic_search": {"fn": semantic_search},
    }

    llm = get_llm()
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        config=AgentConfig(max_steps=max_steps, stop_after_first_tool=True)
    )

    return agent.run(question, useFilters)
