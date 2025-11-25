"""Simple example to test the ReActAgent loop.

Run from the repo root with your venv active:

python run_agent_example.py

This uses a fake LLM (deterministic) and a simple `search` tool to show
how the agent constructs prompts, calls the LLM, executes tools, and
returns a final answer.
"""
from agent_system import ReActAgent, AgentConfig

# Import the real search modules from the project. These imports are required
# — the example will raise ImportError immediately if the project's search
# implementations or embeddings are not available. This makes the script fail
# loudly so you can fix environment/dependencies rather than silently fallback.
from app.db import tfidf_search as tfidf_mod
from app.db import load_embeddings as load_mod
from app.db import embedding_search as emb_mod

# A fake LLM that returns two-step responses. It ignores the prompt
# content for simplicity and returns a Thought/Action pair.
def make_fake_llm():
    calls = {"n": 0}
    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        # Extract the user question from the prompt so the fake LLM can 'decide'
        uq = ""
        m = None
        try:
            # prompt includes a line like: User Question: <text>\n\n
            m = [ln for ln in prompt.splitlines() if ln.strip().startswith("User Question:")]
            if m:
                uq = m[0].split("User Question:", 1)[1].strip()
        except Exception:
            uq = ""

        # If the user question is long (many words) prefer semantic_search, otherwise keyword_search
        word_count = len(uq.split()) if uq else 0
        if calls["n"] == 1:
            if word_count > 6:
                return 'Thought: The user asked a longer question; semantic similarity will help.\nAction: semantic_search[query="' + uq.replace('"', '\\"') + '", k=3]'
            else:
                return 'Thought: I should search for the course details.\nAction: keyword_search[query="CS 5100 course description", k=2]'
        else:
            # finish with a concise answer based on the (simulated) observation
            return 'Thought: I found the course info, now finish.\nAction: finish[answer="CS 5100 (Intro to AI) covers core AI techniques including search, logic, and learning; typically 3 credits."]'
    return fake_llm

# A simple search tool that returns structured results.
def keyword_search(query: str, k: int = 3, bucketLevel: int | None = None, subject: str | None = None):
    """
    Always call the project's TF-IDF `tool_search` implementation.
    This function will raise if `tfidf_mod.tool_search` is not present or if
    the underlying data/DB is not correctly configured.
    """
    if not hasattr(tfidf_mod, "tool_search"):
        raise RuntimeError("app.db.tfidf_search.tool_search is not available")

    payload = tfidf_mod.tool_search(query=query, bucketLevel=bucketLevel, subject=subject, credits=None, k=k)
    # Expect payload to contain a 'results' list of dicts
    return {"query": query, "results": payload.get("results", [])}


def semantic_search(query: str, k: int = 3):
    """
    Always run the project's embedding-based semantic search using the
    repository's `load_embeddings` and `embedding_search` functions.
    This will raise an exception if the embeddings file is missing or if the
    required packages (sentence-transformers, pandas, numpy) are not installed.
    """
    if not hasattr(load_mod, "load_embeddings"):
        raise RuntimeError("app.db.load_embeddings.load_embeddings is not available")
    if not hasattr(emb_mod, "embedding_search"):
        raise RuntimeError("app.db.embedding_search.embedding_search is not available")

    courses, embeddings = load_mod.load_embeddings()
    hits = emb_mod.embedding_search(query, courses, embeddings, k=k)
    results = [{"id": h.get("id"), "title": h.get("title"), "snippet": h.get("text"), "score": h.get("score")} for h in hits]
    return {"query": query, "results": results}

if __name__ == '__main__':
    llm = make_fake_llm()
    tools = {
        "keyword_search": {"fn": keyword_search},
        "semantic_search": {"fn": semantic_search},
    }
    # Use the AgentConfig default allow_tools so both searches are permitted
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        config=AgentConfig(max_steps=6)
    )

    question = "What is CS 5100 (Intro to AI) about and how many credits is it?"
    result = agent.run(question)

    print("\n--- AGENT RUN RESULT ---")
    print(f"Question: {result['question']}")
    print(f"Final answer: {result['final_answer']}")
    print("Steps:")
    for i, s in enumerate(result['steps'], 1):
        print(f"Step {i} - Thought: {s['thought']}")
        print(f"         Action: {s['action']}")
        print(f"         Observation: {s['observation']}\n")
