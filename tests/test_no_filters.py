from app.db.agent_system import ReActAgent, AgentConfig


def test_strip_filters_from_action():
    calls = {}

    def fake_llm(prompt: str) -> str:
        # Model returns a Thought and an Action that includes disallowed filters
        return 'Thought: run a keyword search\nAction: keyword_search[query="algorithms", bucketLevel=3000, subject="CS"]'

    def fake_tool(query: str, k: int = 3, bucketLevel: int | None = None, subject: str | None = None):
        # Capture what the tool actually received
        calls['received'] = {'query': query, 'bucketLevel': bucketLevel, 'subject': subject}
        return {'query': query, 'results': []}

    agent = ReActAgent(
        llm=fake_llm,
        tools={'keyword_search': {'fn': fake_tool}},
        config=AgentConfig(max_steps=1, stop_after_first_tool=True, verbose=False),
    )

    # Run with useFilters=False; tool should NOT receive bucketLevel or subject
    _ = agent.run("find algorithms", useFilters=False)

    assert 'received' in calls
    assert calls['received']['bucketLevel'] is None
    assert calls['received']['subject'] is None
