# Step 4: Integrating Components into an Agent System

# Finally, we will put everything together as an agent system, including the external database, the information search tools, and a base language model.

# The agent will take the input and use a language model to output the Thought and Action.
# The agent will execute the Action (which in this case is searching for a document) and concatenate the returned document as Observation to the next round of the prompt
# The agent will iterate over the above two steps until it identifies the answer or reaches an iteration limit.

# ----------------------------
# We will define the agent controller that combines everything we define above
# ----------------------------
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
from collections import Counter, defaultdict

# import python files from the same folder, such as language_model.py, knowledge_base.py, prompting_techniques.py
# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from prompting_techniques import make_prompt, parse_action

@dataclass
class Step:
    thought: str
    action: str
    observation: str
    llm_output: str = ""
    
@dataclass
class AgentConfig:
    max_steps: int = 6
    # Accept both keyword and semantic search by default so the model can choose
    allow_tools: Tuple[str, ...] = ("keyword_search", "semantic_search")
    verbose: bool = True
    # If True, stop the agent immediately after executing the first tool
    # (useful when you only want the model to perform a single search).
    stop_after_first_tool: bool = False

class ReActAgent:
    def __init__(self, llm: Callable[[str], str], tools: Dict[str, Dict[str, Any]], config: Optional[AgentConfig]=None):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []
        # Track recent identical actions to avoid infinite repeating loops
        self._recent_action_counts: Dict[str, int] = {}

    def run(self, user_query: str, useFilters: bool = True) -> Dict[str, Any]:
        self.trajectory.clear()
        self._recent_action_counts.clear()
        final_answer_from_tool = None
        for step_idx in range(self.config.max_steps):
            # 1. At each step, format the prompt based on the make_prompt function and self.trajectory
            # `make_prompt` expects the trajectory as a list of dicts with keys: thought, action, observation
            prompt = make_prompt(user_query, [asdict(s) for s in self.trajectory], useFilters=useFilters)

            # 2. Use self.llm to process the prompt
            try:
                out = self.llm(prompt)
            except Exception as e:
                out = f"Thought: (llm error)\nAction: finish[answer=\"LLM error: {e}\"]"

            # UNCOMMENT FOR LOGGING
            # If verbose, print the raw LLM output immediately (intermediate output)
            if self.config.verbose:
                print(f"\n--- Intermediate output (step {step_idx+1}) ---")
                print(out)

            # Expect two lines: Thought:..., Action:...
            t_match = re.search(r"Thought:\s*(.*)", out)
            a_match = re.search(r"Action:\s*(.*)", out)
            thought = t_match.group(1).strip() if t_match else "(no thought)"
            action_line = a_match.group(1).strip() if a_match else "finish[answer=\"(no action)\"]"
            # If the captured text already contains a stray 'Action:' prefix (e.g. "Action: finish[...]")
            # remove it to avoid producing "Action: Action: ...". Be defensive about casing/whitespace.
            if re.match(r"^\s*Action:\s*", action_line, flags=re.IGNORECASE):
                action_line = re.sub(r"^\s*Action:\s*", "", action_line, flags=re.IGNORECASE)
            action_line = "Action: " + action_line

            # 3. Parse the action of the action line using the parse_action function
            parsed = parse_action(action_line)

            if not parsed:
                observation = "Invalid action format. Stopping."
                self.trajectory.append(Step(thought, action_line, observation, out))
                break
            name, args = parsed

            if name == "finish":
                observation = "done"
                self.trajectory.append(Step(thought, action_line, observation, out))
                break

            if name not in self.config.allow_tools or name not in self.tools:
                observation = f"Action '{name}' not allowed or not found."
                self.trajectory.append(Step(thought, action_line, observation, out))
                break

            # 4. Execute the action
            try:
                obs_payload = self.tools[name]["fn"](**args)
                observation = json.dumps(obs_payload, ensure_ascii=False)  # show structured obs
            except Exception as e:
                observation = f"Tool error: {e}"

            self.trajectory.append(Step(thought, action_line, observation, out))

            # If configured to stop after the first tool execution, break
            # the loop here so the agent performs exactly one search/tool
            # invocation and then returns.
            if getattr(self.config, "stop_after_first_tool", False):
                break

            # UNCOMMENT FOR LOGGING
            # If verbose, print the parsed thought/action and resulting observation now
            if self.config.verbose:
                print(f"Thought: {thought}")
                print(f"Action: {action_line}")
                print(f"Observation: {observation}\n")
        # Also expose the first observed search `results` (if any) at the
        # top-level return so callers can access it directly without a helper.
        results = None
        for s in self.trajectory:
            try:
                obj = json.loads(s.observation)
            except Exception:
                obj = None
            if isinstance(obj, dict) and "results" in obj:
                results = obj["results"]
                break

        return results
    