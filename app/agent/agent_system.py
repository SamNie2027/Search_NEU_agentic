"""
ReAct (Reasoning and Acting) Agent System.

Implements a ReAct agent that uses a language model to generate thoughts and actions,
executes tools (search functions), and iteratively refines its approach based on observations.
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json
import re
import logging
import traceback
import inspect
from .prompting_techniques import make_prompt, parse_action


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
        """Run the agent with the given query.
        
        Args:
            user_query: The user's search query
            useFilters: Whether to allow filter parameters in tool calls
            
        Returns:
            Dictionary with search results, or None if no results found
        """
        self.trajectory.clear()
        self._recent_action_counts.clear()
        print("Starting ReActAgent.run (useFilters=%s, max_steps=%s)", useFilters, self.config.max_steps)
        for step_idx in range(self.config.max_steps):
            # Format prompt with current trajectory
            prompt = make_prompt(user_query, [asdict(s) for s in self.trajectory], useFilters=useFilters)

            # Get LLM response
            try:
                out = self.llm(prompt)
            except Exception as e:
                out = f"Thought: (llm error)\nAction: finish[answer=\"LLM error: {e}\"]"
                print("LLM call raised an exception: %s", e)

            # Log intermediate output if verbose
            if self.config.verbose:
                print("--- Intermediate output (step %s) ---", step_idx + 1)
                print("LLM output:\n%s", out)

            # Parse Thought and Action from LLM output
            t_match = re.search(r"Thought:\s*(.*)", out)
            a_match = re.search(r"Action:\s*(.*)", out)
            thought = t_match.group(1).strip() if t_match else "(no thought)"
            action_line = a_match.group(1).strip() if a_match else "finish[answer=\"(no action)\"]"
            
            # Remove duplicate 'Action:' prefix if present
            if re.match(r"^\s*Action:\s*", action_line, flags=re.IGNORECASE):
                action_line = re.sub(r"^\s*Action:\s*", "", action_line, flags=re.IGNORECASE)
            action_line = "Action: " + action_line

            # Parse action into name and arguments
            parsed = parse_action(action_line)

            if not parsed:
                observation = "Invalid action format. Stopping."
                print("Failed to parse action. action_line=%s LLM_output=%s", action_line, out)
                self.trajectory.append(Step(thought, action_line, observation, out))
                break
            name, args = parsed

            print("Parsed action: %s", name)
            print("Action args: %s", args)

            if name == "finish":
                observation = "done"
                self.trajectory.append(Step(thought, action_line, observation, out))
                break

            if name not in self.config.allow_tools or name not in self.tools:
                observation = f"Action '{name}' not allowed or not found."
                self.trajectory.append(Step(thought, action_line, observation, out))
                break

            # Execute the action
            # Remove filter arguments if filters are disabled
            if not useFilters:
                disallowed_keys = {"bucketLevel", "subject", "subjectCode", "subject_code"}
                present = disallowed_keys.intersection(args.keys())
                if present:
                    for k in present:
                        args.pop(k, None)
                    if self.config.verbose:
                        print("[agent_system] Stripped disallowed filter args from action '%s': %s", name, sorted(list(present)))

            try:
                print("Executing tool '%s' with args=%s", name, args)
                # Introspect the tool function signature and remove any args
                # that the function does not accept. This prevents TypeError
                # when the LLM emits unexpected parameters.
                func = self.tools[name]["fn"]
                try:
                    sig = inspect.signature(func)
                except Exception:
                    sig = None

                if sig is not None:
                    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if not accepts_var_kw:
                        allowed = set(sig.parameters.keys())
                        filtered_args = {k: v for k, v in args.items() if k in allowed}
                        removed = set(args.keys()) - set(filtered_args.keys())
                        if removed:
                            print("Removed unsupported args for tool '%s': %s", name, sorted(list(removed)))
                    else:
                        filtered_args = args
                else:
                    filtered_args = args

                obs_payload = func(**filtered_args)
                observation = json.dumps(obs_payload, ensure_ascii=False)  # show structured obs
                # Log summary of payload
                if isinstance(obs_payload, dict):
                    print("Tool '%s' returned keys=%s", name, list(obs_payload.keys()))
                else:
                    print("Tool '%s' returned non-dict payload of type %s", name, type(obs_payload))
            except Exception as e:
                observation = f"Tool error: {e}"
                print("Tool '%s' raised an exception while executing with args=%s\n%s", name, args, traceback.format_exc())

            self.trajectory.append(Step(thought, action_line, observation, out))

            # Stop after first tool if configured
            if getattr(self.config, "stop_after_first_tool", False):
                break

            # UNCOMMENT FOR LOGGING
            # If verbose, print the parsed thought/action and resulting observation now
            if self.config.verbose:
                print("Thought: %s", thought)
                print("Action: %s", action_line)
                print("Observation: %s", observation)
        
        # Extract and return the first search results found in trajectory
        results = None
        for s in self.trajectory:
            try:
                obj = json.loads(s.observation)
            except Exception:
                obj = None
            if isinstance(obj, dict) and "results" in obj:
                results = obj["results"]
                break

        print("Agent run completed. Found results=%s", bool(results))
        return results
    