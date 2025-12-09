"""
Language model integration for ReAct agent.

Provides Hugging Face model loading and inference functions that generate
Thought and Action lines in the ReAct format.
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOAD_8BIT = False
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

try:
    import accelerate
    has_accelerate = True
except ImportError:
    has_accelerate = False

if torch.cuda.is_available() and has_accelerate:
    device_map = "auto"
else:
    device_map = None

model_kwargs = {
    "load_in_8bit": LOAD_8BIT,
    "dtype": DTYPE,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}
if device_map is not None:
    model_kwargs["device_map"] = device_map

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

if device_map is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

gen_cfg = GenerationConfig(
    top_k=50,
    num_beams=1,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

T_PATTERN = re.compile(r"Thought:\s*(.*)")


def _postprocess_to_two_lines(text: str) -> str:
    """
    Robustly extract exactly two lines: Thought and Action.
    - Truncate at the first 'Observation:' if present.
    - Find the first 'Thought:' line and the first bracketed Action (non-greedy).
    - If the model echoes prompt content (Observations, system preamble, role tokens), ignore that and return safe defaults.
    """
    # Defensive truncation at Observation
    text = (text or "").split("\nObservation:", 1)[0].strip()

    # Extract Thought (first 'Thought:' line)
    thought = None
    m_t = re.search(r"Thought:\s*(.*)", text)
    if m_t:
        thought = m_t.group(1).splitlines()[0].strip()

    # Extract Action: capture NAME[...] non-greedy so trailing tokens are not included.
    a_match = re.search(r"Action:\s*([a-z_]+\[.*?\])", text, flags=re.IGNORECASE | re.DOTALL)
    action = a_match.group(1).strip() if a_match else None

    # Fallbacks if model failed to follow format
    if not thought:
        thought = "I should search for key facts related to the question."
    if not action:
        action = 'semantic_search[query="(auto) refine the user question", k=3]'

    return f"Thought: {thought}\nAction: {action}"


def hf_llm(prompt: str) -> str:
    """
    Completes from your existing ReAct prompt and returns exactly two lines:
    'Thought: ...' and 'Action: ...'
    """
    format_guard = (
        "\n\nIMPORTANT: Respond with EXACTLY two lines in this format:\n"
        "Thought: <one concise sentence>\n"
        "Action: <either keyword_search[query=\"<text>\", bucketLevel=<bucketLevel>, subject=\"<subject code>\"] or semantic_search[query=\"<text>\"] or finish[answer=\"...\"]>\n"
        "Do NOT include Observation."
    )
    full_prompt = prompt + format_guard

    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    bad_words = ["Observation:", "Observation", "Human:", "Human", "Assistant:", "Assistant"]
    try:
        bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_words]
    except Exception:
        bad_words_ids = None

    with torch.no_grad():
        gen_kwargs = dict(
            generation_config=gen_cfg,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.0,  # Deterministic decoding
            top_p=0.95,
        )
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        output_ids = model.generate(**inputs, **gen_kwargs)

    completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _postprocess_to_two_lines(completion)


LLM = hf_llm