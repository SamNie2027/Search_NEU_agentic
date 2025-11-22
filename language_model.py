# 1. We will load a language model model from huggingface (Qwen 0.5B Instruct)
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"    # swap if you prefer another instruct model
LOAD_8BIT    = False                           # set True if you installed bitsandbytes and want 8-bit loading
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ====== Model load ======
# Load the model from Hugging Face with reasonable defaults for device and dtype.
# Use 8-bit loading if requested, and map device automatically when CUDA is available.
device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=LOAD_8BIT,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map=device_map,
    low_cpu_mem_usage=True,
)

# Generation configuration: tune defaults for concise, deterministic ReAct-style replies
gen_cfg = GenerationConfig(
    temperature=0.0,
    top_p=0.95,
    top_k=50,
    num_beams=1,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

# ====== Helper function: Enforce two-line schema in the decoding ======
T_PATTERN = re.compile(r"Thought:\s*(.+)")
A_PATTERN = re.compile(r"Action:\s*(.+)")

def _postprocess_to_two_lines(text: str) -> str:
    """
    Extract the first 'Thought:' and 'Action:' lines from the model output.
    If the model drifts, fall back to a conservative default Action.
    """
    # Stop at first Observation if present (model shouldn't produce it, but just in case)
    text = text.split("\nObservation:")[0]
    # Keep only the assistant's new tokens (strip any trailing commentary)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Try to find explicit Thought/Action anywhere in the output
    thought = None
    action  = None
    for ln in lines:
        if thought is None:
            m = T_PATTERN.match(ln)
            if m:
                thought = m.group(1).strip()
                continue
        if action is None:
            m = A_PATTERN.match(ln)
            if m:
                action = m.group(1).strip()
                continue

    # Fallbacks if the model didn’t comply perfectly
    if thought is None:
        thought = "I should search for key facts related to the question."
    if action is None:
        # Default to a generic search; your controller will parse it.
        action = 'search[query="(auto) refine the user question", k=3]'

    return f"Thought: {thought}\nAction: {action}"
# ====== Helper function: Enforce two-line schema in the decoding ======



# 2. We define the LLM function. This will be plugged into the agent without changing the controller ---
def hf_llm(prompt: str) -> str:
    """
    Completes from your existing ReAct prompt and returns exactly two lines:
    'Thought: ...' and 'Action: ...'
    """
    # We add a strong instruction to the prompt to improve compliance with the format
    format_guard = (
        "\n\nIMPORTANT: Respond with EXACTLY two lines in this format:\n"
        "Thought: <one concise sentence>\n"
        "Action: <either keyword_search[query=\"<text>\", bucketLevel?=<bucketLevel>, subject?=\"<subject code>\"] or semantic_search[query=\"<text>\"] or finish[answer=\"...\"]>\n"
        "Do NOT include Observation."
    )
    full_prompt = prompt + format_guard

    # Tokenize the prompt and move tensors to the model's device
    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate model output. Use the shared `gen_cfg` for generation defaults.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_cfg,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    # Slice off the prompt tokens to get only the completion
    completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _postprocess_to_two_lines(completion)

# We will wire it into the agent system
LLM = hf_llm