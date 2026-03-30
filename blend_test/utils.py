"""
for attn-only single-sample inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


TASK_METRICS = {
    "hotpotqa": "rouge",
    "2wikimqa": "rouge",
    "musique": "rouge",
}


MAX_NEW_TOKENS = {
    "hotpotqa": 48,
    "2wikimqa": 48,
    "musique": 48,
}


DATASET_PROMPTS = {
    "hotpotqa": (
        "You are a highly precise question-answering assistant.\n\n## Task\nRead the provided passages and answer the user's question based strictly on the information within.\n\n## Output Rules\n- Direct Answer ONLY: Output nothing but the final exact answer.\n- No Explanations: Do not provide reasoning, context, conversational fillers, or extra words.\n\n## Passages\n",
        "Remember to answer the question based strictly on the passages above. Output ONLY the answer and no other words.\n\n## Question\n",
    ),
    "2wikimqa": (
        "You are a highly precise question-answering assistant.\n\n## Task\nRead the provided passages and answer the user's question based strictly on the information within.\n\n## Output Rules\n- Direct Answer ONLY: Output nothing but the final exact answer.\n- No Explanations: Do not provide reasoning, context, conversational fillers, or extra words.\n\n## Passages\n",
        "Remember to answer the question based strictly on the passages above. Output ONLY the answer and no other words.\n\n## Question\n",
    ),
    "musique": (
        "You are a highly precise question-answering assistant.\n\n## Task\nRead the provided passages and answer the user's question based strictly on the information within.\n\n## Output Rules\n- Direct Answer ONLY: Output nothing but the final exact answer.\n- No Explanations: Do not provide reasoning, context, conversational fillers, or extra words.\n\n## Passages\n",
        "Remember to answer the question based strictly on the passages above. Output ONLY the answer and no other words.\n\n## Question\n",
    ),
}


def load_dataset(dataset_path: str) -> List[Dict]:
    data = []
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def normalize_question(question: str) -> str:
    if not question:
        return ""
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:]


def load_split_dataset(dataset_path: str) -> List[Dict]:
    return load_dataset(dataset_path)


def build_prompt_for_dataset(
    example: Dict, dataset_name: str, topk: int = 10
) -> Tuple[List[str], List[str]]:
    context = example.get("context", "")
    if topk > 0:
        nums_docs = min(len(context), topk)
        ctxs = context[:nums_docs]
    else:
        ctxs = context

    doc_prompts = [f"Passage:\n{ctx}\n\n" for ctx in ctxs]
    input_text = normalize_question(example.get("input", ""))

    config = DATASET_PROMPTS.get(dataset_name)
    if config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    q_prompt = [config[1], input_text]
    return doc_prompts, q_prompt


def get_system_prompt(dataset_name: str) -> str:
    config = DATASET_PROMPTS.get(dataset_name)
    if config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return config[0]


def get_max_new_tokens(dataset_name: str) -> int:
    return MAX_NEW_TOKENS.get(dataset_name, 48)
