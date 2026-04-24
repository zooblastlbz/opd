"""OPD repo style math reward for teacher GRPO.

This matches the behavior of thunlp/OPD's ttrl_math reward at the level that
matters for ms-swift training:
- the model answer must be extractable from the last \boxed{...};
- missing boxed answer receives 0;
- reward is 1 only when the boxed answer verifies against the reference.
"""

import re
from typing import List, Optional

from swift.rewards import ORM, orms


def extract_last_boxed(text: str) -> Optional[str]:
    marker = r'\boxed{'
    start = text.rfind(marker)
    if start < 0:
        return None

    idx = start + len(marker)
    depth = 1
    chars = []
    while idx < len(text):
        char = text[idx]
        if char == '{':
            depth += 1
            chars.append(char)
        elif char == '}':
            depth -= 1
            if depth == 0:
                return ''.join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        idx += 1
    return None


def boxed_or_text(text: str) -> str:
    boxed = extract_last_boxed(text)
    return boxed if boxed is not None else text


class OPDTTRLMATHAccuracy(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from math_verify import parse, verify

        rewards = []
        for completion, gold in zip(completions, solution):
            pred = extract_last_boxed(completion)
            if pred is None:
                rewards.append(0.0)
                continue

            gold_answer = boxed_or_text(gold)
            try:
                pred_parsed = parse(pred, extraction_mode='first_match')
                gold_parsed = parse(gold_answer, extraction_mode='first_match')
                reward = float(verify(gold_parsed, pred_parsed)) if gold_parsed and pred_parsed else 0.0
            except Exception:
                pred_norm = re.sub(r'\s+', '', pred)
                gold_norm = re.sub(r'\s+', '', gold_answer)
                reward = 1.0 if pred_norm and pred_norm == gold_norm else 0.0
            rewards.append(reward)
        return rewards


orms['opd_ttrl_math_accuracy'] = OPDTTRLMATHAccuracy
