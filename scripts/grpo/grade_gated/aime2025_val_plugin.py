"""AIME 2025 validation dataset plugin for GRPO/DAPO math evaluation."""

from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset


class AIME2025ValPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = row.get('question', '')
        answer = row.get('answer', '')
        if not question or answer is None:
            return None
        messages: List[Dict[str, str]] = [{
            'role': 'user',
            'content': question,
        }]
        return {
            'messages': messages,
            'solution': str(answer).strip(),
        }


register_dataset(
    DatasetMeta(
        dataset_path='/ytech_m2v5_hdd/workspace/kling_mm/Datasets/AIME2025/aime2025-I.jsonl',
        dataset_name='aime2025_i_val',
        preprocess_func=AIME2025ValPreprocessor(),
        tags=['math', 'eval'],
    ))

register_dataset(
    DatasetMeta(
        dataset_path='/ytech_m2v5_hdd/workspace/kling_mm/Datasets/AIME2025/aime2025-II.jsonl',
        dataset_name='aime2025_ii_val',
        preprocess_func=AIME2025ValPreprocessor(),
        tags=['math', 'eval'],
    ))
