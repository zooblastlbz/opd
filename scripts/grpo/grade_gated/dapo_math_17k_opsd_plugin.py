"""OPSD dataset plugin for open-r1/DAPO-Math-17k-Processed.

Prepares the dataset for privileged-info self-distillation:
- Student sees only the problem.
- Teacher sees the problem + reference solution via teacher_prompt.
- The gold `solution` field is preserved for reward computation.
"""
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

SYSTEM_PROMPT = 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.'

TRANSITION_PROMPT = ('After understanding the reference solution and the rationale behind each step, '
                     'now articulate your own step-by-step reasoning that derives the final answer.')


class DAPOMath17kOPSDPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not row.get('correct', True):
            return None

        problem = row.get('problem', '')
        solution = row.get('solution', '')

        teacher_prompt = (f'{problem}\n\n'
                          f'Here is a reference solution to this problem:\n{solution}\n\n'
                          f'{TRANSITION_PROMPT}')

        messages: List[Dict[str, str]] = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': problem
            },
        ]

        return {
            'messages': messages,
            'teacher_prompt': teacher_prompt,
            'solution': solution,
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='open-r1/DAPO-Math-17k-Processed',
        hf_dataset_id='open-r1/DAPO-Math-17k-Processed',
        dataset_name='dapo_math_17k_opsd',
        preprocess_func=DAPOMath17kOPSDPreprocessor(),
        tags=['math', 'opsd'],
    ))
