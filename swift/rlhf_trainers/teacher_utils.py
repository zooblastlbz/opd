# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


@dataclass
class TeacherOutput:
    """Unified container for teacher model outputs."""

    full_logits: Optional[torch.Tensor] = None
    topk_logprobs: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    opsd_teacher_labels: Optional[torch.Tensor] = None

    @property
    def is_topk_mode(self) -> bool:
        return self.topk_logprobs is not None and self.topk_indices is not None

    def validate(self):
        if self.full_logits is None and not self.is_topk_mode:
            raise ValueError('TeacherOutput must provide either full_logits or '
                             '(topk_logprobs, topk_indices). Got neither.')


def build_opsd_teacher_data(inputs):
    """Build OPSD teacher data by replacing the last user message with teacher_prompt."""
    if not all('teacher_prompt' in data and data['teacher_prompt'] for data in inputs):
        return None

    teacher_data = []
    for data in inputs:
        teacher_item = {k: v for k, v in data.items() if k != 'teacher_prompt'}
        messages = [dict(m) for m in data.get('messages', [])]
        if messages and messages[-1]['role'] == 'assistant':
            messages.pop()
        for msg in reversed(messages):
            if msg['role'] == 'user':
                msg['content'] = data['teacher_prompt']
                break
        teacher_item['messages'] = messages
        teacher_data.append(teacher_item)
    return teacher_data


def align_vocab_size(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    """Align vocab dimensions between student and teacher by padding the smaller one."""
    stu_vocab = student_logits.shape[-1]
    tea_vocab = teacher_logits.shape[-1]
    if stu_vocab == tea_vocab:
        return student_logits, teacher_logits
    if stu_vocab < tea_vocab:
        student_logits = F.pad(student_logits, (0, tea_vocab - stu_vocab), 'constant', 0)
        student_logits[..., stu_vocab:] = teacher_logits[..., stu_vocab:]
    else:
        teacher_logits = F.pad(teacher_logits, (0, stu_vocab - tea_vocab), 'constant', 0)
        teacher_logits[..., tea_vocab:] = student_logits[..., tea_vocab:]
    return student_logits, teacher_logits


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    chunk_size: int = 512,
    topk: Optional[int] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    return_per_sequence: bool = False,
):
    """Compute generalized JSD loss, optionally reduced per sequence."""
    if teacher_logits is not None:
        student_logits, teacher_logits = align_vocab_size(student_logits, teacher_logits)

    if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
        student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
        teacher_logits = teacher_topk_logprobs
    elif topk is not None and teacher_logits is not None:
        teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)

    if teacher_logits is None:
        raise ValueError('teacher_logits must be provided directly or via teacher_topk_logprobs.')

    batch_size = student_logits.shape[0] if student_logits.ndim > 2 else 1
    if labels is not None:
        mask = labels != -100
        seq_ids = None
        if return_per_sequence:
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand_as(mask)[mask]
            counts = mask.sum(dim=1)
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        num_valid = mask.sum()
    else:
        if return_per_sequence:
            seq_len = student_logits.shape[1]
            counts = torch.full((batch_size,), seq_len, dtype=torch.long, device=student_logits.device)
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand(batch_size, seq_len)
            seq_ids = seq_ids.reshape(-1)
        else:
            seq_ids = None
            counts = None
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        num_valid = student_logits.size(0)

    if isinstance(num_valid, torch.Tensor):
        num_valid_int = num_valid.item()
    else:
        num_valid_int = num_valid

    if num_valid_int == 0:
        if return_per_sequence:
            return student_logits.new_zeros(batch_size)
        return student_logits.new_zeros(())

    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    if return_per_sequence:
        total_loss = student_logits.new_zeros(batch_size)
    else:
        total_loss = student_logits.new_zeros(())

    for start_idx in range(0, num_valid_int, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid_int)
        s_chunk = student_logits[start_idx:end_idx] / temperature
        t_chunk = teacher_logits[start_idx:end_idx] / temperature

        s_log_probs = F.log_softmax(s_chunk, dim=-1)
        t_log_probs = F.log_softmax(t_chunk, dim=-1)

        if beta == 0:
            jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        elif beta == 1:
            jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student

        token_loss = jsd_chunk.sum(dim=-1)
        if return_per_sequence:
            total_loss.index_add_(0, seq_ids[start_idx:end_idx], token_loss)
        else:
            total_loss = total_loss + token_loss.sum()

    if return_per_sequence:
        counts = counts.clamp(min=1)
        result = total_loss / counts
        result = torch.where(counts > 0, result, torch.zeros_like(result))
        return result
    return total_loss / num_valid_int


def _reduce_token_losses(token_loss: torch.Tensor,
                         batch_size: int,
                         return_per_sequence: bool,
                         seq_ids: Optional[torch.Tensor] = None,
                         counts: Optional[torch.Tensor] = None):
    if return_per_sequence:
        result = token_loss.new_zeros(batch_size)
        result.index_add_(0, seq_ids, token_loss)
        counts = counts.clamp(min=1)
        result = result / counts
        result = torch.where(counts > 0, result, torch.zeros_like(result))
        return result
    return token_loss.mean() if token_loss.numel() > 0 else token_loss.new_zeros(())


def logp_surrogate_kl(
    ref_logps: torch.Tensor,
    model_logps: torch.Tensor,
    extra_ref_logps: Optional[torch.Tensor] = None,
    extra_model_logps: Optional[torch.Tensor] = None,
    ratio_clip: float = 20.0,
    loss_clip: float = 10.0,
) -> torch.Tensor:
    """Compute the sampled-token KL surrogate, optionally averaged with extra vocab tokens."""

    safe_ratio = torch.clamp(ref_logps - model_logps, min=-ratio_clip, max=ratio_clip)
    base_loss = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-loss_clip, max=loss_clip)

    if extra_ref_logps is None or extra_model_logps is None:
        return base_loss
    if extra_ref_logps.shape != extra_model_logps.shape:
        raise ValueError('extra_ref_logps and extra_model_logps must have the same shape.')
    if extra_ref_logps.numel() == 0 or extra_ref_logps.shape[-1] == 0:
        return base_loss

    extra_safe_ratio = torch.clamp(extra_ref_logps - extra_model_logps, min=-ratio_clip, max=ratio_clip)
    extra_loss = torch.clamp(torch.exp(extra_safe_ratio) - extra_safe_ratio - 1, min=-loss_clip, max=loss_clip)
    return torch.cat([base_loss.unsqueeze(-1), extra_loss], dim=-1).mean(dim=-1)


def _kl_from_log_probs(source_log_probs: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    source_probs = source_log_probs.exp()
    return (source_probs * (source_log_probs - target_log_probs)).sum(dim=-1)


def _compressed_log_probs_from_topk(topk_log_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    finite_mask = torch.isfinite(topk_log_probs)
    topk_probs = torch.where(finite_mask, topk_log_probs.exp(), torch.zeros_like(topk_log_probs))
    other_prob = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)
    compressed_probs = torch.cat([topk_probs, other_prob], dim=-1)
    compressed_probs = compressed_probs / compressed_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
    return compressed_probs.clamp(min=eps).log()


def _kl_loss_from_topk_teacher(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    temperature: float,
    direction: Literal['forward_kl', 'reverse_kl'],
    chunk_size: int,
    return_per_sequence: bool,
    labels: Optional[torch.Tensor] = None,
):
    batch_size = student_logits.shape[0]
    if labels is not None:
        mask = labels != -100
        seq_ids = None
        if return_per_sequence:
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand_as(mask)[mask]
            counts = mask.sum(dim=1)
        else:
            counts = None
        student_logits = student_logits[mask]
        teacher_topk_logprobs = teacher_topk_logprobs[mask]
        teacher_topk_indices = teacher_topk_indices[mask]
    else:
        if return_per_sequence:
            seq_len = student_logits.shape[1]
            counts = torch.full((batch_size,), seq_len, dtype=torch.long, device=student_logits.device)
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand(batch_size, seq_len)
            seq_ids = seq_ids.reshape(-1)
        else:
            counts = None
            seq_ids = None
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_topk_logprobs = teacher_topk_logprobs.view(-1, teacher_topk_logprobs.size(-1))
        teacher_topk_indices = teacher_topk_indices.view(-1, teacher_topk_indices.size(-1))

    if student_logits.numel() == 0:
        if return_per_sequence:
            return student_logits.new_zeros(batch_size)
        return student_logits.new_zeros(())

    token_losses = []
    num_valid = student_logits.shape[0]
    for start_idx in range(0, num_valid, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid)
        student_log_probs = F.log_softmax(student_logits[start_idx:end_idx] / temperature, dim=-1)
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_indices[start_idx:end_idx])
        student_compressed = _compressed_log_probs_from_topk(student_topk_log_probs)

        teacher_compressed = _compressed_log_probs_from_topk(teacher_topk_logprobs[start_idx:end_idx])
        if temperature != 1.0:
            teacher_compressed = F.log_softmax(teacher_compressed / temperature, dim=-1)

        if direction == 'forward_kl':
            chunk_loss = _kl_from_log_probs(teacher_compressed, student_compressed)
        else:
            chunk_loss = _kl_from_log_probs(student_compressed, teacher_compressed)
        token_losses.append(chunk_loss)

    token_loss = torch.cat(token_losses, dim=0)
    return _reduce_token_losses(token_loss, batch_size, return_per_sequence, seq_ids, counts)


def teacher_student_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    chunk_size: int = 512,
    topk: Optional[int] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    direction: Literal['forward_kl', 'reverse_kl'] = 'forward_kl',
    return_per_sequence: bool = False,
):
    """Compute teacher-student KL with configurable direction.

    - `forward_kl`: KL(teacher || student)
    - `reverse_kl`: KL(student || teacher)
    """
    if direction not in {'forward_kl', 'reverse_kl'}:
        raise ValueError(f'Unknown KL direction: {direction}')

    if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
        return _kl_loss_from_topk_teacher(
            student_logits=student_logits,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
            temperature=temperature,
            direction=direction,
            chunk_size=chunk_size,
            return_per_sequence=return_per_sequence,
            labels=labels,
        )

    if teacher_logits is None:
        raise ValueError('teacher_logits must be provided directly or via teacher_topk_logprobs.')

    student_logits, teacher_logits = align_vocab_size(student_logits, teacher_logits)
    if topk is not None:
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_topk_logprobs, teacher_topk_indices = torch.topk(teacher_log_probs, k=topk, dim=-1)
        return _kl_loss_from_topk_teacher(
            student_logits=student_logits,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
            temperature=temperature,
            direction=direction,
            chunk_size=chunk_size,
            return_per_sequence=return_per_sequence,
            labels=labels,
        )

    batch_size = student_logits.shape[0] if student_logits.ndim > 2 else 1
    if labels is not None:
        mask = labels != -100
        seq_ids = None
        if return_per_sequence:
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand_as(mask)[mask]
            counts = mask.sum(dim=1)
        else:
            counts = None
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
    else:
        if return_per_sequence:
            seq_len = student_logits.shape[1]
            counts = torch.full((batch_size,), seq_len, dtype=torch.long, device=student_logits.device)
            seq_ids = torch.arange(batch_size, device=student_logits.device).unsqueeze(1).expand(batch_size, seq_len)
            seq_ids = seq_ids.reshape(-1)
        else:
            counts = None
            seq_ids = None
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

    if student_logits.numel() == 0:
        if return_per_sequence:
            return student_logits.new_zeros(batch_size)
        return student_logits.new_zeros(())

    token_losses = []
    num_valid = student_logits.shape[0]
    for start_idx in range(0, num_valid, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid)
        student_log_probs = F.log_softmax(student_logits[start_idx:end_idx] / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits[start_idx:end_idx] / temperature, dim=-1)
        if direction == 'forward_kl':
            chunk_loss = _kl_from_log_probs(teacher_log_probs, student_log_probs)
        else:
            chunk_loss = _kl_from_log_probs(student_log_probs, teacher_log_probs)
        token_losses.append(chunk_loss)

    token_loss = torch.cat(token_losses, dim=0)
    return _reduce_token_losses(token_loss, batch_size, return_per_sequence, seq_ids, counts)


def approximate_entropy_from_topk_logprobs(topk_logprobs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Approximate entropy from top-k logprobs plus a single remainder bucket."""
    finite_mask = torch.isfinite(topk_logprobs)
    probs = torch.where(finite_mask, topk_logprobs.exp(), torch.zeros_like(topk_logprobs))
    logprobs = torch.where(finite_mask, topk_logprobs, torch.zeros_like(topk_logprobs))
    entropy = -(probs * logprobs).sum(dim=-1)
    remainder = (1.0 - probs.sum(dim=-1)).clamp(min=0.0, max=1.0)
    remainder_term = torch.where(
        remainder > 0,
        -remainder * remainder.clamp(min=eps).log(),
        torch.zeros_like(remainder),
    )
    return entropy + remainder_term


def _build_teacher_session(max_retries=5):
    """Build a requests.Session with transport-level retry for teacher API calls."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=3,
        status_forcelist=[500, 502, 503],
        backoff_factor=2,
        allowed_methods=['POST', 'GET'],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


_teacher_session = None
teacher_model_server_model_name = None


def fetch_teacher_logprobs(base_url, input_ids, topk=20, timeout=300.0):
    """Fetch top-k prompt logprobs from a vLLM-compatible /v1/completions endpoint."""
    import logging
    from concurrent.futures import ThreadPoolExecutor

    global _teacher_session
    if _teacher_session is None:
        _teacher_session = _build_teacher_session()
    session = _teacher_session

    _logger = logging.getLogger(__name__)
    base_url = base_url.rstrip('/')
    batch_size = len(input_ids)
    max_seq_len = max(len(ids) for ids in input_ids)
    url = f'{base_url}/v1/completions'
    global teacher_model_server_model_name
    if teacher_model_server_model_name is None:
        try:
            resp = session.get(f'{base_url}/v1/models', timeout=10)
            model = resp.json()['data'][0]['id'] if resp.ok else 'default'
        except Exception:
            model = 'default'
        teacher_model_server_model_name = model
    else:
        model = teacher_model_server_model_name

    out_len = max_seq_len - 1
    logprobs_out = torch.full((batch_size, out_len, topk), float('-inf'), dtype=torch.float32)
    indices_out = torch.zeros((batch_size, out_len, topk), dtype=torch.long)
    errors = {}

    def _fetch_one(batch_idx):
        payload = {
            'model': model,
            'prompt': input_ids[batch_idx],
            'max_tokens': 1,
            'temperature': 0,
            'prompt_logprobs': topk,
        }
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            prompt_logprobs_list = resp.json()['choices'][0].get('prompt_logprobs', [])
            for raw_pos in range(1, len(prompt_logprobs_list)):
                pos_lp = prompt_logprobs_list[raw_pos]
                if pos_lp is None:
                    continue
                out_pos = raw_pos - 1
                if out_pos >= out_len:
                    break
                sorted_items = sorted(pos_lp.items(), key=lambda x: -x[1]['logprob'])[:topk]
                for k, (tid_str, info) in enumerate(sorted_items):
                    indices_out[batch_idx, out_pos, k] = int(tid_str)
                    logprobs_out[batch_idx, out_pos, k] = info['logprob']
        except Exception as e:
            errors[batch_idx] = e
            _logger.error(f'Failed to get teacher logprobs for sequence {batch_idx}: {e}')

    with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
        list(pool.map(_fetch_one, range(batch_size)))

    if errors:
        failed = sorted(errors.keys())
        raise RuntimeError(f'Failed to fetch teacher logprobs for {len(errors)} sequence(s). '
                           f'Failed indices: {failed}. Last errors: ' + '; '.join(f'seq {i}: {errors[i]}'
                                                                                  for i in failed[:3]))

    return logprobs_out, indices_out
