from types import SimpleNamespace

import torch
import math

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.teacher_utils import (approximate_entropy_from_topk_logprobs, build_opsd_teacher_data,
                                               generalized_jsd_loss, logp_surrogate_kl, teacher_student_kl_loss)


def _build_grpo_stub():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
    trainer.model = SimpleNamespace(training=True)
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)
    trainer.dynamic_num_samples = False
    trainer.num_generations = 2
    trainer.num_generations_eval = 2
    trainer.grade_alpha_piecewise_low = 0.25
    trainer.grade_alpha_piecewise_high = 0.75
    trainer.grade_alpha_sigmoid_temperature = 10.0
    trainer.grade_reward_norm = 'group_minmax'
    trainer.grade_reward_min = 0.0
    trainer.grade_reward_max = 1.0
    trainer.grade_reward_ema_decay = 0.9
    trainer.grade_reward_ema_eps = 1e-6
    trainer.grade_reward_ema_clip = 3.0
    trainer.grade_credit_scale = 1.0
    trainer.grade_credit_clip = 0.2
    trainer.grade_entropy_eps = 1e-6
    trainer._grade_reward_ema_mean = None
    trainer._grade_reward_ema_sq_mean = None
    return trainer


def test_grade_alpha_mapping_modes():
    trainer = _build_grpo_stub()
    reward_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    trainer.grade_alpha_mapping = 'linear'
    assert torch.allclose(trainer._map_grade_alpha(reward_norm), reward_norm)

    trainer.grade_alpha_mapping = 'piecewise'
    piecewise = trainer._map_grade_alpha(reward_norm)
    assert torch.allclose(piecewise, torch.tensor([0.0, 0.5, 1.0]))

    trainer.grade_alpha_mapping = 'sigmoid'
    sigmoid = trainer._map_grade_alpha(reward_norm)
    assert sigmoid[0] < 0.01
    assert torch.isclose(sigmoid[1], torch.tensor(0.5), atol=1e-6)
    assert sigmoid[2] > 0.99


def test_grade_reward_norm_group_minmax_and_zero_variance():
    trainer = _build_grpo_stub()
    trainer.grade_alpha_mapping = 'linear'

    rewards_per_func = torch.tensor([[1.0], [3.0], [5.0], [5.0]], dtype=torch.float32)

    trainer.grade_alpha_granularity = 'sample'
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}, {}, {}], rewards_per_func)
    assert norm_metrics == {}
    assert torch.allclose(gate_signal, rewards_per_func.view(-1))
    assert torch.allclose(reward_norm, torch.tensor([0.0, 1.0, 0.5, 0.5]))
    assert torch.allclose(alpha, reward_norm)

    trainer.grade_alpha_granularity = 'group'
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}, {}, {}], rewards_per_func)
    assert norm_metrics == {}
    assert torch.allclose(gate_signal, torch.tensor([2.0, 2.0, 5.0, 5.0]))
    assert torch.allclose(reward_norm, torch.tensor([0.0, 1.0, 0.5, 0.5]))
    assert torch.allclose(alpha, torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_grade_reward_norm_fixed_range_uses_absolute_sample_or_group_signal():
    trainer = _build_grpo_stub()
    trainer.grade_reward_norm = 'fixed_range'
    trainer.grade_alpha_mapping = 'linear'

    rewards_per_func = torch.tensor([[0.0], [1.0], [0.2], [0.2]], dtype=torch.float32)

    trainer.grade_alpha_granularity = 'sample'
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}, {}, {}], rewards_per_func)
    assert norm_metrics == {}
    assert torch.allclose(gate_signal, torch.tensor([0.0, 1.0, 0.2, 0.2]))
    assert torch.allclose(reward_norm, torch.tensor([0.0, 1.0, 0.2, 0.2]))
    assert torch.allclose(alpha, reward_norm)

    trainer.grade_alpha_granularity = 'group'
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}, {}, {}], rewards_per_func)
    assert norm_metrics == {}
    assert torch.allclose(gate_signal, torch.tensor([0.5, 0.5, 0.2, 0.2]))
    assert torch.allclose(reward_norm, torch.tensor([0.5, 0.5, 0.2, 0.2]))
    assert torch.allclose(alpha, reward_norm)


def test_grade_reward_norm_ema_uses_running_stats():
    trainer = _build_grpo_stub()
    trainer.grade_reward_norm = 'ema'
    trainer.grade_alpha_mapping = 'linear'
    trainer.grade_alpha_granularity = 'sample'

    first_rewards = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}], first_rewards)
    assert torch.allclose(gate_signal, torch.tensor([0.0, 1.0]))
    assert torch.allclose(reward_norm, torch.sigmoid(torch.tensor([-1.0, 1.0])), atol=1e-5)
    assert torch.allclose(alpha, reward_norm)
    assert math.isclose(norm_metrics['ema_mean'], 0.5, rel_tol=1e-5)
    assert math.isclose(norm_metrics['ema_std'], 0.5, rel_tol=1e-5)
    assert torch.isclose(trainer._grade_reward_ema_mean, torch.tensor(0.5), atol=1e-6)

    second_rewards = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    gate_signal, reward_norm, alpha, norm_metrics = trainer._compute_grade_reward_norm_and_alpha(
        [{}, {}], second_rewards)
    assert torch.allclose(gate_signal, torch.tensor([1.0, 1.0]))
    assert torch.allclose(reward_norm, torch.sigmoid(torch.tensor([1.0, 1.0])), atol=1e-5)
    assert torch.allclose(alpha, reward_norm)
    assert math.isclose(norm_metrics['ema_mean'], 0.5, rel_tol=1e-5)
    assert math.isclose(norm_metrics['ema_std'], 0.5, rel_tol=1e-5)
    assert torch.isclose(trainer._grade_reward_ema_mean, torch.tensor(0.55), atol=1e-6)


def test_grade_credit_formula_positive_and_negative_advantages():
    trainer = _build_grpo_stub()
    advantages = torch.tensor([2.0, -1.0], dtype=torch.float32)
    student_entropies = torch.tensor([[3.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    teacher_entropies = torch.tensor([[1.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    completion_mask = torch.tensor([[True, True], [True, False]])

    token_advantages, credit = trainer._compute_grade_token_advantages(
        advantages, student_entropies, teacher_entropies, completion_mask, vocab_size=16)

    expected_pos_credit = max(0.0, (3.0 - 1.0) / (3.0 + trainer.grade_entropy_eps))
    assert torch.isclose(credit[0, 0], torch.tensor(expected_pos_credit), atol=1e-6)
    assert torch.isclose(token_advantages[0, 0], advantages[0] * (1.0 + credit[0, 0]), atol=1e-6)

    h_max = torch.log(torch.tensor(16.0))
    expected_neg_credit = (h_max - 2.0) / h_max
    assert torch.isclose(credit[1, 0], expected_neg_credit, atol=1e-6)
    assert token_advantages[1, 1] == 0


def test_build_opsd_teacher_data_uses_teacher_prompt():
    inputs = [{
        'teacher_prompt': 'teacher question',
        'messages': [
            {
                'role': 'system',
                'content': 'sys'
            },
            {
                'role': 'user',
                'content': 'student question'
            },
            {
                'role': 'assistant',
                'content': 'student answer'
            },
        ]
    }]

    teacher_data = build_opsd_teacher_data(inputs)
    assert teacher_data is not None
    assert teacher_data[0]['messages'][-1]['role'] == 'user'
    assert teacher_data[0]['messages'][-1]['content'] == 'teacher question'


def test_api_topk_entropy_approximation_is_finite():
    logprobs = torch.log(torch.tensor([[[0.6, 0.3]], [[0.5, 0.4]]], dtype=torch.float32))
    entropy = approximate_entropy_from_topk_logprobs(logprobs)
    assert torch.isfinite(entropy).all()
    assert entropy.shape == torch.Size([2, 1])
    assert entropy[1, 0] > entropy[0, 0]


def test_generalized_jsd_loss_returns_zero_for_identical_logits_per_sequence():
    student_logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[0, 0]], dtype=torch.long)
    loss = generalized_jsd_loss(
        student_logits=student_logits,
        teacher_logits=student_logits.clone(),
        labels=labels,
        beta=0.5,
        return_per_sequence=True,
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_teacher_student_kl_loss_returns_zero_for_identical_logits():
    student_logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[0, 0]], dtype=torch.long)
    forward_loss = teacher_student_kl_loss(
        student_logits=student_logits,
        teacher_logits=student_logits.clone(),
        labels=labels,
        direction='forward_kl',
        return_per_sequence=True,
    )
    reverse_loss = teacher_student_kl_loss(
        student_logits=student_logits,
        teacher_logits=student_logits.clone(),
        labels=labels,
        direction='reverse_kl',
        return_per_sequence=True,
    )
    assert torch.allclose(forward_loss, torch.zeros_like(forward_loss), atol=1e-6)
    assert torch.allclose(reverse_loss, torch.zeros_like(reverse_loss), atol=1e-6)


def test_teacher_student_kl_loss_supports_topk_forward_and_reverse():
    student_logits = torch.tensor([[[2.0, 0.5, -1.0], [0.0, 1.0, -0.5]]], dtype=torch.float32)
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.3]]], dtype=torch.float32))
    teacher_topk_indices = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    labels = torch.tensor([[0, 0]], dtype=torch.long)

    forward_loss = teacher_student_kl_loss(
        student_logits=student_logits,
        teacher_topk_logprobs=teacher_topk_logprobs,
        teacher_topk_indices=teacher_topk_indices,
        labels=labels,
        direction='forward_kl',
        return_per_sequence=True,
    )
    reverse_loss = teacher_student_kl_loss(
        student_logits=student_logits,
        teacher_topk_logprobs=teacher_topk_logprobs,
        teacher_topk_indices=teacher_topk_indices,
        labels=labels,
        direction='reverse_kl',
        return_per_sequence=True,
    )
    assert forward_loss.shape == torch.Size([1])
    assert reverse_loss.shape == torch.Size([1])
    assert torch.isfinite(forward_loss).all()
    assert torch.isfinite(reverse_loss).all()


def test_logp_surrogate_kl_matches_original_sampled_token_formula():
    ref_logps = torch.tensor([[math.log(0.4), math.log(0.3)]], dtype=torch.float32)
    model_logps = torch.tensor([[math.log(0.5), math.log(0.2)]], dtype=torch.float32)

    result = logp_surrogate_kl(ref_logps, model_logps)
    expected = torch.exp(ref_logps - model_logps) - (ref_logps - model_logps) - 1
    assert torch.allclose(result, expected, atol=1e-6)


def test_logp_surrogate_kl_averages_sampled_and_extra_vocab_tokens():
    ref_logps = torch.tensor([[math.log(0.4)]], dtype=torch.float32)
    model_logps = torch.tensor([[math.log(0.5)]], dtype=torch.float32)
    extra_ref_logps = torch.tensor([[[math.log(0.2), math.log(0.1)]]], dtype=torch.float32)
    extra_model_logps = torch.tensor([[[math.log(0.1), math.log(0.05)]]], dtype=torch.float32)

    result = logp_surrogate_kl(
        ref_logps=ref_logps,
        model_logps=model_logps,
        extra_ref_logps=extra_ref_logps,
        extra_model_logps=extra_model_logps,
    )

    sampled = torch.exp(ref_logps - model_logps) - (ref_logps - model_logps) - 1
    extras = torch.exp(extra_ref_logps - extra_model_logps) - (extra_ref_logps - extra_model_logps) - 1
    expected = torch.cat([sampled.unsqueeze(-1), extras], dim=-1).mean(dim=-1)
    assert torch.allclose(result, expected, atol=1e-6)
