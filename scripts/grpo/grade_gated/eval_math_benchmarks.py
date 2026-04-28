#!/usr/bin/env python3
"""Run single-model OPD-style math evaluation with ms-swift/EvalScope."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


DEFAULT_DATASETS = ["aime24", "aime25", "math_500", "amc"]
DEFAULT_DATASET_ARGS = {"amc": {"subset_list": ["amc23"]}}


def str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF/modelscope model id or local checkpoint path.")
    parser.add_argument(
        "--adapters",
        nargs="*",
        default=None,
        help="Optional LoRA adapter checkpoint(s). For full-parameter checkpoints, leave this unset.",
    )
    parser.add_argument("--eval-url", default=None, help="Existing OpenAI-compatible service URL.")
    parser.add_argument("--port", type=int, default=8000, help="Local Swift/vLLM deployment start port.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="EvalScope Native dataset names.")
    parser.add_argument(
        "--dataset-args",
        default=json.dumps(DEFAULT_DATASET_ARGS),
        help="JSON dataset_args passed to EvalScope. Default selects AMC subset amc23.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for report.json and summary.json.")
    parser.add_argument("--repeats", type=int, default=16, help="Number of sampled solutions per problem.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the EvalScope task.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-enable-prefix-caching", type=str_to_bool, default=True)
    parser.add_argument("--eval-num-proc", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--infer-backend", default="vllm", choices=["vllm", "transformers", "sglang", "lmdeploy"])
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--system", default=None, help="Optional system prompt override.")
    parser.add_argument("--enable-thinking", default="false", choices=["true", "false"])
    parser.add_argument(
        "--extra-eval-args",
        default='{"ignore_errors": false}',
        help="JSON extra_eval_args passed to EvalScope Native backend.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without launching evaluation.")
    return parser.parse_args()


def json_arg(value: Optional[str], default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if value is None or value == "":
        return dict(default or {})
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object, got: {type(parsed).__name__}")
    return parsed


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    model_name = Path(args.model.rstrip("/")).name or "model"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("eval_output") / "opd_math" / f"{model_name}-{stamp}"


def flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    values: Dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            values.update(flatten_numeric(value, child))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            values.update(flatten_numeric(value, child))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        values[prefix] = float(obj)
    return values


def summarize(report: Mapping[str, Any], backend: str, repeats: int) -> Dict[str, Any]:
    return {
        key: {
            "mean": value,
            "num_repeats": repeats,
            "values": [value],
        }
        for key, value in sorted(flatten_numeric(report.get(backend, {})).items())
    }


def run_eval(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    from swift import EvalArguments, eval_main

    dataset_args = json_arg(args.dataset_args, DEFAULT_DATASET_ARGS)
    extra_eval_args = json_arg(args.extra_eval_args)
    extra_eval_args["repeats"] = args.repeats
    extra_eval_args["seed"] = args.seed
    generation_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k is not None:
        generation_config["top_k"] = args.top_k

    eval_args = EvalArguments(
        model=args.model,
        adapters=args.adapters or [],
        eval_url=args.eval_url,
        port=args.port,
        eval_dataset=args.datasets,
        eval_dataset_args=dataset_args,
        eval_generation_config=generation_config,
        extra_eval_args=extra_eval_args,
        eval_output_dir=str(output_dir / "eval"),
        eval_backend="Native",
        infer_backend=args.infer_backend,
        eval_limit=args.eval_limit,
        eval_num_proc=args.eval_num_proc,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        temperature=args.temperature,
        system=args.system,
        enable_thinking=args.enable_thinking == "true",
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_enable_prefix_caching=args.vllm_enable_prefix_caching,
    )
    return eval_main(eval_args)


def main() -> None:
    args = parse_args()
    out_dir = output_dir_for(args)
    dataset_args = json_arg(args.dataset_args, DEFAULT_DATASET_ARGS)
    resolved_config = {
        "model": args.model,
        "adapters": args.adapters or [],
        "eval_url": args.eval_url,
        "port": args.port,
        "datasets": args.datasets,
        "dataset_args": dataset_args,
        "repeats": args.repeats,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_enable_prefix_caching": args.vllm_enable_prefix_caching,
        "eval_num_proc": args.eval_num_proc,
        "infer_backend": args.infer_backend,
        "torch_dtype": args.torch_dtype,
        "system": args.system,
        "enable_thinking": args.enable_thinking == "true",
        "output_dir": str(out_dir),
    }

    if args.dry_run:
        print(json.dumps(resolved_config, indent=2, ensure_ascii=False))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, ensure_ascii=False) + "\n")

    print(
        f"[opd-eval] repeats={args.repeats}, seed={args.seed}, tp={args.vllm_tensor_parallel_size}",
        flush=True,
    )
    report = run_eval(args, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    summary = {
        "protocol": {
            "metric": "avg@N; EvalScope repeats samples each problem N times in one task",
            "repeats": args.repeats,
            "datasets": args.datasets,
            "amc_subset": dataset_args.get("amc", {}).get("subset_list"),
            "port": args.port,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "vllm_enable_prefix_caching": args.vllm_enable_prefix_caching,
        },
        "summary": summarize(report, "Native", args.repeats),
        "reports": [report],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[opd-eval] summary written to {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
