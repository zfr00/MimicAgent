"""Build the slime training command."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import sys
from .attention import TRAIN_ENV

from .config import DEMO_ROOT, TRAIN_ROOT, EXPERIMENTS, load_config


def model_arguments(checkpoint):
    config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("This launcher targets dense Qwen3-VL; select a Qwen3-VL-Instruct checkpoint")
    text = config["text_config"]
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    args = ["--swiglu", "--group-query-attention", "--use-rotary-position-embeddings",
            "--disable-bias-linear", "--normalization", "RMSNorm", "--qk-layernorm"]
    mapping = {"num_hidden_layers": "num-layers", "hidden_size": "hidden-size", "intermediate_size": "ffn-hidden-size",
               "num_attention_heads": "num-attention-heads", "num_key_value_heads": "num-query-groups",
               "vocab_size": "vocab-size", "rms_norm_eps": "norm-epsilon"}
    for source, target in mapping.items():
        args += ["--" + target, str(text[source])]
    args += ["--kv-channels", str(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])),
             "--rotary-base", str(rope.get("rope_theta", text.get("rope_theta", 5000000)))]
    if not text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        args.append("--untie-embeddings-and-output-weights")
    return args


def build_arguments(args, config):
    if min(args.num_gpus, args.tp, args.pp, args.rollout_gpus_per_engine, args.steps, args.save_interval) < 1:
        raise ValueError("GPU counts, parallel sizes, steps and save interval must be positive")
    if args.num_gpus % (args.tp * args.pp) or args.num_gpus % args.rollout_gpus_per_engine:
        raise ValueError("NUM_GPUS must be divisible by TP*PP and by rollout GPUs per engine")
    if config.prompts_per_step * config.group_size % (args.num_gpus // (args.tp * args.pp)):
        raise ValueError("The B*G batch must be divisible by the number of data-parallel ranks")
    result = model_arguments(args.hf_checkpoint)
    load_path, checkpoint_step = resolve_resume(args.resume) if args.resume else (args.hf_checkpoint, None)
    pairs = {
        "train-backend": "megatron", "hf-checkpoint": args.hf_checkpoint,
        "train-env-vars": json.dumps(TRAIN_ENV),
        "load": load_path, "save": args.output, "save-interval": args.save_interval,
        "actor-num-nodes": 1, "actor-num-gpus-per-node": args.num_gpus, "rollout-num-gpus": args.num_gpus,
        "rollout-num-gpus-per-engine": args.rollout_gpus_per_engine,
        "prompt-data": args.train_data, "input-key": "messages", "label-key": "label", "metadata-key": "metadata",
        "rollout-function-path": "aris.rollout.generate_rollout",
        "custom-generate-function-path": "aris.rollout.generate_student", "custom-rm-path": "aris.rollout.reward_function",
        "custom-reward-post-process-path": "aris.switching.post_process_rewards",
        "loss-type": "custom_loss", "custom-loss-function-path": "aris.loss.loss_function",
        "custom-model-provider-path": "aris.model_provider.model_provider", "megatron-to-hf-mode": "bridge",
        "num-rollout": args.steps, "rollout-batch-size": config.prompts_per_step, "n-samples-per-prompt": config.group_size,
        "global-batch-size": config.prompts_per_step * config.group_size, "micro-batch-size": 1,
        "rollout-temperature": config.temperature, "rollout-top-p": 1.0, "rollout-top-k": -1,
        "rollout-max-prompt-len": config.max_prompt_tokens, "rollout-max-response-len": config.max_response_tokens,
        "rollout-max-context-len": config.max_context_tokens, "seq-length": config.max_context_tokens,
        "reward-key": "total", "eval-reward-key": "answer", "advantage-estimator": "grpo",
        "eps-clip": config.clip_low, "eps-clip-high": config.clip_high,
        "kl-coef": 0, "kl-loss-coef": 0, "entropy-coef": 0,
        "optimizer": "adam", "lr": config.learning_rate, "lr-decay-style": "constant",
        "weight-decay": config.weight_decay, "adam-beta1": config.adam_beta1, "adam-beta2": config.adam_beta2,
        "clip-grad": config.grad_clip, "tensor-model-parallel-size": args.tp, "pipeline-model-parallel-size": args.pp,
        "context-parallel-size": 1, "expert-model-parallel-size": 1, "expert-tensor-parallel-size": 1,
        "recompute-granularity": "full", "recompute-method": "uniform", "recompute-num-layers": 1,
        "attention-dropout": 0, "hidden-dropout": 0, "attention-backend": "flash",
        "sglang-mem-fraction-static": 0.7, "sglang-context-length": config.max_context_tokens,
        "sglang-server-concurrency": config.rollout_concurrency,
        "rollout-seed": args.seed,
    }
    for key, value in pairs.items():
        result.extend(["--" + key, str(value)])
    if checkpoint_step is not None:
        result += ["--ckpt-step", str(checkpoint_step), "--start-rollout-id", str(checkpoint_step + 1)]
    result += ["--colocate", "--rollout-shuffle", "--use-rollout-logprobs", "--accumulate-allreduce-grads-in-fp32",
               "--attention-softmax-in-fp32"]
    if args.tp > 1:
        result.append("--sequence-parallel")
    if args.eval_data:
        result += ["--eval-prompt-data", "sciexplore", str(args.eval_data), "--eval-interval", str(args.eval_interval),
                   "--n-samples-per-eval-prompt", str(config.eval_runs),
                   "--eval-temperature", str(config.eval_temperature)]
    return result


def resolve_resume(resume):
    """Resolve model iteration and dataset progress from one checkpoint root."""
    path = Path(resume).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {path}")
    match = re.fullmatch(r"iter_(\d{7})", path.name)
    root, step = (path.parent, int(match[1])) if match else (path, None)
    if not (root / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError(f"Checkpoint root is missing latest_checkpointed_iteration.txt: {root}")
    return root, step


def resolve_config_path(config_path=None, resume=None):
    if config_path is not None:
        return Path(config_path).resolve()
    if resume is not None:
        resume = Path(resume).resolve()
        candidates = [resume / "aris_config.json"]
        if resume.name.startswith("iter_") and resume.name[5:].isdigit():
            candidates.append(resume.parent / "aris_config.json")
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"No aris_config.json found for {resume}; provide --config to resume this checkpoint")
    return Path(os.environ.get("ARIS_CONFIG") or TRAIN_ROOT / "configs/aris.json").resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", type=Path, default=os.environ.get("HF_CHECKPOINT"))
    parser.add_argument("--train-data", type=Path,
                        default=os.environ.get("TRAIN_DATA", str(DEMO_ROOT / "data/train/aris_train.jsonl")))
    parser.add_argument("--output", type=Path, default=os.environ.get("OUTPUT_DIR", str(TRAIN_ROOT / "outputs/aris")))
    parser.add_argument("--config", type=Path, help="Override the saved resume configuration or default ARIS config")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--prefix-k", type=int, choices=range(1, 5))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "8")))
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--rollout-gpus-per-engine", type=int, default=2)
    parser.add_argument("--steps", type=int, default=int(os.environ.get("NUM_ROLLOUT", "1000")))
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-per-step", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--judge-repeats", type=int)
    parser.add_argument("--eval-temperature", type=float)
    parser.add_argument("--eval-runs", type=int)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.hf_checkpoint is None or args.train_data is None:
        parser.error("Provide --hf-checkpoint/--train-data or HF_CHECKPOINT/TRAIN_DATA")
    for key in ("hf_checkpoint", "train_data", "output", "config", "eval_data", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.config = resolve_config_path(args.config, args.resume)
    config = load_config(args.config)
    overrides = {k: getattr(args, k) for k in ("prompts_per_step", "group_size", "judge_repeats",
                                              "eval_temperature", "eval_runs", "experiment", "prefix_k") if getattr(args, k) is not None}
    config = replace(config, **overrides).validate()
    train_args = build_arguments(args, config)
    print(f"Configuration: {args.config}", flush=True)
    print(f"Experiment: {config.experiment}", flush=True)
    print(shlex.join([sys.executable, str(TRAIN_ROOT / "slime/train.py"), *train_args]), flush=True)
    if args.dry_run:
        print("Training command generated.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_config = args.output / "aris_config.json"
    runtime_config.write_text(json.dumps(asdict(config), indent=2) + "\n")
    os.environ["ARIS_CONFIG"] = str(runtime_config)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Keep local tool/router traffic off an optional remote API proxy.
    no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
    os.environ["no_proxy"] = ",".join(filter(None, [no_proxy, "127.0.0.1", "localhost"]))
    env = {k: v for k, v in os.environ.items() if k.startswith("ARIS_") or k in {
        "PYTHONPATH", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CUDA_DEVICE_MAX_CONNECTIONS",
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"}}
    import ray
    init = {"address": args.ray_address, "runtime_env": {"env_vars": env}}
    if not args.ray_address:
        init.update(num_gpus=args.num_gpus, include_dashboard=False)
    ray.init(**init)
    try:
        sys.argv = [str(TRAIN_ROOT / "slime/train.py"), *train_args]
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
