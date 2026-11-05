"""Clipped GRPO and masked expert NLL."""

import copy
import torch


def per_token_loss(log_probs, old_log_probs, advantages, branch, scale, clip_low, clip_high):
    if branch == "discard":
        return log_probs * 0
    if branch == "imitation":
        return -scale * log_probs
    if branch != "grpo":
        raise ValueError(f"Unknown ARIS branch: {branch}")
    if old_log_probs is None or advantages is None:
        raise ValueError("GRPO needs frozen behavior log probabilities and advantages")
    ratio = (log_probs - old_log_probs.detach()).exp()
    advantage = advantages.detach()
    return -scale * torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)


def loss_function(args, batch, logits, sum_of_sample_mean):
    """slime --custom-loss-function-path hook; return SUM of slot means.

    slime subsequently divides by the global B*G slots. A single teacher slot
    has scale G*lambda, yielding lambda/B; each student slot has scale 1,
    yielding 1/(B*G).
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    if args.calculate_per_token_loss:
        raise ValueError("Invalid loss reduction: calculate_per_token_loss=True")
    if not batch.get("metadata"):
        raise RuntimeError("Missing ARIS metadata: use the bundled train/slime sources")
    meta = batch["metadata"]
    branches = {m["aris_branch"] for m in meta}
    if not branches <= {"grpo", "imitation", "discard"}:
        raise ValueError(f"Invalid ARIS branches {branches}")
    common = dict(unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
                  response_lengths=batch["response_lengths"], with_entropy=False)
    rl_probs = nll_probs = None
    if "grpo" in branches:
        _, values = get_log_probs_and_entropy(logits, args=args, **common)
        rl_probs = values["log_probs"]
    if "imitation" in branches:
        nll_args = copy.copy(args)
        nll_args.rollout_temperature = 1.0
        _, values = get_log_probs_and_entropy(logits, args=nll_args, **common)
        nll_probs = values["log_probs"]
    if rl_probs is None and nll_probs is None:
        zero = logits.sum() * 0
        return zero, {"loss": zero.detach()}
    exemplar = rl_probs if rl_probs is not None else nll_probs
    old = batch.get("rollout_log_probs") if args.use_rollout_logprobs else batch.get("log_probs")
    all_rl, all_nll = [], []
    for i, m in enumerate(meta):
        branch = m["aris_branch"]
        probs = (rl_probs[i] if branch == "grpo" else nll_probs[i]) if branch != "discard" else exemplar[i]
        tokens = per_token_loss(probs, old[i] if old else None,
                                batch["advantages"][i], branch, m["loss_scale"], args.eps_clip, args.eps_clip_high)
        zero = probs * 0
        all_rl.append(tokens if branch == "grpo" else zero)
        all_nll.append(tokens if branch == "imitation" else zero)
    rl = sum_of_sample_mean(torch.cat(all_rl))
    nll = sum_of_sample_mean(torch.cat(all_nll))
    loss = rl + nll + logits.sum() * 0
    return loss, {"loss": loss.detach()}
