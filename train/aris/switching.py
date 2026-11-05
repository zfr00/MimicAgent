"""Per-prompt mutual exclusion and fixed-slot, equal-prompt normalization."""

from collections import defaultdict
import copy
import math


def zero_answer(group):
    if not group:
        raise ValueError("Empty ARIS group")
    correctness = [s.reward["answer"] for s in group]
    if any(c not in (0, 1) for c in correctness):
        raise ValueError("Answer correctness must be binary")
    return max(correctness) == 0


async def switch_group(group, teacher_factory, config):
    if len(group) != config.group_size or len({s.group_index for s in group}) != 1:
        raise ValueError("Expected one complete prompt group")
    if not zero_answer(group):
        for sample in group:
            sample.rollout_id = sample.index
            sample.train_metadata = {"aris_branch": "grpo", "loss_scale": 1.0}
        return group

    teacher = await teacher_factory(group[0])
    if not teacher.response_length or not teacher.loss_mask or sum(teacher.loss_mask) == 0:
        raise ValueError("Teacher trajectory has no supervised tokens")
    teacher.index, teacher.group_index = group[0].index, group[0].group_index
    teacher.rollout_id = teacher.index
    teacher.train_metadata = {"aris_branch": "imitation", "loss_scale": config.group_size * config.imitation_lambda}
    teacher.reward = {"total": 0.0, "answer": 0.0}
    result = [teacher]
    for original in group[1:]:
        pad = copy.copy(original)
        pad.tokens = [teacher.tokens[0], teacher.tokens[-1]]
        pad.prompt, pad.response = "", ""
        pad.response_length, pad.loss_mask, pad.rollout_log_probs = 1, [0], [0.0]
        pad.multimodal_inputs = pad.multimodal_train_inputs = None
        pad.rollout_id = pad.index
        pad.train_metadata = {"aris_branch": "discard", "loss_scale": 0.0}
        pad.metadata = {"aris_padding": True}
        pad.reward = {"total": 0.0, "answer": 0.0}
        pad.status = teacher.status
        result.append(pad)
    return result


def post_process_rewards(args, samples):
    """slime hook, executed globally BEFORE DP partitioning or batch balancing."""
    from .config import load_config
    config = load_config()
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.group_index].append((index, sample))
    normalized = [0.0] * len(samples)
    for members in groups.values():
        if len(members) != config.group_size:
            raise ValueError("Incomplete ARIS group before advantage normalization")
        rl = [(i, s) for i, s in members if s.train_metadata["aris_branch"] == "grpo"]
        if not rl:
            if [s.train_metadata["aris_branch"] for _, s in members].count("imitation") != 1:
                raise ValueError("Zero-answer group must have exactly one teacher")
            continue
        if len(rl) != config.group_size:
            raise ValueError("Mixed training branches in one prompt group")
        rewards = [float(s.reward["total"]) for _, s in rl]
        mean = math.fsum(rewards) / len(rewards)
        variance = math.fsum((r - mean) ** 2 for r in rewards) / (len(rewards) - config.advantage_std_correction)
        std = math.sqrt(variance)
        for (index, _), reward in zip(rl, rewards, strict=True):
            normalized[index] = (reward - mean) / (std + config.advantage_epsilon) if std else 0.0
    return [float(s.reward["total"]) for s in samples], normalized
