"""Agent state transitions shared by SGLang rollout and chat-completions evaluation."""

import copy

from .protocol import Episode, parse_action


def record_reply(episode, reply, round_index, max_tool_rounds, finish="stop", *, allowed_tools=None):
    episode.turns.append({"role": "assistant", "content": reply})
    if finish in {"abort", "error"}:
        raise RuntimeError(f"Model generation aborted: {finish}")
    if finish == "length":
        episode.stopped_reason = "length"
        return None
    action = parse_action(reply)
    if action is None:
        episode.stopped_reason = "format_violation"
    elif action[0] == "answer":
        episode.stopped_reason = "answered"
    elif allowed_tools is not None and action[1]["name"] not in allowed_tools:
        episode.stopped_reason = "tool_disabled"
    elif round_index >= max_tool_rounds:
        episode.stopped_reason = "tool_round_limit"
    else:
        return action[1]
    return None


def record_observation(episode, action, result, content, success, *, accepted=True):
    episode.calls.append(copy.deepcopy({**action, "result": result, "content": content, "success": success,
                          "accepted": accepted, "turn_index": len(episode.turns) - 1}))
    if accepted:
        episode.turns.append({"role": "tool", "content": copy.deepcopy(content)})


async def run_episode(transcript, generate, execute, max_tool_rounds, *, allowed_tools=None,
                      image_bank=None, initial_messages=None, on_observation=None,
                      episode=None, on_progress=None):
    episode = episode if episode is not None else Episode()
    episode.initial_messages = copy.deepcopy(initial_messages or [])

    def progress():
        if image_bank is not None:
            episode.image_bank = image_bank.snapshot()
        if on_progress is not None:
            on_progress(episode, transcript)

    progress()
    for round_index in range(max_tool_rounds + 1):
        if transcript.remaining <= 0:
            episode.stopped_reason = "context_limit"
            break
        reply, ids, log_probs, finish = await generate(transcript)
        try:
            transcript.append_generated(ids, log_probs)
        except BaseException:
            episode.turns.append({"role": "assistant", "content": reply})
            raise
        action = record_reply(episode, reply, round_index, max_tool_rounds, finish, allowed_tools=allowed_tools)
        progress()
        if action is None:
            break
        staged = image_bank.fork() if image_bank is not None else None
        result, content, success = await execute(action, staged) if staged is not None else await execute(action)
        try:
            accepted = transcript.append_observation(content)
        except BaseException:
            record_observation(episode, action, result, content, success, accepted=False)
            raise
        record_observation(episode, action, result, content, success, accepted=accepted)
        if not accepted:
            episode.stopped_reason = "context_limit"
            break
        if staged is not None:
            image_bank.commit(staged)
        if on_observation is not None:
            on_observation(content)
        progress()
    progress()
    return episode
