"""Online multi-turn student rollouts, reward grading, and zero-answer intervention."""

import asyncio
import copy
from contextvars import ContextVar

import aiohttp


from .clients import ModelAPIs
from .codec import ChatCodec, InitialPromptBudgetError, TokenBudgetError, Transcript, encode_teacher
from .config import load_config
from .protocol import Episode, format_score, initial_messages, allowed_tool_names
from tools.image_bank import ImageBank
from tools.agent_messages import with_image_data
from tools.data_records import question_language
from tools.tool_errors import ExemplarPrefixError
from .interaction import record_observation, record_reply, run_episode
from .tool_client import ToolExecutor
from .rewards import Rewarder
from .switching import switch_group, zero_answer
from .async_utils import gather_strict
from .rollout_archive import RolloutArchive

_evaluation_runtime = ContextVar("aris_evaluation_runtime", default=None)


class TeacherBudgetLimit(RuntimeError):
    def __init__(self, target_id, reasons):
        rounds = "tool_round_limit" in reasons
        self.reason = ("teacher_tool_round_limit" if set(reasons) == {"tool_round_limit"}
                       else "teacher_budget_limit" if rounds else "teacher_context_limit")
        super().__init__(f"Teacher attempts exhausted for {target_id}: {reasons}")


class Runtime:
    def __init__(self, args, config, state, *, evaluation=False):
        self.args, self.config, self.state = args, config, state
        self.codec = ChatCodec(state.tokenizer, state.processor, config.image_max_pixels)
        self.evaluation = evaluation
        self.apis = ModelAPIs(config, roles=("primary", "secondary") if evaluation else config.training_api_roles)
        self.rewarder = Rewarder(config, self.apis)
        self.semaphore = asyncio.Semaphore(config.rollout_concurrency)
        self.prefix_tasks = {}
        self.evaluation_tasks = set()
        self.closing = False

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(trust_env=True)
        self.tools = ToolExecutor(self.session, self.config)
        return self

    async def __aexit__(self, *exc):
        self.closing = True
        tasks = [*self.evaluation_tasks, *self.prefix_tasks.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.session.close()
        await self.apis.close()

    def messages(self, sample, teacher=False, image_bank=None, *, include_prefix=True):
        return initial_messages(sample.metadata["question"], sample.metadata["image_paths"], self.config,
                                ground_truth=str(sample.label) if teacher else None,
                                exemplar_prefix=sample.metadata.get("aris_exemplar_prefix")
                                if self.config.uses_exemplar_prefix and include_prefix else None,
                                image_bank=image_bank, target_id=sample.metadata["target_id"])

    def transcript(self, sample, messages):
        base = self.messages(sample, include_prefix=False) if self.config.uses_exemplar_prefix else None
        return Transcript(self.codec, messages, self.config, base_messages=base)

    async def validate_prompt(self, sample):
        async with self.semaphore:
            Transcript(self.codec, self.messages(sample, include_prefix=False), self.config)
            if self.config.uses_exemplar_prefix:
                await self.prepare_prefix(sample)
                self.transcript(sample, self.messages(sample))

    async def prepare_prefix(self, sample):
        if self.config.uses_exemplar_prefix and "aris_exemplar_prefix" not in sample.metadata:
            metadata = sample.metadata
            key = (metadata["target_id"], question_language(metadata["question"],
                   language=metadata.get("language")), self.config.prefix_k)
            if key not in self.prefix_tasks:
                self.prefix_tasks[key] = asyncio.create_task(self._prepare_prefix(sample))
            task = self.prefix_tasks[key]
            try:
                sample.metadata["aris_exemplar_prefix"] = copy.deepcopy(await task)
            except Exception:
                if self.prefix_tasks.get(key) is task:
                    del self.prefix_tasks[key]
                raise

    async def _prepare_prefix(self, sample):
        base = self.messages(sample, include_prefix=False)
        Transcript(self.codec, base, self.config)
        return await self.tools.prefetch_exemplars(sample)

    async def inference(self, transcript, sampling_params):
        from slime.rollout.sglang_rollout import get_model_url
        from slime.utils.http_utils import post
        params = dict(sampling_params)
        params.update(max_new_tokens=transcript.remaining,
                      top_p=1.0, top_k=-1, skip_special_tokens=False, no_stop_trim=True)
        payload = {"input_ids": transcript.tokens, "sampling_params": params, "return_logprob": True}
        if transcript.image_data:
            payload["image_data"] = transcript.image_data
        output = await post(get_model_url(self.args, "actor", "/generate"), payload)
        meta = output["meta_info"]
        records = meta.get("output_token_logprobs")
        if not records:
            raise RuntimeError("Missing SGLang output_token_logprobs")
        ids, log_probs = [v[1] for v in records], [v[0] for v in records]
        if any(v is None for v in log_probs):
            raise RuntimeError("SGLang returned a missing generated-token log probability")
        text = self.codec.tokenizer.decode(ids, skip_special_tokens=False).strip()
        eos_text = self.codec.tokenizer.decode([self.codec.eos], skip_special_tokens=False)
        if text.endswith(eos_text):
            text = text[:-len(eos_text)].strip()
        return text, ids, log_probs, meta["finish_reason"]["type"]

    async def score_episode(self, sample, episode, entry=None):
        try:
            if entry is None:
                reward = await self.rewarder.score(sample.metadata["question"], str(sample.label), episode)
            else:
                reward = await self.rewarder.score(
                    sample.metadata["question"], str(sample.label), episode,
                    on_progress=lambda reward: entry.update(reward=reward))
                entry.update(reward=reward, reward_status="complete")
            return reward
        except BaseException as exc:
            if entry is not None:
                entry.update(reward_status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                             error={"phase": "reward", "type": type(exc).__name__})
            raise

    async def student(self, sample, sampling_params):
        from slime.utils.types import Sample
        async with self.semaphore:
            archive = getattr(self, "archive", None)
            entry = archive.start(sample, source="student") if archive is not None else None
            episode, transcript = Episode(), None

            def progress(episode, transcript):
                if entry is not None:
                    entry.capture(sample, episode, transcript)

            try:
                await self.prepare_prefix(sample)
                bank = ImageBank(max_pixels=self.config.image_max_pixels)
                messages = self.messages(sample, image_bank=bank)
                episode.initial_messages = copy.deepcopy(messages)
                if self.config.uses_exemplar_prefix:
                    episode.base_messages = self.messages(sample, include_prefix=False)
                episode.image_bank = bank.snapshot()
                transcript = self.transcript(sample, messages)
                await run_episode(
                    transcript, lambda transcript: self.inference(transcript, sampling_params),
                    lambda action, staged: self.tools.execute(action, sample, image_bank=staged),
                    self.config.max_tool_rounds, allowed_tools=allowed_tool_names(self.config),
                    image_bank=bank, initial_messages=messages, episode=episode, on_progress=progress,
                )
                transcript.apply_to(sample)
                sample.status = (Sample.Status.TRUNCATED if episode.stopped_reason in {"length", "context_limit"}
                                 else Sample.Status.COMPLETED)
            except BaseException as exc:
                if isinstance(exc, (InitialPromptBudgetError, ExemplarPrefixError)) and getattr(self, "evaluation", False):
                    prefix_failed = isinstance(exc, ExemplarPrefixError)
                    sample.status = Sample.Status.FAILED if prefix_failed else Sample.Status.TRUNCATED
                    sample.tokens, sample.response, sample.response_length = [], "", 0
                    sample.loss_mask, sample.rollout_log_probs = [], []
                    sample.multimodal_inputs = sample.multimodal_train_inputs = None
                    sample.reward = {"answer": 0.0, "verdict": "unknown"}
                    sample.metadata["aris_prefix_error" if prefix_failed else "aris_budget_error"] = str(exc)
                    episode.stopped_reason = "prefix_error" if prefix_failed else "context_limit"
                    if not episode.initial_messages:
                        bank = ImageBank(max_pixels=self.config.image_max_pixels)
                        episode.initial_messages = self.messages(sample, image_bank=bank, include_prefix=not prefix_failed)
                        episode.image_bank = bank.snapshot()
                    if entry is not None:
                        entry.capture(sample, episode, generation_status="complete",
                                      reward=sample.reward, reward_status="complete")
                    return sample
                sample.status = Sample.Status.ABORTED
                episode.stopped_reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                if entry is not None:
                    entry.capture(sample, episode, transcript, generation_status=episode.stopped_reason,
                                  reward_status="skipped",
                                  error={"phase": "generation", "type": type(exc).__name__})
                raise
            finally:
                sample.metadata["aris_episode"] = episode.as_dict()
            if entry is not None:
                entry.capture(sample, episode, transcript, generation_status="complete")
            if getattr(self, "evaluation", False):
                from evaluation.extract_answer import extract_final_answer
                verdict = await self.rewarder.answer_verdict(
                    sample.metadata["question"], str(sample.label), extract_final_answer(episode.final_reply))
                sample.reward = {"answer": float(verdict == "correct"), "verdict": verdict}
            else:
                sample.reward = await self.score_episode(sample, episode, entry)
            return sample

    async def teacher_episode(self, original, *, episode=None, on_progress=None):
        episode = episode if episode is not None else Episode()
        bank, transcript = ImageBank(max_pixels=self.config.image_max_pixels), None

        def progress():
            episode.image_bank = bank.snapshot()
            if on_progress is not None:
                on_progress(episode, transcript)

        try:
            await self.prepare_prefix(original)
            student_messages = self.messages(original, image_bank=bank)
            episode.initial_messages = copy.deepcopy(student_messages)
            if self.config.uses_exemplar_prefix:
                episode.base_messages = self.messages(original, include_prefix=False)
            transcript = self.transcript(original, student_messages)
            messages = self.messages(original, teacher=True, image_bank=bank)
            progress()
            for round_index in range(self.config.max_tool_rounds + 1):
                if transcript.remaining <= 0:
                    episode.stopped_reason = "context_limit"
                    break
                reply = await self.apis.chat("teacher", with_image_data(messages, transcript.image_data),
                                             min(self.config.teacher_max_tokens_per_turn, transcript.remaining))
                ids = self.codec.tokenizer.encode(reply, add_special_tokens=False)
                if not ids or ids[-1] != self.codec.eos:
                    ids.append(self.codec.eos)
                action = record_reply(episode, reply, round_index, self.config.max_tool_rounds,
                                      finish=getattr(reply, "finish_reason", "stop"),
                                      allowed_tools=allowed_tool_names(self.config))
                try:
                    transcript.append_generated(ids, [0.0] * len(ids))
                except TokenBudgetError:
                    episode.stopped_reason = "context_limit"
                    break
                progress()
                if action is None:
                    break
                staged = bank.fork()
                result, content, success = await self.tools.execute(action, original, image_bank=staged)
                try:
                    accepted = transcript.append_observation(content)
                except BaseException:
                    record_observation(episode, action, result, content, success, accepted=False)
                    raise
                record_observation(episode, action, result, content, success, accepted=accepted)
                if not accepted:
                    episode.stopped_reason = "context_limit"
                    break
                bank.commit(staged)
                messages.extend([{"role": "assistant", "content": reply}, {"role": "user", "content": content}])
                progress()
            return episode
        finally:
            progress()

    async def teacher(self, original):
        from slime.utils.types import Sample
        errors = []
        budget_failures = 0
        archive = getattr(self, "archive", None)
        for attempt in range(1, self.config.teacher_attempts + 1):
            sample = copy.copy(original)
            sample.metadata = {k: v for k, v in original.metadata.items() if k != "aris_episode"}
            sample.metadata["aris_teacher_attempt"] = attempt
            sample.tokens, sample.loss_mask, sample.rollout_log_probs = [], [], None
            sample.response, sample.response_length, sample.reward = "", 0, None
            sample.status = Sample.Status.PENDING
            episode = Episode()
            entry = archive.start(sample, source="teacher", attempt=attempt) if archive is not None else None

            def progress(episode, transcript):
                if entry is not None:
                    entry.capture(sample, episode, transcript)

            try:
                await self.teacher_episode(original, episode=episode, on_progress=progress)
                rejected = None
                over_budget = episode.stopped_reason in {"length", "context_limit", "tool_round_limit"}
                if episode.stopped_reason != "answered" or format_score(episode) != 1:
                    rejected = episode.stopped_reason or "format_violation"
                    if episode.stopped_reason == "answered":
                        rejected = "format_violation"
                else:
                    try:
                        base = self.messages(original, include_prefix=False) if self.config.uses_exemplar_prefix else None
                        encode_teacher(self.codec, episode.initial_messages, episode, self.config, sample,
                                       base_messages=base)
                    except TokenBudgetError as exc:
                        rejected = str(exc)
                        over_budget = True
                sample.metadata["aris_episode"] = episode.as_dict()
                sample.status = (Sample.Status.COMPLETED if rejected is None else
                                 Sample.Status.TRUNCATED if over_budget
                                 else Sample.Status.FAILED)
                if entry is not None:
                    entry.update(episode=episode.as_dict(), status=sample.status.value,
                                 generation_status="rejected" if rejected else "complete",
                                 rejection_reason=rejected,
                                 training_branch="discard" if rejected else "pending")
                    if rejected is None:
                        entry.capture(sample, episode)
            except BaseException as exc:
                sample.status = Sample.Status.ABORTED
                episode.stopped_reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                if entry is not None:
                    entry.update(episode=episode.as_dict(), status=sample.status.value,
                                 generation_status=episode.stopped_reason, reward_status="skipped",
                                 error={"phase": "generation", "type": type(exc).__name__})
                raise
            if entry is not None:
                await self.score_episode(sample, episode, entry)
            if rejected is not None:
                errors.append(rejected)
                budget_failures += int(over_budget)
                continue
            return sample
        if budget_failures == self.config.teacher_attempts:
            raise TeacherBudgetLimit(original.metadata['target_id'], errors)
        raise RuntimeError(f"No valid teacher trajectory for {original.metadata['target_id']}: {errors}")


async def _training_groups(runtime, data_source, *, count=None, visited=None):
    """Select complete prompt groups within the initial-input budgets."""
    config = runtime.config
    total = len(data_source)
    if total < config.prompts_per_step:
        raise ValueError(f"Need at least {config.prompts_per_step} unique prompts; found {total}")
    count = config.prompts_per_step if count is None else count
    selected = []
    visited = set() if visited is None else visited

    async def eligible(group):
        try:
            await runtime.validate_prompt(group[0])
        except InitialPromptBudgetError:
            return False
        except ExemplarPrefixError as exc:
            archive = getattr(runtime, "archive", None)
            if archive is not None:
                from slime.utils.types import Sample
                for sample in group:
                    sample.status = Sample.Status.FAILED
                    entry = archive.start(sample, source="student")
                    entry.update(generation_status="failed", reward_status="skipped",
                                 episode=Episode(stopped_reason="prefix_error").as_dict(),
                                 error={"phase": "prefix", **exc.result})
                archive.discard_group(group[0].group_index, "exemplar_prefix_failed")
            return False
        if config.uses_exemplar_prefix:
            for sample in group[1:]:
                sample.metadata["aris_exemplar_prefix"] = copy.deepcopy(group[0].metadata["aris_exemplar_prefix"])
        return True

    while len(selected) < count:
        if len(visited) >= total:
            raise ValueError(f"Need {count} additional eligible unique prompts; found {len(selected)}")
        groups = data_source.get_samples(count - len(selected))
        if not groups or any(len(group) != config.group_size for group in groups):
            raise ValueError("Dataset did not supply complete trajectory groups")
        candidates = []
        for group in groups:
            metadata = group[0].metadata
            key = (metadata["target_id"], question_language(metadata["question"], language=metadata.get("language")))
            if key not in visited:
                visited.add(key)
                candidates.append(group)
        valid = await gather_strict(*(eligible(group) for group in candidates))
        selected.extend(group for group, accepted in zip(candidates, valid) if accepted)
        if len(visited) >= total and len(selected) < count:
            raise ValueError(f"Need at least {count} eligible unique prompts; found {len(selected)}")
    return selected


async def _generate_train(args, rollout_id, data_source):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.base_types import RolloutFnTrainOutput

    config = load_config()
    if args.n_samples_per_prompt != config.group_size or args.rollout_batch_size != config.prompts_per_step:
        raise ValueError("slime group/batch arguments disagree with ARIS_CONFIG")
    if args.partial_rollout or args.rollout_top_p != 1 or args.rollout_top_k not in (-1, 0):
        raise ValueError("This ARIS implementation uses complete on-policy groups and untruncated sampling")
    state = GenerateState(args)
    with RolloutArchive(args.save, rollout_id, config) as archive:
        async with Runtime(args, config, state) as runtime:
            runtime.archive = archive

            async def one_group(group):
                students = await gather_strict(*(runtime.student(s, state.sampling_params) for s in group))
                branch = "discard" if zero_answer(students) else "grpo"
                for sample in students:
                    archive.branch(sample, branch)
                try:
                    return await switch_group(students, runtime.teacher, config)
                except TeacherBudgetLimit as exc:
                    archive.discard_group(group[0].group_index, exc.reason)
                    return None

            switched, visited = [], set()
            while len(switched) < config.prompts_per_step:
                groups = await _training_groups(runtime, data_source,
                                               count=config.prompts_per_step - len(switched), visited=visited)
                completed = await gather_strict(*(one_group(group) for group in groups))
                switched.extend(group for group in completed if group is not None)
        for group in switched:
            for sample in group:
                branch = sample.train_metadata["aris_branch"]
                if branch == "grpo":
                    archive.select(sample, branch, source="student")
                elif branch == "imitation":
                    archive.select(sample, branch, source="teacher", attempt=sample.metadata["aris_teacher_attempt"])
    return RolloutFnTrainOutput(samples=switched)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run
    if evaluation:
        return run(_generate_evaluation(args, rollout_id))
    return run(_generate_train(args, rollout_id, data_source))


async def _generate_evaluation(args, rollout_id):
    from slime.rollout.sglang_rollout import GenerateState, eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput

    async with Runtime(args, load_config(), GenerateState(args), evaluation=True) as runtime:
        token = _evaluation_runtime.set(runtime)
        try:
            results = await gather_strict(*(eval_rollout_single_dataset(args, rollout_id, dataset)
                                           for dataset in args.eval_datasets))
            return RolloutFnEvalOutput(data={name: value for result in results for name, value in result.items()})
        finally:
            _evaluation_runtime.reset(token)


async def generate_student(args, sample, sampling_params, evaluation=False):
    """Custom generate hook used by slime's standard evaluation loop."""
    runtime = _evaluation_runtime.get() if evaluation else None
    if runtime is not None:
        if runtime.closing:
            raise asyncio.CancelledError('Evaluation runtime has closed')
        task = asyncio.current_task()
        runtime.evaluation_tasks.add(task)
        try:
            return await runtime.student(sample, sampling_params)
        finally:
            runtime.evaluation_tasks.discard(task)
    from slime.rollout.sglang_rollout import GenerateState
    async with Runtime(args, load_config(), GenerateState(args), evaluation=evaluation) as runtime:
        return await runtime.student(sample, sampling_params)


async def reward_function(args, sample):
    if sample.reward is None:
        raise ValueError("ARIS reward requires generate_student to have completed")
    return sample.reward
