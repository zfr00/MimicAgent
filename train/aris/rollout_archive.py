"""Persist student interactions and teacher attempts as atomic trajectory records."""

import json
import os
from pathlib import Path
import tempfile

from tools.data_records import question_language


class ArchivedTrajectory:
    def __init__(self, archive, sample, source, attempt):
        self.source = source
        self.path = archive.path / f"{source}_g{sample.group_index}_s{sample.index}_attempt_{attempt:02d}.json"
        if self.path.exists():
            raise FileExistsError(self.path)
        metadata = sample.metadata
        self._replace({
            "rollout_id": archive.rollout_id,
            "group_index": sample.group_index,
            "sample_index": sample.index,
            "target_id": metadata["target_id"],
            "language": question_language(metadata["question"], language=metadata.get("language")),
            "source": source,
            "attempt": attempt,
            "training_branch": "pending",
            "selected_for_training": False,
            "generation_status": "running",
            "reward_status": "pending",
            "question": metadata["question"],
            "ground_truth": sample.label,
            "episode": None,
            "reward": None,
            "reward_weights": {name: getattr(archive.config, f"{name}_weight")
                               for name in ("answer", "format", "tool", "analogy")},
            "tokens": [], "response_length": 0, "loss_mask": [], "rollout_log_probs": None,
            "status": sample.status.value,
        })

    def _replace(self, record):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".trajectory-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def update(self, **fields):
        record = json.loads(self.path.read_text(encoding="utf-8"))
        record.update(fields)
        self._replace(record)

    def capture(self, sample, episode, transcript=None, **fields):
        self.update(
            episode=episode.as_dict(),
            tokens=transcript.tokens if transcript is not None else sample.tokens,
            response_length=len(transcript.loss_mask) if transcript is not None else sample.response_length,
            loss_mask=transcript.loss_mask if transcript is not None else sample.loss_mask,
            rollout_log_probs=(transcript.log_probs if transcript is not None else sample.rollout_log_probs)
                              if self.source == "student" else None,
            status=sample.status.value,
            **fields,
        )


class RolloutArchive:
    def __init__(self, output_dir, rollout_id, config):
        directory = Path(output_dir) / "rollouts"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_{rollout_id:07d}"
        attempt = 1
        while True:
            suffix = "" if attempt == 1 else f"_attempt_{attempt:02d}"
            self.path = directory / f"{stem}{suffix}"
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                attempt += 1
        self.rollout_id, self.config = rollout_id, config
        self.entries = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def start(self, sample, *, source, attempt=1):
        entry = ArchivedTrajectory(self, sample, source, attempt)
        self.entries[(source, sample.group_index, sample.index, attempt)] = entry
        return entry

    def select(self, sample, branch, *, source, attempt=1):
        self.entries[(source, sample.group_index, sample.index, attempt)].update(
            training_branch=branch, selected_for_training=branch in {"grpo", "imitation"})

    def branch(self, sample, branch):
        self.entries[("student", sample.group_index, sample.index, 1)].update(training_branch=branch)

    def discard_group(self, group_index, reason):
        for (_, group, _, _), entry in self.entries.items():
            if group == group_index:
                entry.update(training_branch="discard", selected_for_training=False, skip_reason=reason)
