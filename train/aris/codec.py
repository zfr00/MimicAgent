"""Multimodal trajectory token assembly."""

import base64
import copy
from dataclasses import dataclass
import io
import math

from PIL import Image


class TokenBudgetError(ValueError):
    pass


class InitialPromptBudgetError(TokenBudgetError):
    pass


def _hf_messages(messages):
    result, images = copy.deepcopy(messages), []
    for message in result:
        if not isinstance(message["content"], list):
            continue
        for part in message["content"]:
            if part["type"] == "image_url":
                url = part.pop("image_url")["url"]
                if not url.startswith("data:image/"):
                    raise ValueError("Training image messages must contain embedded image data")
                with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                    images.append(image.convert("RGB"))
                part.update(type="image", image=url)
    return result, images


def _image_data(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class Encoded:
    tokens: list[int]
    image_data: list[str]
    train_inputs: dict | None


class ChatCodec:
    def __init__(self, tokenizer, processor=None, image_max_pixels=262144):
        self.tokenizer, self.processor = tokenizer, processor
        self.image_max_pixels = image_max_pixels
        self.dummy = [{"role": "system", "content": "System."},
                      {"role": "user", "content": "Question."}]
        marker = "ARIS_ASSISTANT_MARKER_7fb1"
        self.closed_dummy = self.dummy + [{"role": "assistant", "content": marker}]
        closed_text = self.render(self.closed_dummy, False)
        self.closing = self.tokenizer.encode(closed_text.split(marker, 1)[1], add_special_tokens=False)
        # The supported Qwen3-VL chat template closes an assistant with im_end.
        eos = tokenizer.eos_token_id
        if eos is None or not self.closing or self.closing[0] != eos:
            raise ValueError("Expected a Qwen chat template whose assistant suffix starts with EOS")
        self.eos = eos

    def render(self, messages, generation):
        hf, _ = _hf_messages(messages)
        return self.tokenizer.apply_chat_template(hf, tokenize=False, add_generation_prompt=generation)

    def encode_text(self, text, images):
        if images:
            resized = []
            for image in images:
                if image.width * image.height > self.image_max_pixels:
                    scale = math.sqrt(self.image_max_pixels / (image.width * image.height))
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                                         Image.Resampling.LANCZOS)
                resized.append(image)
            images = resized
            if self.processor is None:
                raise ValueError("Image-bearing trajectories require a multimodal processor")
            output = self.processor(text=[text], images=images, return_tensors="pt")
            ids = output["input_ids"][0]
            tokens = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            inputs = {k: v for k, v in output.items() if k in {"pixel_values", "image_grid_thw"}}
            if set(inputs) != {"pixel_values", "image_grid_thw"}:
                raise ValueError("Missing Qwen3-VL image tensors")
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            inputs = None
        return Encoded(list(tokens), [_image_data(i) for i in images], inputs)

    def initial(self, messages):
        _, images = _hf_messages(messages)
        return self.encode_text(self.render(messages, True), images)

    def observation(self, content):
        message = {"role": "user", "content": content}
        prefix = self.render(self.closed_dummy, False)
        complete = self.render(self.closed_dummy + [message], True)
        if not complete.startswith(prefix):
            raise ValueError("Chat template prefix mismatch")
        _, images = _hf_messages([message])
        return self.encode_text(complete[len(prefix):], images)

    def missing_closing(self, last_generated):
        for overlap in range(min(len(last_generated), len(self.closing)), 0, -1):
            if last_generated[-overlap:] == self.closing[:overlap]:
                return self.closing[overlap:]
        return self.closing[:]


class Transcript:
    def __init__(self, codec, messages, config, *, base_messages=None):
        self.codec, self.config = codec, config
        base = codec.initial(base_messages if base_messages is not None else messages)
        self.base_prompt_length = len(base.tokens)
        if self.base_prompt_length > config.max_prompt_tokens:
            raise InitialPromptBudgetError(
                f"Base prompt has {self.base_prompt_length} tokens, limit {config.max_prompt_tokens}")
        initial = codec.initial(messages) if base_messages is not None else base
        if len(initial.tokens) >= config.max_context_tokens:
            raise InitialPromptBudgetError(
                f"Initial context has {len(initial.tokens)} tokens, limit {config.max_context_tokens}")
        self.tokens = initial.tokens[:]
        self.prompt_length = len(self.tokens)
        self.image_data = initial.image_data[:]
        self.mm_chunks = [initial.train_inputs] if initial.train_inputs else []
        self.loss_mask, self.log_probs, self.last_generated = [], [], []

    @property
    def remaining(self):
        return min(self.config.max_context_tokens - len(self.tokens),
                   self.config.max_response_tokens - len(self.loss_mask))

    def append_generated(self, tokens, log_probs):
        if not tokens or len(tokens) != len(log_probs):
            raise ValueError("SGLang must return one log probability per actual generated token")
        if len(tokens) > self.remaining:
            raise TokenBudgetError("Generated response exceeds the requested token budget")
        self.tokens.extend(tokens)
        self.loss_mask.extend([1] * len(tokens))
        self.log_probs.extend(log_probs)
        self.last_generated = list(tokens)

    def append_observation(self, content):
        encoded = self.codec.observation(content)
        ids = self.codec.missing_closing(self.last_generated) + encoded.tokens
        if len(ids) >= self.remaining:  # reserve at least one next assistant token
            return False
        self.tokens.extend(ids)
        self.loss_mask.extend([0] * len(ids))
        self.log_probs.extend([0.0] * len(ids))
        self.image_data.extend(encoded.image_data)
        if encoded.train_inputs:
            self.mm_chunks.append(encoded.train_inputs)
        return True

    def apply_to(self, sample):
        import torch
        sample.tokens = self.tokens
        sample.response_length = len(self.loss_mask)
        sample.loss_mask, sample.rollout_log_probs = self.loss_mask, self.log_probs
        sample.response = self.codec.tokenizer.decode(self.tokens[self.prompt_length:], skip_special_tokens=False)
        sample.multimodal_train_inputs = (
            {key: torch.cat([chunk[key] for chunk in self.mm_chunks], dim=0)
             for key in self.mm_chunks[0]} if self.mm_chunks else None
        )
        sample.multimodal_inputs = None  # processed tensors above are authoritative
        if not sample.response_length:
            raise TokenBudgetError("Trajectory contains no response tokens")
        return sample


def encode_teacher(codec, student_messages, episode, config, sample, *, base_messages=None):
    """Replay assistant/tool turns under the STUDENT prompt and STUDENT tokenizer."""
    transcript = Transcript(codec, episode.initial_messages or student_messages, config,
                            base_messages=base_messages if base_messages is not None else episode.base_messages or None)
    for turn in episode.turns:
        if turn["role"] == "assistant":
            ids = codec.tokenizer.encode(turn["content"], add_special_tokens=False)
            ids = ids + [codec.eos] if not ids or ids[-1] != codec.eos else ids
            transcript.append_generated(ids, [0.0] * len(ids))
        elif turn["role"] == "tool":
            if not transcript.append_observation(turn["content"]):
                raise TokenBudgetError("Teacher observation exceeds student context budget")
        else:
            raise ValueError("Invalid teacher replay role")
    return transcript.apply_to(sample)
