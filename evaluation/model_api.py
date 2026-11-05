"""Text and image chat adapters for OpenAI-compatible and Anthropic endpoints."""

import json
import os


def anthropic_messages(messages):
    system, conversation = [], []
    for message in messages:
        content = message['content']
        parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        blocks = []
        for part in parts:
            if part['type'] == 'text':
                if part['text']:
                    blocks.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image_url':
                url = part['image_url']['url']
                if url.startswith('data:image/'):
                    header, data = url.split(',', 1)
                    if not header.endswith(';base64'):
                        raise ValueError('Expected a base64 image URL')
                    source = {'type': 'base64', 'media_type': header[5:].split(';')[0], 'data': data}
                else:
                    source = {'type': 'url', 'url': url}
                blocks.append({'type': 'image', 'source': source})
            else:
                raise ValueError(f"Unsupported message part: {part['type']}")
        if message['role'] == 'system':
            if any(block['type'] != 'text' for block in blocks):
                raise ValueError('Anthropic system messages must be text')
            system.extend(blocks)
        else:
            role = 'assistant' if message['role'] == 'assistant' else 'user'
            if conversation and conversation[-1]['role'] == role:
                conversation[-1]['content'].extend(blocks)
            else:
                conversation.append({'role': role, 'content': blocks})
    return system, conversation


class ChatAPI:
    def __init__(self, *, base_url, api_key, model, api_format='openai', token_parameter='auto',
                 sampling_parameters='auto', timeout=180, retries=3):
        if api_format not in {'openai', 'anthropic'}:
            raise ValueError('api_format must be openai or anthropic')
        if token_parameter not in {'auto', 'max_tokens', 'max_completion_tokens'}:
            raise ValueError('Unsupported token parameter')
        if sampling_parameters not in {'auto', 'on', 'off'}:
            raise ValueError('sampling_parameters must be auto, on or off')
        self.model, self.api_format = model, api_format
        reasoning = model.lower().split('/')[-1].startswith(('gpt-5', 'o1', 'o3', 'o4'))
        self.token_parameter = ('max_completion_tokens' if reasoning else 'max_tokens') if token_parameter == 'auto' else token_parameter
        self.sampling = (not reasoning) if sampling_parameters == 'auto' else sampling_parameters == 'on'
        self.extra = json.loads(os.environ.get('ARIS_STUDENT_EXTRA_BODY', '{}'))
        if not isinstance(self.extra, dict):
            raise ValueError('ARIS_STUDENT_EXTRA_BODY must be a JSON object')
        reserved = {'model', 'messages', 'system', 'max_tokens', 'max_completion_tokens', 'stream'}
        if reserved & self.extra.keys():
            raise ValueError('Reserved fields in ARIS_STUDENT_EXTRA_BODY')
        if api_format == 'anthropic':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries - 1)
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or 'EMPTY', base_url=base_url, timeout=timeout, max_retries=retries - 1)

    def close(self):
        self.client.close()

    def chat(self, messages, *, temperature, max_tokens, seed=None):
        from evaluation.student_agent import Generation
        options = {'temperature': temperature} if self.sampling else {}
        if self.api_format == 'anthropic':
            system, conversation = anthropic_messages(messages)
            response = self.client.messages.create(model=self.model, system=system, messages=conversation,
                max_tokens=max_tokens, extra_body=self.extra, **options)
            text = ''.join(block.text for block in response.content if block.type == 'text')
            reason = response.stop_reason
            finish = 'length' if reason == 'max_tokens' else 'stop'
            if reason in {'refusal', 'pause_turn'}:
                finish = 'error'
        else:
            if self.sampling:
                options['top_p'] = 1.0
            if seed is not None:
                options['seed'] = seed
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                **{self.token_parameter: max_tokens}, extra_body=self.extra, **options)
            if not response.choices:
                raise RuntimeError('Student returned no choices')
            choice = response.choices[0]
            text, finish = choice.message.content, choice.finish_reason or 'stop'
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Student returned no visible text')
        return Generation(text, finish)
