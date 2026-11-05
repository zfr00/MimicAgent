"""Evaluate SciExplore with repeated independent runs and the shared ARIS judge."""

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / 'train'):
    sys.path.insert(0, str(path))

from aris.clients import Endpoint, ModelAPIs
from aris.config import EXPERIMENTS, load_config
from evaluation.extract_answer import extract_final_answer
from evaluation.llm_judge import judge_two_stage_async, verdict_to_score
from evaluation.student_agent import load_codec, run_student_async, trajectory_to_dict_list
from tools.data_records import question_language, question_subject
from tools.input_validation import validate_images
from tools.tool_errors import ExemplarPrefixError
from evaluation.model_api import ChatAPI


def resolve_image_paths(item, images_root=None):
    image_root = Path(images_root) if images_root else REPO_ROOT / 'data/bench/images'
    names = ((item.get('image') or {}).get('question') or [])
    if 'metadata' in item and 'image_paths' in item['metadata']:
        names = item['metadata']['image_paths']
    paths = []
    for name in names:
        path = (image_root / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Question image not found: {path}')
        paths.append(str(path))
    return paths


def subject_of(item, path):
    return question_subject(item, path)


def load_records(paths, images_root, max_samples=0):
    records, ids = [], set()
    checked_images = set()
    counts = defaultdict(int)
    for path in map(Path, paths):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            subject = subject_of(item, path)
            metadata = item.get('metadata') or {}
            target_id = str(item.get('id') or metadata.get('target_id') or '')
            question = item.get('question', metadata.get('question'))
            reference = item.get('answer', item.get('label'))
            if not target_id or not isinstance(question, str) or reference is None:
                raise ValueError(f'{path}: each record requires id, question and answer')
            language = question_language(question, path, metadata.get('language'))
            bucket = language, subject
            if max_samples and counts[bucket] >= max_samples:
                continue
            key = (target_id, language)
            if key in ids:
                raise ValueError(f'Duplicate evaluation target ID: {target_id}')
            ids.add(key)
            image_paths = resolve_image_paths(item, images_root)
            validate_images(question, image_paths, label=f'{target_id}.question', checked=checked_images)
            records.append({'id': target_id, 'question': question, 'reference': str(reference),
                            'subject': subject, 'language': language,
                            'image_paths': image_paths})
            counts[bucket] += 1
    if not records:
        raise ValueError('Empty evaluation dataset')
    return records


def summarize(rows, runs):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row['run']].append(row)
    if set(by_run) != set(range(1, runs + 1)):
        raise ValueError('Missing evaluation runs')
    def identity(row):
        return row['id'], row.get('language', '')
    expected = {identity(r) for r in by_run[1]}
    if any(len(group) != len(expected) or {identity(r) for r in group} != expected for group in by_run.values()):
        raise ValueError('Each evaluation run must cover the same unique target IDs')
    details = []
    def language_of(row):
        return row.get('language') or 'unknown'

    def accuracy(group):
        return sum(row['score'] for row in group) / len(group)

    def subject_accuracy(group):
        return {subject: accuracy([row for row in group if row['subject'] == subject])
                for subject in sorted({row['subject'] for row in group})}

    for run_id in range(1, runs + 1):
        group = by_run[run_id]
        languages = {language: [row for row in group if language_of(row) == language]
                     for language in sorted({language_of(row) for row in group})}
        details.append({'run': run_id, 'count': len(group),
                        'accuracy': accuracy(group), 'subject_accuracy': subject_accuracy(group),
                        'language_counts': {language: len(rows) for language, rows in languages.items()},
                        'language_accuracy': {language: accuracy(rows) for language, rows in languages.items()},
                        'language_subject_accuracy': {
                            language: subject_accuracy(rows) for language, rows in languages.items()}})
    return {'runs': runs, 'questions': len(expected), 'per_run': details,
            'mean_accuracy': sum(r['accuracy'] for r in details) / runs,
            'mean_subject_accuracy': {
                s: sum(r['subject_accuracy'][s] for r in details) / runs for s in details[0]['subject_accuracy']},
            'mean_language_accuracy': {
                language: sum(r['language_accuracy'][language] for r in details) / runs
                for language in details[0]['language_accuracy']},
            'mean_language_subject_accuracy': {
                language: {subject: sum(r['language_subject_accuracy'][language][subject] for r in details) / runs
                           for subject in subjects}
                for language, subjects in details[0]['language_subject_accuracy'].items()}}


async def evaluate_records(records, config, *, student, apis, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix('.summary.json')
    summary_path.unlink(missing_ok=True)
    rows = []
    with output.open('w', encoding='utf-8') as stream:
        for run_id in range(1, config.eval_runs + 1):
            for record in records:
                try:
                    result = await student(record, run_id)
                except ExemplarPrefixError as exc:
                    stream.write(json.dumps({'id': record['id'], 'run': run_id, 'subject': record['subject'],
                        'language': record.get('language', question_language(record['question'])),
                        'experiment': config.experiment, 'prefix_k': config.prefix_k,
                        'stopped_reason': 'prefix_error', 'error': exc.result}, ensure_ascii=False) + '\n')
                    stream.flush()
                    raise
                prediction = extract_final_answer(result.final_reply)
                verdict = await judge_two_stage_async(
                    question=record['question'], ground_truth=record['reference'],
                    model_answer=prediction, config=config, apis=apis,
                )
                row = {'id': record['id'], 'run': run_id, 'subject': record['subject'],
                       'experiment': config.experiment,
                       'language': record.get('language', question_language(record['question'])),
                       'ground_truth': record['reference'], 'prediction': prediction,
                       'verdict': verdict, 'score': verdict_to_score(verdict),
                       'stopped_reason': result.stopped_reason, 'rounds_used': result.rounds_used,
                       'trajectory': trajectory_to_dict_list(result.trajectory), 'episode': result.episode}
                if config.uses_exemplar_prefix:
                    row['prefix_k'] = config.prefix_k
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                print(f"[run {run_id}/{config.eval_runs}] {record['id']}: {verdict}", flush=True)
    summary = summarize(rows, config.eval_runs)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, action='append')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--images-root', type=Path, default=REPO_ROOT / 'data/bench/images')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--experiment', choices=EXPERIMENTS)
    parser.add_argument('--prefix-k', type=int, choices=range(1, 5), help='Number of static prefix exemplars (default: 3)')
    presets = json.loads((REPO_ROOT / 'evaluation/models.json').read_text())
    parser.add_argument('--model-preset', choices=presets)
    parser.add_argument('--reference-tokenizer', '--hf-checkpoint', dest='reference_tokenizer',
                        default=os.environ.get('ARIS_EVAL_TOKENIZER'))
    parser.add_argument('--student-base-url', default=os.environ.get('ARIS_STUDENT_BASE_URL', os.environ.get('OPENAI_BASE_URL')))
    parser.add_argument('--student-api-key', default=os.environ.get('ARIS_STUDENT_API_KEY', os.environ.get('OPENAI_API_KEY', 'EMPTY')))
    parser.add_argument('--student-model')
    parser.add_argument('--api-format', choices=('openai', 'anthropic'))
    parser.add_argument('--token-parameter', choices=('auto', 'max_tokens', 'max_completion_tokens'),
                        default=os.environ.get('ARIS_STUDENT_TOKEN_PARAMETER', 'auto'))
    parser.add_argument('--sampling-parameters', choices=('auto', 'on', 'off'),
                        default=os.environ.get('ARIS_STUDENT_SAMPLING_PARAMETERS', 'auto'))
    parser.add_argument('--judge-base-url')
    parser.add_argument('--judge-api-key', default=os.environ.get('JUDGE_API_KEY'))
    for role, default in (('primary', 'gpt-4.1-mini'), ('secondary', 'gemini-3-flash')):
        parser.add_argument(f'--judge-{role}-model', default=os.environ.get(f'ARIS_{role.upper()}_MODEL', default))
        parser.add_argument(f'--judge-{role}-base-url')
        parser.add_argument(f'--judge-{role}-api-key')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--judge-repeats', type=int)
    parser.add_argument('--max-tool-rounds', type=int)
    parser.add_argument('--max-context-tokens', type=int)
    parser.add_argument('--max-samples-per-subject', type=int, default=0,
                        help='Maximum questions per language and subject (0 = all)')
    parser.add_argument('--seed', type=int, help='Optional base seed; each run uses a distinct seed')
    args = parser.parse_args(argv)
    preset = presets.get(args.model_preset, {})
    args.student_model = args.student_model or preset.get('model') or os.environ.get(
        'ARIS_STUDENT_MODEL', os.environ.get('STUDENT_MODEL', 'Qwen3-VL-8B-Instruct'))
    args.api_format = args.api_format or preset.get('api_format') or os.environ.get('ARIS_STUDENT_API_FORMAT', 'openai')
    args.student_base_url = args.student_base_url or preset.get('base_url') or (
        'https://api.anthropic.com' if args.api_format == 'anthropic' else 'http://127.0.0.1:30000/v1')
    if args.max_samples_per_subject < 0:
        parser.error('--max-samples-per-subject must be nonnegative')
    return args


async def main_async(args):
    config = load_config(args.config)
    names = {'temperature': 'eval_temperature', 'runs': 'eval_runs', 'judge_repeats': 'judge_repeats',
             'max_tool_rounds': 'max_tool_rounds', 'max_context_tokens': 'eval_max_context_tokens',
             'experiment': 'experiment', 'prefix_k': 'prefix_k'}
    config = replace(config, **{field: getattr(args, name) for name, field in names.items()
                               if getattr(args, name) is not None}).validate()
    records = load_records(args.input, args.images_root, args.max_samples_per_subject)
    codec = load_codec(args.reference_tokenizer, config)
    endpoints = {}
    for role in ('primary', 'secondary'):
        ep = Endpoint.from_env(role, getattr(args, f'judge_{role}_model'))
        endpoints[role] = replace(ep, model=getattr(args, f'judge_{role}_model'),
                                  base_url=getattr(args, f'judge_{role}_base_url') or args.judge_base_url or ep.base_url,
                                  api_key=getattr(args, f'judge_{role}_api_key') or args.judge_api_key or ep.api_key)
    apis = ModelAPIs(config, roles=('primary', 'secondary'), endpoints=endpoints)
    client = None

    async def student(record, run_id):
        return await run_student_async(
            question=record['question'], image_paths=record['image_paths'], target_id=record['id'],
            base_url=args.student_base_url, api_key=args.student_api_key, model=args.student_model,
            config=config, codec=codec, seed=None if args.seed is None else args.seed + run_id - 1,
            target_language=record['language'],
            client=client,
        )
    try:
        client = ChatAPI(base_url=args.student_base_url, api_key=args.student_api_key, model=args.student_model,
                         api_format=args.api_format, token_parameter=args.token_parameter,
                         sampling_parameters=args.sampling_parameters, timeout=config.api_timeout, retries=config.api_retries)
        summary = await evaluate_records(records, config, student=student, apis=apis, output=args.output)
    finally:
        if client is not None:
            client.close()
        await apis.close()
    for subject, accuracy in summary['mean_subject_accuracy'].items():
        print(f'{subject}: {accuracy:.2%}')
    for language, subjects in summary['mean_language_subject_accuracy'].items():
        for subject, accuracy in subjects.items():
            print(f'{language}/{subject}: {accuracy:.2%}')
        print(f"{language} mean: {summary['mean_language_accuracy'][language]:.2%}")
    print(f"Mean accuracy ({config.eval_runs} runs): {summary['mean_accuracy']:.2%}")


if __name__ == '__main__':
    asyncio.run(main_async(parse_args()))
