from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trajectory_generation.teacher_agent import run_teacher  # noqa: E402
from evaluation.student_agent import load_codec
from tools.data_records import question_language


def resolve_image_paths(item: Dict[str, Any], images_root=None) -> List[str]:
    image_root = Path(images_root) if images_root else REPO_ROOT / "data/train/images"
    out: List[str] = []
    for name in (item.get("image") or {}).get("question", []) or []:
        p = image_root / name
        if not p.is_file():
            raise FileNotFoundError(f"Question image not found: {p}")
        out.append(str(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher demonstration trajectories.")
    parser.add_argument("--input", required=True, help="path to SciExplore-Bench JSONL slice")
    parser.add_argument("--output", required=True, help="path to write trajectory JSONL")
    parser.add_argument("--base-url", default=os.environ.get("ARIS_TEACHER_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    parser.add_argument("--api-key", default=os.environ.get("ARIS_TEACHER_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--model", default=os.environ.get("ARIS_TEACHER_MODEL", "gpt-5.2"))
    parser.add_argument("--images-root", type=Path, default=REPO_ROOT / "data/train/images")
    parser.add_argument("--max-tool-rounds", type=int, default=6)
    parser.add_argument("--hf-checkpoint", default=os.environ.get("HF_CHECKPOINT"),
                        help="Student tokenizer/processor for the shared trajectory token budget")
    parser.add_argument("--temperature", type=float, help="Optional provider-supported temperature")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = run all")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("ERROR: --api-key not provided and OPENAI_API_KEY is unset")

    codec = load_codec(args.hf_checkpoint)

    input_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if args.max_samples and written >= args.max_samples:
                break

            images = resolve_image_paths(item, args.images_root)
            res = run_teacher(
                question=item["question"],
                ground_truth=str(item.get("answer", "")),
                image_paths=images,
                target_id=str(item.get("id", "")) or None,
                target_language=question_language(item["question"], input_path),
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_tool_rounds=args.max_tool_rounds,
                hf_checkpoint=args.hf_checkpoint,
                codec=codec,
            )

            record = {
                "id": item.get("id"),
                "question": item["question"],
                "ground_truth": item.get("answer"),
                "image_paths": images,
                "trajectory": [asdict(s) for s in res.trajectory],
                "episode": res.episode,
                "final_answer": res.final_answer,
                "rounds_used": res.rounds_used,
                "stopped_reason": res.stopped_reason,
                "model": args.model,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  wrote trajectory for id={item.get('id')} stop={res.stopped_reason} rounds={res.rounds_used}")

    print(f"done: {written} trajectories -> {out_path}")


if __name__ == "__main__":
    main()
