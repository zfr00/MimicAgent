"""Convert SciExplore JSONL into training records."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.data_records import question_language, question_subject


def convert_record(item, images_root, source=None):
    target_id = str(item.get("id") or "")
    if not target_id or not isinstance(item.get("question"), str) or "answer" not in item:
        raise ValueError("Every record needs nonempty id, question and answer")
    paths = []
    for name in (item.get("image") or {}).get("question", []) or []:
        path = (images_root / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    return {
        "messages": [{"role": "user", "content": item["question"]}],
        "label": str(item["answer"]),
        "metadata": {"target_id": target_id, "question": item["question"], "image_paths": paths,
                     "subject": question_subject(item, source),
                     "language": question_language(item["question"], source)},
    }


def prepare(inputs, images_root, output):
    records, ids = [], set()
    for input_path in inputs:
        with Path(input_path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = convert_record(json.loads(line), Path(images_root), input_path)
                tid = record["metadata"]["target_id"]
                key = (tid, record["metadata"]["language"])
                if key in ids:
                    raise ValueError(f"Duplicate target id {tid} at {input_path}:{number}")
                ids.add(key)
                records.append(record)
    if not records:
        raise ValueError("Empty dataset")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(args.input, args.images_root, args.output)
    print(f"Wrote {count} records to {args.output}.")


if __name__ == "__main__":
    main()
