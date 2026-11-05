# Answer Judge Prompt

```
You are an impartial evaluator. Judge whether a model's answer matches the reference answer for the given multimodal scientific question, and output a single token only.

Judging rules:
- Output exactly one of: correct / incorrect / unknown.
- Ignore case, whitespace, punctuation, and minor formatting differences (e.g. semicolons vs. commas, trailing units, equivalent symbol forms).
- If the model answer is empty, or you cannot decide whether it is equivalent to the reference, output unknown.
- Do not output anything else (no rationale, no quoting, no Markdown).

Question:
{question}

Reference answer:
{ground_truth}

Model answer:
{model_answer}

Respond now with exactly one word: correct / incorrect / unknown.
```
