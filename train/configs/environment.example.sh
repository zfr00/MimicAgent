# Copy outside version control and fill in actual local paths and credentials.
export ARIS_REPO_DIR=/path/to/公开代码demo
export HF_CHECKPOINT=/path/to/Qwen3-VL-8B-Instruct
export TRAIN_DATA="${ARIS_REPO_DIR}/data/train/aris_train.jsonl"
export OUTPUT_DIR=/path/to/aris_checkpoints
export NUM_GPUS=8
export NUM_ROLLOUT=1000

# Standalone evaluation API.
export STUDENT_MODEL=Qwen3-VL-8B-Instruct
export ARIS_STUDENT_BASE_URL=https://your-model-provider.example/v1
export ARIS_STUDENT_API_KEY=REPLACE_ME
export ARIS_STUDENT_API_FORMAT=openai
# Optional local Qwen3-VL tokenizer/processor directory for reference counting.
# export ARIS_EVAL_TOKENIZER=/path/to/Qwen3-VL-tokenizer
# export ARIS_STUDENT_EXTRA_BODY='{"reasoning_effort":"none"}'
export EVAL_DATA="${ARIS_REPO_DIR}/data/bench/aris_eval.jsonl"
export EVAL_OUTPUT=/path/to/evaluation/results.jsonl
# export SCEMQA_DATA="${ARIS_REPO_DIR}/data/ScEMQA/ScEMQA_EN.jsonl"

# All four roles may use separate OpenAI-compatible providers.
export ARIS_TEACHER_MODEL=gpt-5.2
export ARIS_TEACHER_BASE_URL=https://your-teacher-provider.example/v1
export ARIS_TEACHER_API_KEY=REPLACE_ME
export ARIS_TEACHER_TOKEN_PARAMETER=max_completion_tokens
# For endpoints that support it, configure the reasoning budget explicitly:
# export ARIS_TEACHER_EXTRA_BODY='{"reasoning_effort":"none"}'
export ARIS_PRIMARY_MODEL=gpt-4.1-mini
export ARIS_PRIMARY_BASE_URL=https://your-primary-provider.example/v1
export ARIS_PRIMARY_API_KEY=REPLACE_ME
export ARIS_SECONDARY_MODEL=gemini-3-flash
export ARIS_SECONDARY_BASE_URL=https://your-secondary-provider.example/v1
export ARIS_SECONDARY_API_KEY=REPLACE_ME
export ARIS_ANALOGY_MODEL=gemini-3-flash
export ARIS_ANALOGY_BASE_URL=https://your-analogy-provider.example/v1
export ARIS_ANALOGY_API_KEY=REPLACE_ME

# English exemplar replies; credentials may fall back to ARIS_PRIMARY_*.
export ARIS_TRANSLATOR_MODEL=gpt-4.1-mini
# export ARIS_TRANSLATOR_BASE_URL=https://your-translator-provider.example/v1
# export ARIS_TRANSLATOR_API_KEY=REPLACE_ME
export EXEMPLAR_TRANSLATION_CACHE_DIR="${ARIS_REPO_DIR}/cache/exemplar_translations"
export EXEMPLAR_TRANSLATION_RETRIES=3
export EXEMPLAR_TRANSLATION_TIMEOUT=60
export EXEMPLAR_TRANSLATION_MAX_TOKENS=8192

# Set these BEFORE starting ../tools/start_services_demo.sh.
export CURATORIAL_REPO_PATH="${ARIS_REPO_DIR}/data/exemplars/curatorial_repo.jsonl"
export EXEMPLAR_RECALL_K=3
export SERPER_API_KEY=REPLACE_ME
# export JINA_API_KEY=REPLACE_ME
export ARIS_TOOL_API_KEY=REPLACE_WITH_YOUR_TOOL_SERVICE_KEY
export TOOL_MAX_CONCURRENCY=128
export FATHOM_BASE_URL=http://127.0.0.1:8904
export WEB_LLM_MODEL=gpt-4.1-mini
# WEB_LLM_BASE_URL / WEB_LLM_API_KEY default to ARIS_PRIMARY_*.
export WEB_MAX_TOKENS=3000
export KG_BASE_URL=http://127.0.0.1:10087
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=REPLACE_ME
export NEO4J_DATABASE=wiki-csv-json-text
export KG_LLM_MODEL=gpt-4.1-mini
# KG_LLM_BASE_URL / KG_LLM_API_KEY default to ARIS_PRIMARY_*.
export CODE_EXECUTION_TIMEOUT=360
export CODE_MAX_OUTPUT_BYTES=32768
# Training workers and tool services must see the same image paths.
# For remote services override any/all endpoints as a JSON map:
# export ARIS_TOOL_ENDPOINTS='{"kg_knowledge":"http://tool-host:8000/kg_knowledge"}'
