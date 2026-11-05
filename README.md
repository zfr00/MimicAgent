# MimicAgent / ARIS

This repository contains ARIS training for MimicAgent, SciExplore-Bench evaluation, scripts for tool and exemplar ablations, and services for five types of tools. Training builds on [THUDM/slime](https://github.com/THUDM/slime) and uses Qwen3-VL-8B-Instruct, SGLang for rollouts, and Megatron Bridge.

## Repository Layout

```text
data/
  train/                     Training questions and images
  bench/                     SciExplore-Bench questions and images
  ScEMQA/                    SceMQA free-response evaluation data for three subjects
  exemplars/                 Curated exemplars paired with training and evaluation questions
cache/exemplar_translations/ Cached English translations of exemplars
train/                       ARIS training implementation, slime, configuration, and launch scripts
evaluation/                  Model APIs, dual-judge scoring, and five-run evaluation
experiments/                 Entry points for tool comparisons and three core ablations
tools/                       HTTP services for five types of tools
prompts/                     Student, teacher, and answer judge prompts
trajectory_generation/       Standalone teacher trajectory generation
```

## Downloading and Organizing Data

The ARIS training set, SciExplore-Bench evaluation set, and curated exemplar repository are bundled in one archive. Extract it into `data/` to create `train/`, `bench/`, and `exemplars/`. The exemplars and their reference solutions are paired by target question ID.

| Dataset | Download | Location |
|---|---|---|
| ARIS archive (training set, SciExplore-Bench evaluation set, and curated exemplars) |  | `data/` |
| SceMQA evaluation subset (150 questions) | [Official download](https://huggingface.co/datasets/Haozy/SceMQA-main/tree/main) | `data/ScEMQA/` |

The SceMQA evaluation uses all 150 questions in the `Physics_Free_Response`, `Chemistry_Free_Response`, and `Biology_Free_Response` subsets of the official `Free_Response/free_response.json`, with 50 questions per subject. Place the questions and their images in `data/ScEMQA/`. See the [official SceMQA project](https://github.com/SceMQA/SceMQA) for the source data.

The training and benchmark data retain their original file layout and schema. After extraction, the directories have the following structure.

```text
data/
  train/
    physics_cn.jsonl
    physics_en.jsonl
    chemistry_cn.jsonl
    chemistry_en.jsonl
    biology_cn.jsonl
    biology_en.jsonl
    images/
  bench/
    CN_240.jsonl
    EN_240.jsonl
    images/
  exemplars/
    curatorial_repo.jsonl
    images/
```

### Question Format and Conversion

The training and evaluation sets use JSONL with one question per line.

| Field | Description |
|---|---|
| `id` | Original question ID, which matches `target_id` in the exemplar repository. Chinese and English versions may share the same ID. |
| `question` | Question text, which may contain `<image>` placeholders |
| `answer` | Reference answer |
| `image.question` | List of image filenames relative to the dataset's `images/` directory |
| `image.answer`, `image.analysis` | Original answer and solution images |
| `analysis` | Original solution explanation, which may be empty |
| `knowledge` | Topic or concept as a string |
| `year`, `difficulty_level`, `question_type` | Original question metadata |
| `grade_level` | Grade level of a training question |

The student receives `question` and `image.question` from the target question. The teacher and answer judges use the text reference answer. The code detects Chinese or English from the filename and determines the subject from the filename or the ID prefix. When a question is passed directly, language detection uses its text. At runtime, `(id, language)` distinguishes records that share an ID, while exemplar recall uses the original ID.

Convert the training questions to ARIS input with the following command.

```bash
python train/aris/prepare_data.py \
  --input data/train/physics_cn.jsonl \
  --input data/train/physics_en.jsonl \
  --input data/train/chemistry_cn.jsonl \
  --input data/train/chemistry_en.jsonl \
  --input data/train/biology_cn.jsonl \
  --input data/train/biology_en.jsonl \
  --images-root data/train/images \
  --output data/train/aris_train.jsonl
```

Use the same script to combine and convert the evaluation questions.

```bash
python train/aris/prepare_data.py \
  --input data/bench/CN_240.jsonl \
  --input data/bench/EN_240.jsonl \
  --images-root data/bench/images \
  --output data/bench/aris_eval.jsonl
```

The output contains `messages`, `label`, and `metadata`. The reference answer goes in `label`. The `metadata` field records the original question ID, question, language, subject, and absolute paths to the question images. Run the conversion on the machine that will use the data and make the same image paths available to the training workers and tool services.

### Curated Exemplar Format

The exemplar repository consists of a single Chinese-language `curatorial_repo.jsonl` file and an `images/` directory. Each line corresponds to one original `target_id` and contains four ordered `companions`. The Chinese and English versions of the target question share that entry. The four exemplars have IDs `<target_id>_01`, `<target_id>_02`, `<target_id>_03`, and `<target_id>_04` in array order. For `target_id` equal to `hs_phy_0003`, the first exemplar has the following structure.

```json
{
  "id": "hs_phy_0003_01",
  "question": "Chinese exemplar question <image>",
  "answer": "Chinese reference answer, optionally with <image>",
  "analysis": "Chinese solution explanation, optionally with <image>",
  "knowledge": "Topic or concept",
  "image": {
    "question": ["hs_phy_0003_01_01.png"],
    "answer": ["hs_phy_0003_01_02.png"],
    "analysis": ["hs_phy_0003_01_03.png"]
  }
}
```

Image filenames are relative to the `images/` directory next to the exemplar file. Images for each field follow the order of its `<image>` placeholders. Use `[]` for a field with no images.

If an exemplar has one image, name it `<exemplar_id>.<extension>`. If it has multiple images, number them consecutively across the question, answer, and solution, using `<exemplar_id>_01.<extension>`, `<exemplar_id>_02.<extension>`, and so on. Preserve the image order within each field. For example, the single image for `hs_phy_0003_02` is named `hs_phy_0003_02.png`. The references in `image.question`, `image.answer`, and `image.analysis` determine where each image belongs.

The recall service looks up the current question ID and returns the first three exemplars by default. Set `EXEMPLAR_RECALL_K` to a value from 1 to 4 before starting the service. The model calls the tool with `{"name":"exemplar_recall","arguments":{}}`, and the runner injects `target_id` and `target_language`. For static prefixes, `--prefix-k` controls the number of exemplars, with a default of 3 and a range of 1 to 4. The model and the analogy reward model receive the question, answer, and solution images as pixel content.

### Multimodal Model Inputs

The target question text and images appear in the same user message. The system message defines the role, tools, and answer rules. When building a message, the runner inserts image blocks at the corresponding `<image>` positions and preserves the surrounding text order. For a field with images and no placeholders, it appends the images to the end of that section. If placeholders are present, their count must match the number of images in the field.

Each exemplar displays its ID, question, answer, solution, and topic in that order, with images inserted in the corresponding sections. For English exemplars, the runner builds the same multimodal message after translating the text. Training, evaluation, teacher generation, static exemplar prefixes, and analogy scoring share this rendering logic.

Each trajectory keeps an image registry with indices starting at `image_index=0`. The model can use the displayed index to call `image_crop` on question images, retrieved exemplar images, and earlier crops. An image retains its index when shown again. The display includes the original dimensions. The crop tool uses the original pixels from the registry and interprets coordinates as normalized by default. Set `bbox_type=px` to use original pixel coordinates. The tool returns the source index, the actual crop region, and the cropped image.

Visual inputs are resized according to the configuration and encoded by the student processor. Image tokens count toward the context budget. The runner adds the text, images, and new image indices to the context only after the entire tool result passes the length check. If it exceeds the budget, the runner keeps the call record and raw result, sets `accepted=false`, and ends the trajectory.

### English Exemplar Translation and Caching

Chinese target questions retrieve the original Chinese exemplars. For an English target, the service sends the selected exemplars' `question`, `answer`, `analysis`, and `knowledge` fields to GPT-4.1-mini in a batch. The model translates Chinese annotations within formulas while preserving formula structure, values, units, and image placeholders. The service preserves the images, including any text within them, and retains the IDs and image references.

Translations are stored in `cache/exemplar_translations/`. Requests with identical exemplar content and translation settings reuse the same translation, including concurrent requests. A change to the content or settings creates a new cache entry. The source file stays in Chinese. Delete a cached file to trigger a new translation.

If translation requests or validation checks still fail after retries, the recall service returns the original Chinese text and images with the language set to `zh`. Training and evaluation continue. Only validated translations are cached. If writing to the cache fails, the service uses the current translation and logs the failure.

| Setting | Default |
|---|---|
| `ARIS_TRANSLATOR_MODEL` | `gpt-4.1-mini` |
| `ARIS_TRANSLATOR_BASE_URL` / `ARIS_TRANSLATOR_API_KEY` | Can be configured independently. Falls back to `ARIS_PRIMARY_*` and then `OPENAI_*`. |
| `EXEMPLAR_TRANSLATION_CACHE_DIR` | `cache/exemplar_translations/` within the repository |
| `EXEMPLAR_TRANSLATION_RETRIES` | 3 attempts |
| `EXEMPLAR_TRANSLATION_TIMEOUT` | 60 seconds per API request |
| `EXEMPLAR_TRANSLATION_MAX_TOKENS` | 8192 |


## Environment and Configuration

Install the dependencies for the tool services and standalone evaluation.

```bash
python -m pip install -r requirements.txt
```

Use Python 3.11 or newer. The Python tool runs in a Linux bubblewrap sandbox. Install the system package `bubblewrap` and enable user, filesystem, network, and process namespaces. Set `CODE_BWRAP` to specify the executable path.

The GPU training environment is built from the slime image and the [ARIS Dockerfile](train/Dockerfile). Training uses FlashAttention-2 2.8.3. The slime commit is pinned to `2a60f00dd27f716f8761464086f02b02061adff1` in [SLIME_COMMIT](train/SLIME_COMMIT). The three-line [metadata patch](train/patches/slime-aris.patch) passes the ARIS branch and weights to the training loss.

```bash
docker build -f train/slime/docker/Dockerfile \
  --build-arg SLIME_COMMIT="$(cat train/SLIME_COMMIT)" \
  -t mimicagent-slime:base train/slime
docker build -f train/Dockerfile -t mimicagent-slime:aris train
```

In the GPU environment, install the vendored slime package and the ARIS dependencies.

```bash
python -m pip install -e ./train/slime --no-deps
python -m pip install -r train/requirements-aris.txt
bash train/apply_slime_patch.sh
```

Copy the [example environment configuration](train/configs/environment.example.sh) to `environment.local.sh`. Fill in your paths, service URLs, and API keys, then load it.

```bash
source environment.local.sh
```

| Setting | Purpose or Default |
|---|---|
| `HF_CHECKPOINT` | Hugging Face directory containing the student model, tokenizer, and processor |
| `TRAIN_DATA` | `data/train/aris_train.jsonl` |
| `EVAL_DATA` | `data/bench/aris_eval.jsonl` |
| `CURATORIAL_REPO_PATH` | `data/exemplars/curatorial_repo.jsonl` |
| `OUTPUT_DIR` | Training checkpoint directory, defaults to `train/outputs/aris` |
| `EVAL_OUTPUT` | Evaluation output, defaults to `evaluation/output/results.jsonl` |
| `ARIS_STUDENT_*` | Model API URL, key, and protocol for standalone evaluation |
| `ARIS_EVAL_TOKENIZER` | Reference Qwen3-VL tokenizer and processor for standalone evaluation |
| `ARIS_TEACHER_*` | Online teacher, defaults to gpt-5.2 |
| `ARIS_PRIMARY_*` | Primary judge, defaults to gpt-4.1-mini |
| `ARIS_SECONDARY_*` | Secondary judge, defaults to gemini-3-flash |
| `ARIS_ANALOGY_*` | Analogy reward model, defaults to gemini-3-flash |
| `ARIS_TRANSLATOR_*` | Exemplar translator, defaults to gpt-4.1-mini |

Each model role supports `MODEL`, `BASE_URL`, and `API_KEY`. Roles can share `OPENAI_BASE_URL` and `OPENAI_API_KEY`. `TOKEN_PARAMETER` accepts `max_tokens` or `max_completion_tokens`, and `EXTRA_BODY` takes service-specific JSON parameters. The teacher uses `max_completion_tokens` by default. Set translation options before starting the tool services.

## Tool Services

One FastAPI executor serves all five tool types at `http://127.0.0.1:8000` by default. Requests use bearer authentication with `ARIS_TOOL_API_KEY`. The default concurrency limit is 128 requests (`TOOL_MAX_CONCURRENCY`). Install the tool dependencies and Chromium for Fathom's browser fallback.

```bash
python -m pip install -r tools/requirements.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/cache/browsers" python -m playwright install chromium
```

| Tool | HTTP Route | Default Port | Configuration |
|---|---|---|---|
| `exemplar_recall` | `POST /exemplar_recall` | 8000 | Curated exemplar repository path |
| `web_search.search` | `POST /search` | 8000 | `SERPER_API_KEY` and the Fathom backend |
| `web_search.query` | `POST /query` | 8000 | `WEB_LLM_*` and the Fathom backend |
| `kg_knowledge` | `POST /kg_knowledge` | 8000 | Neo4j and `KG_LLM_*` |
| `image_crop` | `POST /image_crop` | 8000 | The runner supplies the original pixels of the image selected by its session index |
| `code_interpreter` | `POST /code_interpreter` | 8000 | Python scientific computing dependencies |

`web_search` uses `sandbox_serper.py` from [Fathom-DeepResearch](https://github.com/FractalAIResearchLabs/Fathom-DeepResearch), vendored in `tools/vendor/fathom/`. The pinned commit is recorded in `tools/FATHOM_COMMIT`. Search calls `search_urls` (Serper), while page queries call `query_url` to fetch a page and extract information relevant to the query with an LLM. Returned JSON is capped at 3,000 tokens by default (`WEB_MAX_TOKENS`), using `cl100k_base`. Each backend request times out after 350 seconds, and the outer tool call has a 360-second budget. Page extraction defaults to `WEB_LLM_MODEL=gpt-4.1-mini`. Set `WEB_LLM_BASE_URL` and `WEB_LLM_API_KEY` to override the inherited `ARIS_PRIMARY_*` settings. The deployment patch at `tools/patches/fathom.patch` handles LLM timeouts and model names, lazy loading for PDF and browser fetching modules, and sentence segmentation when the English spaCy model is unavailable. The vendored code includes the upstream MIT license.

Serper uses `https://google.serper.dev/search` with `SERPER_API_KEY`. Results are cached by query and `top_k` in `cache/web_search/serper/`. `SEARCH_CACHE_TTL` sets the cache lifetime in seconds, with the default of 0 allowing entries to persist indefinitely. Jina Reader uses `https://r.jina.ai/` with `JINA_API_KEY` and caches page content by normalized URL in `cache/web_search/jina/jina_read/`. Page fetching selects Jina, direct fetching, or a browser based on the domain and reuses cached content when available.

The knowledge graph backend is included in `tools/kg_backend.py`. The [knowledge graph guide](knowledge_graph.md) covers the ATLAS-Wiki download, Neo4j startup and indexing, the bilingual filtering API, and launch commands.

```bash
bash tools/start_services_demo.sh
bash tools/start_services_demo.sh --stop
```

Logs and process records are stored in `tools/logs/`. The launch script starts and stops services by process group and cleans up the services it started if startup fails. Configure remote services with the `ARIS_TOOL_ENDPOINTS` JSON mapping. The unified executor runs each request in its own process and defaults to `TOOL_EXECUTION_TIMEOUT` of 360 seconds. It passes the deadline to the Fathom and knowledge graph backends and terminates the corresponding process group if the request times out, the caller disconnects, or the request is canceled. Ordinary tool errors are returned to the model so it can continue.

The script starts local Fathom (8904), the knowledge graph backend (10087), and the unified executor (8000) by default. Start Neo4j first by following the knowledge graph guide. To use remote backends, set `START_FATHOM=0` or `START_KG_BACKEND=0` as applicable and configure `FATHOM_BASE_URL` or `KG_BASE_URL`. You can also start the backends individually with `tools/start_fathom.sh` and `tools/start_kg.sh`.

`image_crop` crops in original image coordinates and uses Lanczos resampling. It can upscale by at most 2× and defaults to a limit of 262,144 pixels (`CROP_MAX_PIXELS`). The student and teacher receive the resulting image.

The Python tool starts a separate sandbox for every request and preloads NumPy, SciPy, SymPy, Matplotlib, and pandas. Defaults are 360 seconds for `CODE_EXECUTION_TIMEOUT`, 32,768 bytes for `CODE_MAX_OUTPUT_BYTES`, and 4 GiB for `CODE_MEMORY_BYTES`. If a limit is exceeded, the tool terminates the process and returns an error.

## ARIS Training

After converting the data, configuring the environment, and starting the tool services, run the training script.

```bash
bash train/run_aris.sh --dry-run
bash train/run_aris.sh --steps 1000
```

Training reads `data/train/aris_train.jsonl` by default. Override it with `TRAIN_DATA` or `--train-data`. The default is one node with 8 GPUs, TP=2, and PP=2. Training and rollouts share the GPUs. Pass `--ray-address auto` to use an existing Ray cluster. Configure parallelism with `--num-gpus`, `--tp`, `--pp`, and `--rollout-gpus-per-engine`.

```bash
# Resume training
bash train/run_aris.sh --resume "$OUTPUT_DIR" --steps 1000

# Evaluate during training
bash train/run_aris.sh --eval-data data/bench/aris_eval.jsonl --eval-interval 20
```

`--resume` reads `aris_config.json` from the checkpoint directory and restores the experiment mode, reward weights, interaction budgets, and other ARIS settings. If you specify an `iter_XXXXXXX` subdirectory, the runner reads the configuration and corresponding data progress from the parent directory, loads the selected iteration's weights and optimizer state, and continues from the next rollout. Pass `--config` to use a different configuration file. Explicit flags such as `--experiment` and `--group-size` still override individual settings. The resumed configuration takes precedence over the `ARIS_CONFIG` environment variable.

`--eval-data` enables evaluation before training and every `--eval-interval` steps. The current student answers questions in the specified evaluation set and receives answer scores under the training interaction budget. Questions whose initial input exceeds the budget are recorded as `context_limit`, assigned an answer score of 0, and included in the evaluation count. Training resumes after evaluation.

By default, each step uses B=8 questions and G=8 student trajectories per question. A group with at least one correct answer uses GRPO. For a group with all incorrect answers, the online teacher generates one trajectory and training uses imitation loss with λ=10. The branches are mutually exclusive for each question, and the loss is averaged across questions.

Student training retains the token IDs and log probabilities returned by SGLang. Teacher trajectories are re-encoded using the student prompt. GRPO uses a sampling temperature of 0.8, and teacher NLL uses a temperature of 1.

| Parameter | Default |
|---|---|
| Prompt / response / context token limits | 3600 / 7680 / 10240 |
| Student sampling temperature / maximum tool rounds | 0.8 / 6 |
| GRPO clipping lower / upper bound | 0.20 / 0.28 |
| KL / entropy coefficients | 0 / 0 |
| Adam learning rate / betas | Constant 1e-6 / (0.9, 0.95) |
| Weight decay / gradient clipping | 0.1 / 1.0 |
| Activation recomputation | full + uniform |
| SGLang memory fraction / concurrency | 0.7 / 160 |

The six-round limit allows up to six tool interactions, with at most one tool call per round. The model can still generate a final answer after the sixth tool result. The 3,600-token limit applies to the base input, which includes the system prompt, question text, and question images. The static exemplar prefix and base input both count toward the 10,240-token context budget. The 7,680-token limit applies to the full trajectory after the complete initial input, including assistant output, tool results, image tokens, and message separator tokens. Both the trajectory and total context limits apply.

Teacher imitation uses the complete multi-turn interaction. `aris_episode` stores the student's initial prompt and images, assistant outputs from each turn, tool names and arguments, raw tool results, the text and image observations shown to the model, and the image registry with original pixels and source information. Replay reuses the saved observations and indices. The student tokenizer and processor re-encode the full trajectory. The imitation loss covers assistant reasoning, tool calls, and the final answer.

The reward is `0.40 R_answer + 0.15 R_format + 0.25 R_tool + 0.20 R_analogy`. Tool and analogy base scores are multiplied by 1, 0.5, or 0.2 for correct, unknown, or incorrect verdicts, respectively. Format is scored separately. See [aris.json](train/configs/aris.json) for the full algorithm configuration.

The tool base score is the number of distinct successful calls divided by the total number of calls. Within each trajectory, deduplication uses exact matching after argument normalization. Natural-language arguments for search and knowledge graph calls also use term and character similarity, with `tool_similarity_threshold` set to 0.90 by default. Values, units, formulas, and other arguments are compared separately. Calls are executed as requested, and deduplication applies to reward scoring.

Training skips groups whose base input exceeds 3,600 tokens. In static prefix mode, the runner checks the complete initial input after the first prefetch and skips groups that reach 10,240 tokens. If all teacher attempts end at a context or tool-round limit, the runner saves the student trajectories, teacher attempts, and reward records for that question, then draws another question. The `skip_reason` is `teacher_context_limit` for context limits, `teacher_tool_round_limit` for tool-round limits, or `teacher_budget_limit` for a mixture of both. Sampling continues until the step has 8 question groups with 8 trajectories each. Data progress reflects the actual samples drawn. Trajectories for the same question and language share prefetched exemplars. Evaluation during training shares the API client and uses `api_concurrency` to cap concurrent judge requests for that evaluation job.

Tool requests share a `tool_timeout` deadline. Transient connection failures, timeouts, and service errors are attempted up to `tool_retries` times, with a default of 3. After the last attempt, the runner returns a tool error observation. If static prefix preparation fails during training, the runner records `exemplar_prefix_failed` and draws a replacement group. During evaluation within training, it records `prefix_error` and assigns the question an answer score of 0. Standalone evaluation saves an error record and stops the run.

The training process selects FlashAttention-2 through `--train-env-vars` and checks the dependency version at startup. SGLang rollouts use a separate inference backend configuration.

The output directory stores checkpoints, data progress, and `aris_config.json`. Interactions from each step are saved to directories such as `rollouts/rollout_0000000/`, numbered by the zero-based `rollout_id`. Each student trajectory and teacher attempt gets its own JSON file, named `student_g<group>_s<sample>_attempt_01.json` or `teacher_g<group>_s<sample>_attempt_<attempt>.json`. An all-incorrect group with G=8 stores all 8 student records and all teacher attempts.

Each record contains the question and reference answer, group and sample numbers, `source` (`student` or `teacher`), `attempt`, `training_branch` (`pending`, `grpo`, `imitation`, or `discard`), `selected_for_training`, the complete `episode`, token IDs, and the loss mask. `selected_for_training=true` means the trajectory was selected for the training batch returned by that step. The `episode` preserves the initial multimodal input, assistant outputs, tool calls and raw results, the observations actually shown, and the image registry. The `reward` field stores `answer`, `format`, `tool_base`, `tool`, `analogy_base`, `analogy`, `s_id`, `s_map`, `s_trans`, `verdict`, and the weighted `total`. The `reward_weights` field stores their weights. Student records also include sampling log probabilities. The same judges score teacher trajectories for the archive, and imitation loss is computed on teacher tokens.

After each assistant response, tool result, and scoring update, the runner atomically replaces the corresponding trajectory file. The `generation_status` and `reward_status` fields track generation and scoring. If a run fails or is canceled, the file retains the interactions and reward components collected so far. The `total` stays `null` until scoring is complete. The `attempt` and `rejection_reason` fields distinguish teacher retries. Repeated runs of a step use directories such as `rollout_0000000_attempt_02/` to preserve each attempt.

Export a Megatron checkpoint to Hugging Face format with the following command.

```bash
PYTHONPATH="$PWD/train/slime${PYTHONPATH:+:$PYTHONPATH}" \
python train/slime/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/checkpoint/iter_0001000 \
  --output-dir /path/to/student_hf \
  --origin-hf-dir "$HF_CHECKPOINT"
```

## SciExplore-Bench Evaluation

Configure the model API name, URL, key, and protocol, then run the evaluation script.

```bash
export ARIS_STUDENT_BASE_URL=https://your-model-provider.example/v1
export ARIS_STUDENT_API_KEY=YOUR_API_KEY
bash evaluation/run_eval.sh --student-model YOUR_MODEL --api-format openai
```

`--api-format` accepts `openai` for OpenAI-compatible APIs and `anthropic` for the Messages API. `--student-base-url` and `--student-api-key` override the environment variables. `--student-model` specifies the model name accepted by the endpoint.

[Model presets](evaluation/models.json) include ARIS, InternVL2.5-8B/26B, Qwen2.5-VL-7B/32B, Qwen3-VL-8B/30B, GPT-5-mini, GPT-5.2, Claude-Haiku-4.5, Claude-Sonnet-4.5, Gemini-2.5-Flash, and Gemini-3-Flash.

```bash
bash evaluation/run_eval.sh --model-preset gpt-5-mini --output evaluation/output/gpt-5-mini/results.jsonl
bash evaluation/run_eval.sh --model-preset claude-haiku-4.5 \
  --student-base-url https://api.anthropic.com \
  --output evaluation/output/claude-haiku-4.5/results.jsonl
```

The default input is `data/bench/aris_eval.jsonl`. You can also pass the original question files directly.

```bash
bash evaluation/run_eval.sh \
  --input data/bench/CN_240.jsonl \
  --input data/bench/EN_240.jsonl \
  --images-root data/bench/images
```

Explicit `--input` arguments take precedence over `EVAL_DATA`. Pass the Chinese file, the English file, or both.

Model presets supply a default model name and protocol, which explicit arguments can override. The GPT-5 and o-series presets use `max_completion_tokens` by default and leave sampling settings to the API. Other OpenAI-compatible models default to `max_tokens`, temperature 0.8, and top-p 1. Configure API parameters with `--token-parameter`, `--sampling-parameters on|off|auto`, and `ARIS_STUDENT_EXTRA_BODY`. The Anthropic API receives text and image content blocks.

Standalone evaluation uses a Qwen3-VL tokenizer and processor for reference token counts. The total context limit defaults to **25,600** and includes the question, images, static exemplars, generated text, and tool results. Set `--max-context-tokens` to change the limit. The initial prompt and cumulative response share this budget. Each generation uses the remaining budget, and the runner checks the reference token count again after it returns. Tokenizer resources are cached in `cache/tokenizers/` on first use. Use `--reference-tokenizer` or `ARIS_EVAL_TOKENIZER` to point to a local Qwen3-VL tokenizer and processor directory. The model API receives the multimodal messages.

Evaluation runs the full question set five times and reports mean accuracy. Use `--temperature` and `--runs` to change these settings. Evaluation shares action parsing, tool execution, and dual-judge scoring with training. Answers are extracted from a complete `<answer>...</answer>` block, including every subquestion, unit, and supplementary text. The runner removes the outer `\boxed{...}` wrapper only if it encloses the entire answer.

By default, the primary judge scores each answer three times. The runner accepts a unanimous correct or incorrect verdict. Otherwise, the secondary judge scores the answer once. A final verdict of unknown receives an answer score of 0. Change the number of primary judge calls with `--judge-repeats`. The run stops if API calls still fail after the retry limit.

The output JSONL stores the run number, prediction, score, and trajectory for each question. The `episode` contains the complete initial input, raw tool results, the multimodal observations shown to the model, and the image registry. The evaluation also writes a matching `.summary.json` file with `per_run`, `mean_accuracy`, `mean_subject_accuracy`, `mean_language_accuracy`, and `mean_language_subject_accuracy`. The language codes `zh` and `en` correspond to CN and EN. With both input files, the summary reports physics, chemistry, and biology accuracy separately for each language. `--max-samples-per-subject N` selects the first N questions for each subject and language. The default of 0 uses the full dataset.

SceMQA evaluation uses the 150 free-response questions listed in the data download section. It shares the model, tools, budgets, and dual-judge scoring with the main benchmark. The default input is `data/ScEMQA/ScEMQA_EN.jsonl`, with images in `data/ScEMQA/images/`.

```bash
bash evaluation/run_scemqa.sh
```

This script wraps `run_eval.sh`, defaults to `--runs 1`, and forwards the same arguments. Set `SCEMQA_DATA` or pass `--input` to specify a file. For JSONL files containing all three subjects, use `physics`, `chemistry`, or `biology` in the `subject` field, or use question IDs with recognizable subject prefixes.

## Comparisons and Ablations

Experiments share the data, model APIs, budgets, and judge settings of the main training and evaluation entry points. Each of the three core ablations provides `train` and `eval` stages. The online teacher and student use the same tool and exemplar settings.

| Experiment | Launch Script | Behavior |
|---|---|---|
| w/o tools | `bash experiments/wo_tools.sh` | Keep the main prompt and disable all tool dispatch |
| prefix | `bash experiments/prefix.sh` | Add 3 exemplars to the prompt and disable all tool dispatch |
| Exemplar count K ablation | `bash experiments/prefix_k.sh` | Evaluate static prefixes with K=1, 2, 3, and 4 on the English subset, with tool dispatch disabled |
| w/o exemplar_recall | `bash experiments/wo_exemplar_recall.sh train\|eval` | Remove the exemplar recall tool and its prompt instructions while keeping the other tools |
| static exemplar prefix (core ablation) | `bash experiments/static_exemplar_prefix.sh train\|eval` | Add 3 exemplars to the prompt, remove the exemplar recall tool, and keep the other tools |
| w/o R_analogy | `bash experiments/wo_analogy_reward.sh train\|eval` | Disable the analogy reward model during training, keep all tools, and evaluate the resulting model |

Before the first generation in prefix mode, the runner uses the target ID to fetch the first K exemplars in their curated order, including their questions, answers, explanations, and images. K defaults to 3. English questions reuse the English translations and cache. The runner passes `--prefix-k` to the service to set the static prefix length. The model's recall call takes no arguments. The complete initial input for static prefix training is subject to the 10,240-token context budget. Standalone evaluation uses a 25,600-token context budget.

With the analogy reward removed, the total reward is `0.40 R_answer + 0.15 R_format + 0.25 R_tool`. ARIS still switches to teacher imitation when all answers in a group are incorrect.

```bash
bash experiments/wo_tools.sh --input data/bench/EN_240.jsonl
bash experiments/prefix.sh --input data/bench/EN_240.jsonl
bash experiments/prefix_k.sh
# Evaluate a single value of K
bash experiments/prefix.sh --prefix-k 2 --input data/bench/EN_240.jsonl

# Train each core ablation, then export its HF checkpoint and start its student service
bash experiments/wo_exemplar_recall.sh train --steps 1000
bash experiments/static_exemplar_prefix.sh train --steps 1000
bash experiments/wo_analogy_reward.sh train --steps 1000

# Configure each ablated model's API and run its evaluation
bash experiments/wo_exemplar_recall.sh eval --input data/bench/EN_240.jsonl
bash experiments/static_exemplar_prefix.sh eval --input data/bench/EN_240.jsonl
bash experiments/wo_analogy_reward.sh eval --input data/bench/EN_240.jsonl
```

Training saves to `train/outputs/<experiment>/` by default, and evaluation saves to `evaluation/output/<experiment>/results.jsonl`. Override the output with `--output`. The scripts forward arguments to the main entry points. Training supports `--dry-run`, `--resume`, and evaluation during training, which automatically uses the active ablation settings. The `experiment` field is saved with the training checkpoint configuration and can also be set through the main entry point with `--experiment`.

The K ablation reads `data/bench/EN_240.jsonl` by default, with 80 questions per subject and 240 total. Each K uses five evaluation runs and the same budget. Outputs go to `evaluation/output/prefix_k/k1/` through `k4/`. Set `PREFIX_K_OUTPUT_DIR` to change the output root, or use `PREFIX_K_DATA` or `--input` to specify the input file. All four K values share one exemplar service.

## Standalone Teacher Trajectory Generation

ARIS calls the teacher online during training. To generate demonstration trajectories separately for selected training questions, set the student's `HF_CHECKPOINT` or pass `--hf-checkpoint`. This uses the same image encoding and token budget checks as training.

```bash
python trajectory_generation/generate_trajectories.py \
  --input data/train/physics_en.jsonl \
  --images-root data/train/images \
  --output trajectory_generation/output/physics_traj.jsonl
```

The output `episode` uses the online teacher's schema and contains the student's initial input and complete interaction for replay. The reference answer is stored separately in `ground_truth`.

## License

The original code is licensed under the [MIT License](LICENSE). Third-party code retains its respective license. See [slime (Apache-2.0)](train/slime/LICENSE) and [Fathom (MIT)](tools/vendor/fathom/LICENSE).
