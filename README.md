# MimicAgent / ARIS

This repository provides ARIS training, SciExplore-Bench evaluation, tool services, and ablation experiments. Training is based on [THUDM/slime](https://github.com/THUDM/slime).

## Data

| Data | Download | Location |
|---|---|---|
| ARIS training data, SciExplore-Bench, and exemplars | [Download](https://huggingface.co/datasets/fanrui00/SciExplore) | `data/` |
| SceMQA | [Official download](https://huggingface.co/datasets/Haozy/SceMQA-main/tree/main) | `data/ScEMQA/` |

Prepare SceMQA questions as `data/ScEMQA/ScEMQA_EN.jsonl` and place images in `data/ScEMQA/images/`.

## Environment and Configuration

Use Linux and Python 3.11+. Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pip install -r tools/requirements.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/cache/browsers" python -m playwright install chromium
```

Copy the [configuration template](train/configs/environment.example.sh) to `environment.local.sh`, fill in your settings, and load it:

```bash
source environment.local.sh
```

## Tool Services

Follow the [knowledge graph setup guide](knowledge_graph.md) to prepare and start Neo4j, then start the tool services:

```bash
bash tools/start_services_demo.sh
```

## ARIS Training

Build the training image:

```bash
docker build -f train/slime/docker/Dockerfile \
  --build-arg SLIME_COMMIT="$(cat train/SLIME_COMMIT)" \
  -t mimicagent-slime:base train/slime
docker build -f train/Dockerfile -t mimicagent-slime:aris train
```

In the training container, install the dependencies and load the environment configuration:

```bash
python -m pip install -e ./train/slime --no-deps
python -m pip install -r train/requirements-aris.txt
bash train/apply_slime_patch.sh
source environment.local.sh
```

Convert the data and start training:

```bash
python train/aris/prepare_data.py \
  --input data/train/physics_cn.jsonl --input data/train/physics_en.jsonl \
  --input data/train/chemistry_cn.jsonl --input data/train/chemistry_en.jsonl \
  --input data/train/biology_cn.jsonl --input data/train/biology_en.jsonl \
  --images-root data/train/images --output data/train/aris_train.jsonl

bash train/run_aris.sh --steps 1000
```

## Evaluation

SciExplore-Bench:

```bash
bash evaluation/run_eval.sh \
  --input data/bench/CN_240.jsonl --input data/bench/EN_240.jsonl \
  --images-root data/bench/images \
  --student-model YOUR_MODEL
```

SceMQA: `bash evaluation/run_scemqa.sh`.

## Comparisons and Ablations

| Experiment | Script |
|---|---|
| No tools / Static exemplar prefix | [wo_tools.sh](experiments/wo_tools.sh) / [prefix.sh](experiments/prefix.sh) |
| Number of exemplars (K) | [prefix_k.sh](experiments/prefix_k.sh) |
| Without exemplar recall | [wo_exemplar_recall.sh](experiments/wo_exemplar_recall.sh) |
| Static exemplar prefix with other tools | [static_exemplar_prefix.sh](experiments/static_exemplar_prefix.sh) |
| Without analogy reward | [wo_analogy_reward.sh](experiments/wo_analogy_reward.sh) |

## License

Original code is released under the [MIT License](LICENSE). Third-party code retains its respective licenses: [slime (Apache-2.0)](train/slime/LICENSE) and [Fathom (MIT)](tools/vendor/fathom/LICENSE).
