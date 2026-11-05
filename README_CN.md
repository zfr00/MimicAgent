# MimicAgent / ARIS

本项目提供 ARIS 训练、SciExplore-Bench 评测、工具服务及消融实验，训练基于 [THUDM/slime](https://github.com/THUDM/slime)。

## 数据

| 数据 | 下载 | 放置目录 |
|---|---|---|
| ARIS 训练集、SciExplore-Bench 与配套题库 | [下载](https://huggingface.co/datasets/fanrui00/SciExplore) | `data/` |
| SceMQA | [官方下载](https://huggingface.co/datasets/Haozy/SceMQA-main/tree/main) | `data/ScEMQA/` |

SceMQA 题目整理为 `data/ScEMQA/ScEMQA_EN.jsonl`，图片放在 `data/ScEMQA/images/`。

## 环境与配置

使用 Linux、Python 3.11+。以下命令均在仓库根目录执行。

```bash
python -m pip install -r requirements.txt
python -m pip install -r tools/requirements.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/cache/browsers" python -m playwright install chromium
```

复制 [配置示例](train/configs/environment.example.sh) 为 `environment.local.sh`，填写后加载：

```bash
source environment.local.sh
```

## 工具服务

先按 [知识图谱部署说明](knowledge_graph.md) 准备并启动 Neo4j，再启动工具服务：

```bash
bash tools/start_services_demo.sh
```

## ARIS 训练

构建训练镜像：

```bash
docker build -f train/slime/docker/Dockerfile \
  --build-arg SLIME_COMMIT="$(cat train/SLIME_COMMIT)" \
  -t mimicagent-slime:base train/slime
docker build -f train/Dockerfile -t mimicagent-slime:aris train
```

在训练容器内安装依赖，并加载上述环境配置：

```bash
python -m pip install -e ./train/slime --no-deps
python -m pip install -r train/requirements-aris.txt
bash train/apply_slime_patch.sh
source environment.local.sh
```

转换数据并启动训练：

```bash
python train/aris/prepare_data.py \
  --input data/train/physics_cn.jsonl --input data/train/physics_en.jsonl \
  --input data/train/chemistry_cn.jsonl --input data/train/chemistry_en.jsonl \
  --input data/train/biology_cn.jsonl --input data/train/biology_en.jsonl \
  --images-root data/train/images --output data/train/aris_train.jsonl

bash train/run_aris.sh --steps 1000
```

## 评测

SciExplore-Bench：

```bash
bash evaluation/run_eval.sh \
  --input data/bench/CN_240.jsonl --input data/bench/EN_240.jsonl \
  --images-root data/bench/images \
  --student-model YOUR_MODEL
```

SceMQA：`bash evaluation/run_scemqa.sh`。

## 对比与消融

| 实验 | 入口 |
|---|---|
| 无工具 / 静态例题前缀 | [wo_tools.sh](experiments/wo_tools.sh) / [prefix.sh](experiments/prefix.sh) |
| 例题数量 K | [prefix_k.sh](experiments/prefix_k.sh) |
| 移除例题召回 | [wo_exemplar_recall.sh](experiments/wo_exemplar_recall.sh) |
| 静态例题前缀并保留其他工具 | [static_exemplar_prefix.sh](experiments/static_exemplar_prefix.sh) |
| 移除类比奖励 | [wo_analogy_reward.sh](experiments/wo_analogy_reward.sh) |

## 许可证

自有代码采用 [MIT License](LICENSE)。第三方代码保留各自许可证：[slime（Apache-2.0）](train/slime/LICENSE)、[Fathom（MIT）](tools/vendor/fathom/LICENSE)。
