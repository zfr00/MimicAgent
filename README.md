# MimicAgent / ARIS

本仓库提供 MimicAgent 的 ARIS 训练、SciExplore-Bench 评测、工具与例题消融脚本和五类工具服务。训练基于 [THUDM/slime](https://github.com/THUDM/slime)，使用 Qwen3-VL-8B-Instruct、SGLang rollout 和 Megatron Bridge。

## 目录

```text
data/
  train/                   训练题目及图片
  bench/                   SciExplore-Bench 评测题目及图片
  ScEMQA/                  SceMQA 三学科 free-response 评测数据
  exemplars/               训练和评测题目对应的配套题库
cache/exemplar_translations/ 英文例题译文缓存
train/                     ARIS 训练实现、slime、配置和启动脚本
evaluation/                模型 API、双裁判与五次平均评测
experiments/               工具对比和三项核心消融的训练／评测入口
tools/                     五类工具的 HTTP 服务
prompts/                   学生、教师和答案裁判提示词
trajectory_generation/     教师轨迹生成入口
```


## 数据下载与放置

ARIS 训练集、SciExplore-Bench 评测集和配套题库合为一个数据包，解压到 `data/` 后得到 `train/`、`bench/`、`exemplars/`。`train` 存放训练集，`bench` 存放评测集，`exemplars` 存放按目标题 ID 配对的例题及参考解答。

| 数据 | 下载链接 | 放置目录 |
|---|---|---|
| ARIS 数据包（训练集、SciExplore-Bench 评测集、配套题库） |  | `data/` |
| SceMQA 评测子集（150 题） | [官方下载](https://huggingface.co/datasets/Haozy/SceMQA-main/tree/main) | `data/ScEMQA/` |

SceMQA 使用官方 `Free_Response/free_response.json` 中的 `Physics_Free_Response`、`Chemistry_Free_Response`、`Biology_Free_Response` 三个子集的全部题目，各 50 题，共 150 题。题目及对应图片放在 `data/ScEMQA/`；数据来源见 [SceMQA 官方项目](https://github.com/SceMQA/SceMQA)。

训练集和 bench 保持原始文件组织和字段格式。解压后的布局如下：

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

### 题目格式与转换

训练集和评测集均为 JSONL，每行一道题：

| 字段 | 含义 |
|---|---|
| `id` | 原始题目 ID，与配套题库的 `target_id` 对应；中英文版本可共用 |
| `question` | 题目文本，可包含 `<image>` 占位符 |
| `answer` | 参考答案 |
| `image.question` | 图片文件名列表，相对于该数据集的 `images/` |
| `image.answer`、`image.analysis` | 原始答案图片与解析图片 |
| `analysis` | 原始解析文本，可为空 |
| `knowledge` | 知识点字符串 |
| `year`、`difficulty_level`、`question_type` | 原始题目元数据 |
| `grade_level` | 训练题目的年级 |

学生输入由目标题的 `question` 和 `image.question` 构成，教师和答案裁判使用文字参考答案。代码从文件名识别中英文，从文件名或 ID 前缀识别学科；单题直接调用时根据题干文字识别语言。同 ID 的中英文记录在运行时按 `(id, language)` 区分，召回仍使用原始 ID。


将训练题目转换为 ARIS 输入：

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

评测题目可同样合并转换：

```bash
python train/aris/prepare_data.py \
  --input data/bench/CN_240.jsonl \
  --input data/bench/EN_240.jsonl \
  --images-root data/bench/images \
  --output data/bench/aris_eval.jsonl
```

转换后包含 `messages`、`label`、`metadata`。参考答案保存在 `label`；`metadata` 记录原始题目 ID、原题、语言、学科和题干图片绝对路径。在使用数据的机器上执行转换，训练 worker 和工具服务使用相同的图片路径。

### 配套题库格式

配套题库只有一个中文 `curatorial_repo.jsonl` 和一个 `images/`。每行对应一个原始 `target_id`，包含四道有序 `companions`；中英文目标题共用这一条记录。四道例题的 `id` 按数组顺序依次为 `<target_id>_01`、`<target_id>_02`、`<target_id>_03`、`<target_id>_04`。例如 `target_id` 为 `hs_phy_0003` 时，第一道 companion 的结构如下：

```json
{
  "id": "hs_phy_0003_01",
  "question": "中文例题题干 <image>",
  "answer": "中文参考答案，可含 <image>",
  "analysis": "中文解析，可含 <image>",
  "knowledge": "知识点",
  "image": {
    "question": ["hs_phy_0003_01_01.png"],
    "answer": ["hs_phy_0003_01_02.png"],
    "analysis": ["hs_phy_0003_01_03.png"]
  }
}
```

图片文件名相对于例题库旁的 `images/`，每个字段的图片按对应 `<image>` 的顺序排列；无图时为 `[]`。

同一道例题只有一张图片时，文件名为 `<例题id>.<扩展名>`；有多张图片时，按题干、答案、解析的顺序统一连续编号，文件名为 `<例题id>_01.<扩展名>`、`<例题id>_02.<扩展名>` 等，各部分内部保持图片列表顺序。例如单图例题 `hs_phy_0003_02` 使用 `hs_phy_0003_02.png`。图片归属由 `image.question`、`image.answer`、`image.analysis` 中的引用确定。

召回服务按当前题目 ID 查表，默认返回前三道。`EXEMPLAR_RECALL_K` 可在启动服务前设置为 1–4；模型调用为 `{"name":"exemplar_recall","arguments":{}}`。runner 注入 `target_id` 和 `target_language`。静态前缀的数量由运行参数 `--prefix-k` 控制，默认 3，可选 1–4。例题题干、答案和解析的图片以像素内容返回模型，也传给类比奖励模型。

### 模型的图文输入

目标题文字和题干图片放在同一条 user 消息中，system 消息提供角色、工具与作答规则。构造消息时，程序按字段中的 `<image>` 位置插入真实图片块，保留前后文字顺序；有图但完全没有占位符的字段将图片附在本段末尾。有占位符时，其数量必须与该字段的图片数量一致。

例题按“例题 ID → 题干 → 答案 → 解析 → 知识点”展示，各部分图片在对应位置插入。英文例题先完成文字翻译，再构造同样的图文消息。训练、评测、教师、静态例题前缀和类比奖励共用这套展示逻辑。

每条轨迹单独维护图片表，从 `image_index=0` 开始编号。题干图、召回例题图和裁图结果均可通过图片旁的编号调用 `image_crop`；已登记图片重复展示时沿用编号。原图尺寸随编号标明。裁图使用图片表中的原始像素，默认归一化坐标；`bbox_type=px` 按原图尺寸解释。返回内容为来源编号、实际裁剪区域和真实裁图。

视觉输入按配置缩放并由学生 processor 编码，图片 token 计入上下文预算。工具返回整体通过长度检查后，图文内容和新增图片编号才进入后续上下文；超限时保留调用记录和原始返回，标记 `accepted=false` 并停止该轨迹。

### 英文例题翻译与缓存

中文目标题直接召回中文例题。英文目标题将所选例题的 `question`、`answer`、`analysis`、`knowledge` 原文整批发送给 GPT-4.1-mini。翻译包含公式内的中文说明，并保持公式结构、数值、单位和图片占位；图片及图中文字原样保留，ID 和图片引用由服务保留。

译文写入仓库内独立的 `cache/exemplar_translations/`。相同例题内容与翻译配置命中缓存时直接复用，并发请求共用同一份译文；内容或翻译配置变化后生成新缓存。中文题库文件保持原样，删除缓存文件可触发重新翻译。

翻译或译文校验重试耗尽后，召回返回中文原文及图片，训练和评测继续，返回语言标记为 `zh`。英文缓存保存校验通过的译文；缓存写入失败时直接使用当次译文。失败原因记录到工具服务日志。

| 配置 | 默认值 |
|---|---|
| `ARIS_TRANSLATOR_MODEL` | `gpt-4.1-mini` |
| `ARIS_TRANSLATOR_BASE_URL` / `API_KEY` | 可独立配置，否则使用 `ARIS_PRIMARY_*`，再回退到 `OPENAI_*` |
| `EXEMPLAR_TRANSLATION_CACHE_DIR` | 仓库内 `cache/exemplar_translations/` |
| `EXEMPLAR_TRANSLATION_RETRIES` | 3 次 |
| `EXEMPLAR_TRANSLATION_TIMEOUT` | 单次 API 请求 60 秒 |
| `EXEMPLAR_TRANSLATION_MAX_TOKENS` | 8192 |

翻译 API 或译文格式检查重试后仍失败时，错误通过工具服务传回，训练和评测立即停止。

## 环境与配置

工具服务和独立评测依赖：

```bash
python -m pip install -r requirements.txt
```

使用 Python 3.11 或更新版本。Python 工具运行在 Linux bubblewrap 沙箱中，安装系统依赖 `bubblewrap`，并启用用户、文件系统、网络和进程命名空间。可通过 `CODE_BWRAP` 指定可执行文件路径。

GPU 训练环境由 slime 镜像和 [ARIS Dockerfile](train/Dockerfile) 构建，训练注意力固定为 FlashAttention-2 2.8.3。slime 固定提交为 `2a60f00dd27f716f8761464086f02b02061adff1`，记录在 [SLIME_COMMIT](train/SLIME_COMMIT)。三行 [metadata 补丁](train/patches/slime-aris.patch) 将 ARIS 分支及权重传入训练损失。

```bash
docker build -f train/slime/docker/Dockerfile \
  --build-arg SLIME_COMMIT="$(cat train/SLIME_COMMIT)" \
  -t mimicagent-slime:base train/slime
docker build -f train/Dockerfile -t mimicagent-slime:aris train
```

在 GPU 环境中安装仓库的 slime 和 ARIS 依赖：

```bash
python -m pip install -e ./train/slime --no-deps
python -m pip install -r train/requirements-aris.txt
bash train/apply_slime_patch.sh
```

复制 [环境配置示例](train/configs/environment.example.sh) 到 `environment.local.sh`，填写路径、服务地址与密钥后加载：

```bash
source environment.local.sh
```

| 配置 | 用途 / 默认路径 |
|---|---|
| `HF_CHECKPOINT` | 训练学生的 HF 模型、tokenizer 和 processor 目录 |
| `TRAIN_DATA` | `data/train/aris_train.jsonl` |
| `EVAL_DATA` | `data/bench/aris_eval.jsonl` |
| `CURATORIAL_REPO_PATH` | `data/exemplars/curatorial_repo.jsonl` |
| `OUTPUT_DIR` | 训练 checkpoint 目录，默认 `train/outputs/aris` |
| `EVAL_OUTPUT` | 评测输出，默认 `evaluation/output/results.jsonl` |
| `ARIS_STUDENT_*` | 独立评测的模型 API 地址、密钥及协议 |
| `ARIS_EVAL_TOKENIZER` | 独立评测的 Qwen3-VL 参考 tokenizer/processor |
| `ARIS_TEACHER_*` | 在线教师，默认 gpt-5.2 |
| `ARIS_PRIMARY_*` | 主裁判，默认 gpt-4.1-mini |
| `ARIS_SECONDARY_*` | 副裁判，默认 gemini-3-flash |
| `ARIS_ANALOGY_*` | 类比奖励模型，默认 gemini-3-flash |
| `ARIS_TRANSLATOR_*` | 例题翻译模型，默认 gpt-4.1-mini |

这些模型角色分别支持 `MODEL`、`BASE_URL`、`API_KEY`；地址和密钥可共用 `OPENAI_BASE_URL` / `OPENAI_API_KEY`。`TOKEN_PARAMETER` 支持 `max_tokens` 或 `max_completion_tokens`，`EXTRA_BODY` 接收服务特定 JSON 参数；教师默认使用 `max_completion_tokens`。翻译配置在启动工具服务前设置。

## 工具服务

五类工具由同一个 FastAPI executor 提供，默认地址为 `http://127.0.0.1:8000`。所有调用使用 `ARIS_TOOL_API_KEY` 的 Bearer 鉴权，默认最多并发 128 个请求（`TOOL_MAX_CONCURRENCY`）。先安装工具依赖和 Fathom 的浏览器抓取后备通道：

```bash
python -m pip install -r tools/requirements.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/cache/browsers" python -m playwright install chromium
```

| 工具接口 | HTTP 路由 | 默认端口 | 配置 |
|---|---|---|---|
| `exemplar_recall` | `POST /exemplar_recall` | 8000 | 配套题库路径，见上文 |
| `web_search.search` | `POST /search` | 8000 | `SERPER_API_KEY`、Fathom 后端 |
| `web_search.query` | `POST /query` | 8000 | `WEB_LLM_*`、Fathom 后端 |
| `kg_knowledge` | `POST /kg_knowledge` | 8000 | Neo4j 与 `KG_LLM_*` |
| `image_crop` | `POST /image_crop` | 8000 | runner 按会话编号提供选定图片的原始像素 |
| `code_interpreter` | `POST /code_interpreter` | 8000 | Python 科学计算依赖 |

`web_search` 使用 [Fathom-DeepResearch](https://github.com/FractalAIResearchLabs/Fathom-DeepResearch) `sandbox_serper.py`，代码位于 `tools/vendor/fathom/`，固定提交见 `tools/FATHOM_COMMIT`。搜索调用其 `search_urls`（Serper），网页读取调用其 `query_url`（网页抓取与按目标进行 LLM 提取）；返回 JSON 按 `cl100k_base` 限制到默认 3,000 tokens（`WEB_MAX_TOKENS`），单次后端请求超时 350 秒，外层工具预算 360 秒。网页提取默认 `WEB_LLM_MODEL=gpt-4.1-mini`，地址和 key 可用 `WEB_LLM_BASE_URL`、`WEB_LLM_API_KEY` 设置，否则继承 `ARIS_PRIMARY_*`。Fathom 的部署补丁处理 LLM 超时与模型名、PDF／浏览器抓取模块按需加载，以及 spaCy 缺少英文模型时的句子分割；补丁见 `tools/patches/fathom.patch`，上游 MIT 许可保留在 vendor 中。

Serper 使用 `https://google.serper.dev/search` 和 `SERPER_API_KEY`，按搜索词与 `top_k` 缓存在 `cache/web_search/serper/`，`SEARCH_CACHE_TTL` 可设置有效期秒数，默认 0 表示永久有效。Jina Reader 使用 `https://r.jina.ai/` 和 `JINA_API_KEY`，按规范化 URL 将网页原文缓存在 `cache/web_search/jina/jina_read/`。网页抓取根据域名选择 Jina、直接读取或浏览器等通道，缓存命中时复用已有内容。

KG 后端已包含在 `tools/kg_backend.py`。ATLAS-Wiki 下载、Neo4j 启动与索引、双语过滤 API 和启动命令见 [knowledge_graph.md](knowledge_graph.md)。

```bash
bash tools/start_services_demo.sh
bash tools/start_services_demo.sh --stop
```

日志和进程记录位于 `tools/logs/`。服务按进程组启动和停止，启动失败时清理本次启动的服务。远程服务可通过 `ARIS_TOOL_ENDPOINTS` JSON 映射配置。统一 executor 的 `TOOL_EXECUTION_TIMEOUT` 默认 360 秒，每个请求在独立进程中执行；截止时间传递给 Fathom 和 KG 后端。超时、调用方断开或请求取消时终止对应进程组。普通工具错误返回给模型继续处理，例题翻译失败则停止运行。

脚本默认启动本地 Fathom（8904）、KG 后端（10087）和统一 executor（8000）；Neo4j 先按 KG 文档启动。使用远程后端时设置 `START_FATHOM=0` / `START_KG_BACKEND=0`，并配置 `FATHOM_BASE_URL` / `KG_BASE_URL`。后端也可分别用 `tools/start_fathom.sh`、`tools/start_kg.sh` 启动。

`image_crop` 按原图坐标裁剪并进行 Lanczos 重采样，最多放大 2 倍，像素上限默认 262,144（`CROP_MAX_PIXELS`）；裁图作为真实图像回传给学生和教师。

Python 工具每次请求启动独立沙箱，预加载 numpy、scipy、sympy、matplotlib、pandas。`CODE_EXECUTION_TIMEOUT` 默认 360 秒，`CODE_MAX_OUTPUT_BYTES` 默认 32768，`CODE_MEMORY_BYTES` 默认 4 GiB；超时或超量会终止进程并返回错误。

## ARIS 训练

完成数据转换、配置和工具服务启动后：

```bash
bash train/run_aris.sh --dry-run
bash train/run_aris.sh --steps 1000
```

训练默认读取 `data/train/aris_train.jsonl`，可用 `TRAIN_DATA` 或 `--train-data` 覆盖。默认单节点 8 GPU、TP=2、PP=2，与 rollout 共用 GPU。已有 Ray 集群可传 `--ray-address auto`；并行度由 `--num-gpus`、`--tp`、`--pp`、`--rollout-gpus-per-engine` 调整。

```bash
# 恢复训练
bash train/run_aris.sh --resume "$OUTPUT_DIR" --steps 1000

# 训练期间评测
bash train/run_aris.sh --eval-data data/bench/aris_eval.jsonl --eval-interval 20
```

`--resume` 自动读取 checkpoint 目录中的 `aris_config.json`，恢复原实验模式、奖励权重、交互预算等 ARIS 配置；指定 `iter_XXXXXXX` 子目录时从父目录读取配置和对应的数据进度，加载指定迭代的权重及优化器状态，并从下一 rollout 继续。显式 `--config` 可替换配置文件，`--experiment`、`--group-size` 等显式参数继续覆盖对应值。续训配置优先于 `ARIS_CONFIG` 环境变量。

`--eval-data` 启用训练期间评测：训练开始前及每隔 `--eval-interval` 步，用当前学生模型回答指定评测集并计算答案分数。该阶段沿用训练交互预算，初始输入超限的题目记录为 `context_limit`、答案分数为 0，并计入评测题数；评测完成后继续训练。

每步默认 B=8 道题、每题 G=8 条学生轨迹。组内至少一个答案正确时使用 GRPO；全错组在线生成一条教师轨迹，使用 λ=10 的 imitation loss。同题两个分支互斥，按题平均。

学生训练保留 SGLang 返回的 token IDs 和 log probabilities。教师轨迹在学生提示下重新编码。GRPO 使用采样温度 0.8，教师 NLL 使用温度 1。

| 参数 | 默认值 |
|---|---|
| prompt / response / context token 上限 | 3600 / 7680 / 10240 |
| 学生采样温度 / 最大工具轮数 | 0.8 / 6 |
| GRPO clipping 下界 / 上界 | 0.20 / 0.28 |
| KL / entropy 系数 | 0 / 0 |
| Adam 学习率 / betas | 常数 1e-6 / (0.9, 0.95) |
| weight decay / grad clip | 0.1 / 1.0 |
| activation recompute | full + uniform |
| SGLang memory fraction / 并发 | 0.7 / 160 |

六轮指六次工具交互，每轮最多调用一个工具；第六次工具返回后仍可再生成最终答案。3,600 限制系统提示、题干和题干图片组成的基础输入；静态例题前缀与基础输入合计计入 10,240 的总上下文预算。7,680 是完整初始输入之后整段轨迹的上限，包括助手输出、工具返回、图片 token 和消息分隔 token；轨迹长度与总上下文限制同时生效。

教师模仿使用完整多轮交互。`aris_episode` 保存学生初始提示及图片、每轮助手输出、工具名称和参数、工具原始返回、实际展示的图文观察，以及带原始像素和来源的图片表。回放复用已保存的观察和编号；学生 tokenizer/processor 重新编码整条轨迹。每轮助手推理、工具调用和最终答案参与模仿损失。

奖励为 `0.40 R_answer + 0.15 R_format + 0.25 R_tool + 0.20 R_analogy`。工具与类比基础分按 correct / unknown / incorrect 分别乘 1 / 0.5 / 0.2；格式单独评分。完整算法配置见 [aris.json](train/configs/aris.json)。

工具基础分为去重后的成功调用数除以总调用次数。同一轨迹内，参数规范化后进行精确匹配；搜索与知识图谱的自然语言参数同时进行词项和字符相似度匹配，`tool_similarity_threshold` 默认 0.90。数值、单位、公式和其他参数单独比较。工具调用按原请求执行，去重用于奖励计算。

训练时跳过基础输入超过 3,600 tokens 的题组；静态前缀在首次预取后检查完整初始输入，达到 10,240 tokens 时跳过题组。教师全部尝试因长度或工具轮数限制而结束时，保存该题的学生轨迹、教师尝试及奖励记录，并补取下一题。`skip_reason` 分别为 `teacher_context_limit`、`teacher_tool_round_limit`，混合超限为 `teacher_budget_limit`。采样继续补齐每步 8 个题组、每组 8 条轨迹，数据进度随实际取样保存。同题同语言的多条轨迹复用预取结果。训练期间评测共用 API 客户端，`api_concurrency` 控制该评测任务的裁判请求总并发。

工具请求共享 `tool_timeout` 期限，临时连接、超时和服务错误最多尝试 `tool_retries` 次，默认 3 次。尝试耗尽后返回工具错误观察。静态例题前缀准备失败时，训练记录 `exemplar_prefix_failed` 并补取题组；训练期间评测将该题记为 `prefix_error`、答案分数为 0；独立评测保存错误记录并结束运行。

训练进程通过 `--train-env-vars` 选择 FlashAttention-2，启动时检查依赖版本；SGLang rollout 使用独立的推理后端配置。

输出目录保存 checkpoint、数据源进度和 `aris_config.json`。每步交互自动保存到 `rollouts/rollout_0000000/` 等目录，编号为从 0 开始的 `rollout_id`。每条学生轨迹、每次教师尝试各保存一个 JSON 文件，文件名为 `student_g<组号>_s<样本号>_attempt_01.json` 或 `teacher_g<组号>_s<样本号>_attempt_<次数>.json`。G=8 的全错组保存 8 条学生记录及全部教师尝试。

记录包含题目与参考答案、组号与样本号、`source`（`student` / `teacher`）、`attempt`、`training_branch`（`pending` / `grpo` / `imitation` / `discard`）、`selected_for_training`、完整 `episode`、token IDs 和损失掩码。`selected_for_training=true` 表示该轨迹已被选入本步返回的训练批次。`episode` 保留初始图文输入、助手输出、工具调用及原始返回、实际图文观察和图片表。`reward` 保存 `answer`、`format`、`tool_base`、`tool`、`analogy_base`、`analogy`、`s_id`、`s_map`、`s_trans`、`verdict` 和加权 `total`，`reward_weights` 保存对应权重。学生记录同时保存采样 log probabilities。教师使用相同裁判评分并存档，模仿损失按教师 token 计算。

每轮助手回复、工具返回及评分进度通过原子替换写入对应轨迹文件。`generation_status` 和 `reward_status` 记录生成与评分状态；失败或取消时保留已有交互及已得到的奖励分项，评分未完成时 `total` 为 `null`。教师重试通过 `attempt` 和 `rejection_reason` 区分。重复执行同一步时使用 `rollout_0000000_attempt_02/` 等目录保留各次记录。

Megatron checkpoint 可导出为 HF 格式：

```bash
PYTHONPATH="$PWD/train/slime${PYTHONPATH:+:$PYTHONPATH}" \
python train/slime/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/checkpoint/iter_0001000 \
  --output-dir /path/to/student_hf \
  --origin-hf-dir "$HF_CHECKPOINT"
```

## SciExplore-Bench 评测

配置模型 API 的名称、地址、密钥与协议后运行：

```bash
export ARIS_STUDENT_BASE_URL=https://your-model-provider.example/v1
export ARIS_STUDENT_API_KEY=YOUR_API_KEY
bash evaluation/run_eval.sh --student-model YOUR_MODEL --api-format openai
```

`--api-format` 支持 `openai`（OpenAI 兼容接口）和 `anthropic`（Messages 接口）。`--student-base-url`、`--student-api-key` 可覆盖环境变量，`--student-model` 指定接口接受的模型名。

[模型预设](evaluation/models.json) 包含 ARIS、InternVL2.5-8B/26B、Qwen2.5-VL-7B/32B、Qwen3-VL-8B/30B、GPT-5-mini、GPT-5.2、Claude-Haiku-4.5、Claude-Sonnet-4.5、Gemini-2.5-Flash 和 Gemini-3-Flash。例如：

```bash
bash evaluation/run_eval.sh --model-preset gpt-5-mini --output evaluation/output/gpt-5-mini/results.jsonl
bash evaluation/run_eval.sh --model-preset claude-haiku-4.5 \
  --student-base-url https://api.anthropic.com \
  --output evaluation/output/claude-haiku-4.5/results.jsonl
```

默认读取 `data/bench/aris_eval.jsonl`。也可直接指定原始题目文件：

```bash
bash evaluation/run_eval.sh \
  --input data/bench/CN_240.jsonl \
  --input data/bench/EN_240.jsonl \
  --images-root data/bench/images
```

显式 `--input` 优先于 `EVAL_DATA`，可分别输入中文或英文文件，也可同时输入。

模型预设提供默认模型名和协议，显式参数可覆盖。GPT-5、o 系列默认使用 `max_completion_tokens` 并由接口控制采样参数；其他 OpenAI 兼容模型默认使用 `max_tokens`、温度 0.8 和 top-p 1。`--token-parameter`、`--sampling-parameters on|off|auto` 和 `ARIS_STUDENT_EXTRA_BODY` 可配置接口参数。Anthropic 接口传入文字及图片内容块。

独立评测统一用 Qwen3-VL tokenizer/processor 计算参考 token 数，总上下文上限为 **25,600**，题干、图片、静态例题、生成文字和工具返回均计入。`--max-context-tokens` 可调整总上限。初始 prompt 与累计响应受同一总预算约束，每轮生成使用剩余预算，返回后按参考计数复核。计数资源首次加载到 `cache/tokenizers/`；`--reference-tokenizer` 或 `ARIS_EVAL_TOKENIZER` 可指定本地 Qwen3-VL tokenizer/processor 目录。模型 API 接收正常图文消息。

对全部题目运行五次并取准确率均值，可用 `--temperature`、`--runs` 配置。评测与训练共用动作解析、工具执行及双裁判。答案从完整 `<answer>...</answer>` 提取，保留所有小问、单位及补充文字；仅当一个 `\boxed{...}` 包住整个答案时移除外层包装。

主裁判默认重复三次，全部一致且为 correct / incorrect 时采纳，否则副裁判一次。最终 unknown 的答案分为 0。`--judge-repeats` 可调整主裁判次数；API 有限重试后仍失败时停止执行。

输出 JSONL 保存每题的运行编号、预测、得分和轨迹；`episode` 包含完整的初始输入、工具原始结果、实际图文观察和图片表。完成后生成同名 `.summary.json`，包含 `per_run`、`mean_accuracy` 和 `mean_subject_accuracy`，以及按语言区分的 `mean_language_accuracy`、`mean_language_subject_accuracy`。`zh` 和 `en` 分别对应 CN、EN；同时输入两个文件时，仍分别输出两种语言下物理、化学、生物的准确率。`--max-samples-per-subject N` 分别为每种语言的每个学科选取前 N 题，默认 0 表示全量。

SceMQA 使用上文下载部分列出的 150 道 free-response 题目。它与主 bench 共用模型、工具、预算和双裁判。评测入口默认读取 `data/ScEMQA/ScEMQA_EN.jsonl`，图片目录为 `data/ScEMQA/images/`，运行：

```bash
bash evaluation/run_scemqa.sh
```

该入口复用 `run_eval.sh`，默认 `--runs 1`，可透传同样的参数；`SCEMQA_DATA` 或 `--input` 可指定文件路径。混合三学科的 JSONL 可用 `subject` 字段标明 `physics`、`chemistry`、`biology`，或沿用可识别学科的题目 ID。

## 对比与消融实验

各实验复用主训练和评测入口的数据、模型 API、预算与裁判配置。三项核心消融分别提供 `train` 和 `eval` 阶段，在线教师与学生使用相同的工具和例题设置。

| 实验 | 启动脚本 | 行为 |
|---|---|---|
| w/o tools | `bash experiments/wo_tools.sh` | 保持主提示词，关闭全部工具调度 |
| prefix | `bash experiments/prefix.sh` | 预先放入 3 条例题，关闭全部工具调度 |
| 例题数量 K 消融 | `bash experiments/prefix_k.sh` | EN 子集上依次评测 K=1、2、3、4 的静态前缀，关闭全部工具调度 |
| w/o exemplar_recall | `bash experiments/wo_exemplar_recall.sh train\|eval` | 移除例题召回工具及其提示，保留其他工具 |
| static exemplar prefix（核心消融） | `bash experiments/static_exemplar_prefix.sh train\|eval` | 预先放入 3 条例题，移除例题召回工具，保留其他工具 |
| w/o R_analogy | `bash experiments/wo_analogy_reward.sh train\|eval` | 训练时关闭类比奖励模型，保留所有工具；评测对应训练出的模型 |

前缀模式在首次生成前按目标题 ID 读取例题服务，使用固定策展顺序的前 K 条例题及其题干、答案、解析和图片，默认 K=3。英文题复用英文翻译与缓存。`--prefix-k` 由 runner 传给服务，仅控制静态前缀数量；模型的召回调用保持无参数。静态前缀训练的完整初始输入受 10,240 总上下文预算约束，独立评测使用 25,600 总上下文预算。

移除类比奖励时，总奖励为 `0.40 R_answer + 0.15 R_format + 0.25 R_tool`；ARIS 的答案全错切换规则保持一致。

```bash
bash experiments/wo_tools.sh --input data/bench/EN_240.jsonl
bash experiments/prefix.sh --input data/bench/EN_240.jsonl

bash experiments/prefix_k.sh
# 单独运行其中一个 K
bash experiments/prefix.sh --prefix-k 2 --input data/bench/EN_240.jsonl

# 核心消融：先训练，再按上文导出 HF checkpoint 并启动学生服务
bash experiments/wo_exemplar_recall.sh train --steps 1000
bash experiments/static_exemplar_prefix.sh train --steps 1000
bash experiments/wo_analogy_reward.sh train --steps 1000

# 配置各自消融模型的 API 后运行对应评测
bash experiments/wo_exemplar_recall.sh eval --input data/bench/EN_240.jsonl
bash experiments/static_exemplar_prefix.sh eval --input data/bench/EN_240.jsonl
bash experiments/wo_analogy_reward.sh eval --input data/bench/EN_240.jsonl
```

训练默认保存到 `train/outputs/<实验名>/`，评测默认保存到 `evaluation/output/<实验名>/results.jsonl`。可用 `--output` 覆盖，各脚本透传主入口参数；训练支持 `--dry-run`、`--resume` 和训练期间评测。训练期间评测自动沿用该消融设置。配置中的 `experiment` 字段会随训练 checkpoint 配置保存，也可通过主入口的 `--experiment` 指定。

K 消融默认读取 `data/bench/EN_240.jsonl`（三学科各 80 题，共 240 题），每个 K 沿用五次评测及相同预算。输出分别保存到 `evaluation/output/prefix_k/k1/` 至 `k4/`，可用 `PREFIX_K_OUTPUT_DIR` 修改输出根目录；`PREFIX_K_DATA` 或 `--input` 可指定输入文件。四种 K 共用同一个例题服务。

## 单独生成教师轨迹

ARIS 在训练中在线调用教师。单独生成指定训练题目的示范轨迹时，设置学生的 `HF_CHECKPOINT`（或传入 `--hf-checkpoint`），用于与训练一致的图像编码和 token 预算检查：

```bash
python trajectory_generation/generate_trajectories.py \
  --input data/train/physics_en.jsonl \
  --images-root data/train/images \
  --output trajectory_generation/output/physics_traj.jsonl
```

输出的 `episode` 与在线教师使用同一结构，包含可直接回放的学生初始输入及完整交互。`ground_truth` 作为单独的参考字段保存。

## 许可证

自有代码采用 [MIT License](LICENSE)。第三方代码保留各自许可证：[slime（Apache-2.0）](train/slime/LICENSE)、[Fathom（MIT）](tools/vendor/fathom/LICENSE)。
