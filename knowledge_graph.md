# ATLAS-Wiki 知识图谱配置

`kg_knowledge` 使用 AutoSchemaKG 的 ATLAS-Wiki 图谱，通过 Neo4j 查询实体关系，再由 `gpt-4.1-mini` 过滤证据并按问题语言返回三元组与结论。

仓库提供 `tools/kg_backend.py`、`tools/start_kg.sh` 和统一服务中的 `/kg_knowledge` 接口。后端通过 Neo4j 全文匹配实体，进行多跳遍历并过滤双语证据。

## 下载图谱

- [AutoSchemaKG 项目](https://github.com/HKUST-KnowComp/AutoSchemaKG)
- [公开数据](https://huggingface.co/datasets/gzone0111/AutoSchemaKG/tree/main)
- [Wiki 预构建服务包](https://huggingface.co/datasets/gzone0111/AutoSchemaKG/blob/main/ATLAS%20Neo4j%20Server%20Zip/neo4j-server-wiki.zip)

选择 `ATLAS Neo4j Server Zip/neo4j-server-wiki.zip`，约 74.1 GB。它已经导入图谱数据，上游给出的 Wiki 服务目录解压后约 342 GB。

从仓库根目录执行：

```bash
python -m pip install huggingface_hub
hf download gzone0111/AutoSchemaKG \
  'ATLAS Neo4j Server Zip/neo4j-server-wiki.zip' \
  --repo-type dataset --local-dir data/knowledge_graph
unzip 'data/knowledge_graph/ATLAS Neo4j Server Zip/neo4j-server-wiki.zip' \
  -d data/knowledge_graph
```

后续以解压目录 `data/knowledge_graph/neo4j-server-wiki` 为例，实际外层目录不同则相应修改 `KG_NEO4J_DIR`。

## Neo4j 配置与启动

安装 Java 21 并设置 `JAVA_HOME`。编辑服务包中的 `conf/neo4j.conf`：

```properties
initial.dbms.default_database=wiki-csv-json-text
server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=:7687
server.bolt.advertised_address=127.0.0.1:7687
server.http.enabled=true
server.http.listen_address=:7474
server.http.advertised_address=127.0.0.1:7474
```

按机器内存调整 `server.memory.heap.initial_size`、`server.memory.heap.max_size`、`server.memory.pagecache.size`。

```bash
export KG_NEO4J_DIR="$PWD/data/knowledge_graph/neo4j-server-wiki"
"$KG_NEO4J_DIR/bin/neo4j" start
"$KG_NEO4J_DIR/bin/cypher-shell" -a bolt://127.0.0.1:7687 -u neo4j
```

在 shell 中选择 Wiki 数据库、创建全文索引：

```cypher
:use wiki-csv-json-text
CREATE FULLTEXT INDEX k12_name_fulltext IF NOT EXISTS
FOR (n:Node) ON EACH [n.name];
CALL db.awaitIndexes(3600);
SHOW INDEXES YIELD name, state WHERE name='k12_name_fulltext'
RETURN name, state;
```

确认索引为 `ONLINE`。用 `SHOW DATABASES;` 核对实际数据库名，名称不同则同步调整 `NEO4J_DATABASE`。

## 后端配置

```bash
python -m pip install -r tools/requirements.txt
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='填入数据库当前密码'
export NEO4J_DATABASE=wiki-csv-json-text
export KG_FULLTEXT_INDEX=k12_name_fulltext
export KG_LLM_MODEL=gpt-4.1-mini
export KG_LLM_BASE_URL=https://your-api-provider.example/v1
export KG_LLM_API_KEY='填入 API key'
export KG_BASE_URL=http://127.0.0.1:10087
```

`KG_LLM_BASE_URL`、`KG_LLM_API_KEY` 未单独设置时依次使用 `ARIS_PRIMARY_*`、`OPENAI_*`。

默认查找 10 个实体候选、尝试其中 2 个，最多 2 跳、每跳扩展 5 个节点；将最多 48 条原始证据交给模型过滤，返回最多 10 条三元组。后端请求可设置 `hops`、`top_k`、`limit_per_hop`、`fuzzy_limit`。

## 启动与验证

只启动 KG 后端：

```bash
bash tools/start_kg.sh
```

另一个终端检查：

```bash
curl http://127.0.0.1:10087/health
curl http://127.0.0.1:10087/tools/kg_knowledge \
  -H 'Content-Type: application/json' \
  -d '{"query":"法拉第电磁感应定律与楞次定律的关系","hops":2,"top_k":10}'
```

响应包含 `status`、`chosen_entity`、`triples`、`warnings`、`conclusion`。中文问题生成英文查询与实体词，检索英文图谱后返回中文证据；英文问题返回英文证据。结论基于筛选出的图谱证据。

与其余工具一起运行时，先按根 README 配置例题库、搜索 API 和 `ARIS_TOOL_API_KEY`，再执行 `bash tools/start_services_demo.sh`。该入口启动 KG、Fathom 后端及 8000 端口的统一服务。KG 已在其他机器启动时，设置 `START_KG_BACKEND=0` 并修改 `KG_BASE_URL`。

训练、评测和教师使用统一服务的 `/kg_knowledge`，其 `ARIS_TOOL_API_KEY` 与服务保持一致：

```bash
curl http://127.0.0.1:8000/kg_knowledge \
  -H "Authorization: Bearer $ARIS_TOOL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What determines the period of a simple pendulum?"}'
```

`bash tools/start_services_demo.sh --stop` 停止脚本启动的工具服务。Neo4j 单独用 `"$KG_NEO4J_DIR/bin/neo4j" stop` 停止。
