# 垂直客服 Agent（Phase 1）

基于 **LangGraph + Redis + PostgreSQL (pgvector)** 构建的垂直领域（默认电商售后场景）客服 Agent 第一阶段核心底座，完整实现了 **Agent Loop**、**分层记忆系统（L1/L2/L3）** 与 **工具调用体系**。

## 核心架构

1. **Agent Loop (`src/agent/`)**
   - `load_memory -> agent_llm <-> tool_executor -> save_memory` 状态机闭环。
   - 内置单轮最大工具调用次数熔断（`MAX_TOOL_ITERATIONS`），超限、异常或触发转人工时优雅路由至 `fallback_or_escalate`，并自动闭合挂起的 `tool_call_id` 防止后续对话协议报错。
2. **分层记忆 (`src/memory/`)**
   - **L1 工作记忆（Redis）**：`cs:session:{id}:messages`（`List` + `LTRIM` 滑动窗口）与 `cs:session:{id}:slots`（独立 `Hash` 活跃业务槽位）物理分离存储，活跃槽位不受最近 $N$ 轮滑窗裁剪影响。
   - **L2 会话状态（Postgres Checkpointer）**：LangGraph `AsyncPostgresSaver` 全量持久化；Redis 过期未命中时自动从 Postgres 恢复全量状态并回填 Redis。
   - **L3 长期记忆（Postgres + pgvector）**：条件触发 + 异步沉淀（会话结束/转人工/每 $N$ 轮/建单成功），过滤无意义寒暄，情景摘要按 `session_id` 幂等覆盖，用户事实按 `(user_id, fact_key)` Upsert 防冲突。
3. **工具调用与并发保障 (`src/tools/`, `src/db/redis.py`)**
   - `RedisSessionLock`（`SET NX EX` + 原子 Lua 释放）保证单会话并发安全，支持可选 `request_id` 幂等去重。
   - 统一 `safe_execute_tool` 包装：单工具超时控制、参数校验/运行异常转结构化 `ToolMessage`（不崩图）、只读工具短时 Redis 缓存、全量工具审计日志（`tool_audit_logs`）。
   - 内置 5 个首批工具：`search_knowledge_base`、`query_order_status`、`get_user_orders`、`create_support_ticket`（幂等防重）、`transfer_to_human`。

## 快速启动

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动 Redis & PostgreSQL (pgvector)

```bash
docker compose up -d
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 在 .env 中填入你的 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 与 EMBEDDING_* 配置
```

### 4. 初始化垂直知识库示例数据

```bash
uv run python -m scripts.seed_kb
```

### 5. 启动后端服务与 Web UI

**Windows 一键启动（推荐）**：直接双击根目录的 `start.bat`（自动检查 `.env`、拉起 Docker 容器、补齐依赖，并在独立窗口启动后端 `8000` + 前端 `3000` 且自动打开浏览器）。

也可手动分终端启动：

```bash
# 1）启动 FastAPI 后端（默认 8000 端口）
uv run uvicorn src.api.main:app --reload --port 8000

# 2）启动 Web UI 前端（默认 3000 端口，自动代理 /api/* 到后端）
cd frontend
pnpm install
pnpm dev
```

打开浏览器访问 `http://localhost:3000` 即可使用控制台（含控制台概览、知识库管理与检索测试、多轮客服对话与 L1/L3 记忆检视、以及左侧菜单栏的 **设置** 页面配置 LLM / Embedding / Reranker 模型）。

也可以直接用 CLI 终端调试：

```bash
uv run python -m scripts.cli
```

### 6. 运行自动化测试（无需外部 LLM Key 或 Docker）

```bash
uv run pytest -v
```
