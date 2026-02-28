# PLANNER_GUIDE.md — 规划器使用指南

> Luna 创建 plan 前必读此文件。这是规划器所有功能的参考手册。

## Step Schema

每个 step 支持以下字段：

```json
{
  "title": "步骤标题（简短，显示在规划图上）",
  "prompt": "详细的执行指令（给子 agent 看的）",
  "depends_on": [0],
  "timeout_minutes": 15,
  "step_type": "conditional",
  "branches": [
    {"name": "重试", "next_steps": [1]},
    {"name": "通过", "next_steps": [3]}
  ]
}
```

> ⚠️ 注意：不要填 model 字段！Planner 会自动根据 prompt 内容选择最优模型。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| title | string | ✅ | 简短标题，50 字以内，显示在规划图节点上 |
| prompt | string | ✅ | 子 agent 的完整执行指令，越具体越好 |
| depends_on | int[] | ❌ | 依赖的步骤索引（0-based），空=无依赖可并行 |
| timeout_minutes | int | ❌ | 超时分钟数，不填则自动预估 |
| model | string | ❌ | **通常不要填！** Planner 会根据 prompt 关键词自动选择最优模型。只有当你确定需要 override 时才填 |
| step_type | string | ❌ | 步骤类型，可选值：`"conditional"`（条件节点）。不填则为普通步骤 |
| branches | object[] | ❌ | 条件节点的分支定义，只有 `step_type="conditional"` 时才需要。每个分支包含 `name` 和 `next_steps` |

## 条件节点与分支 (2026-02-28 新增)

### 什么是条件节点

条件节点允许根据执行结果选择不同的后续路径，支持：
- **分支选择**：根据条件选择不同的下一步
- **循环重试**：失败时回到前面的步骤重新执行
- **多路径**：一个分支可以指向多个后续步骤

### Branch Schema

```json
{
  "name": "分支名称",
  "next_steps": [2, 3]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | ✅ | 分支名称，显示在依赖图上（如"重试"、"通过"、"失败"） |
| next_steps | int[] | ✅ | 该分支指向的后续步骤索引（0-based）。可以指向更早的步骤形成循环 |

### 循环机制

当 `next_steps` 包含小于等于当前步骤编号的步骤时，会触发循环：
- 目标步骤被重置为 `pending` 状态
- `execution_count` 递增（用于追踪循环次数）
- 最大循环次数：10 次（防止无限循环）

### 条件节点示例

#### 简单重试循环
```json
[
  {
    "title": "执行任务",
    "prompt": "尝试执行任务，如果失败则报告失败原因",
    "depends_on": []
  },
  {
    "title": "检查结果",
    "prompt": "检查任务结果。如果成功选择'通过'，失败选择'重试'。\n使用命令：python3 scripts/emit_event.py step.done --task-id YOUR_TASK_ID --branch '通过' 或 '重试'",
    "depends_on": [0],
    "step_type": "conditional",
    "branches": [
      {"name": "重试", "next_steps": [0]},
      {"name": "通过", "next_steps": [2]}
    ]
  },
  {
    "title": "完成",
    "prompt": "任务成功完成，生成最终报告",
    "depends_on": [1]
  }
]
```

#### 多分支部署流程
```json
[
  {
    "title": "构建",
    "prompt": "构建应用",
    "depends_on": []
  },
  {
    "title": "测试",
    "prompt": "运行测试套件",
    "depends_on": [0],
    "step_type": "conditional",
    "branches": [
      {"name": "失败", "next_steps": [0]},
      {"name": "通过", "next_steps": [2]}
    ]
  },
  {
    "title": "部署到 staging",
    "prompt": "部署到 staging 环境",
    "depends_on": [1]
  },
  {
    "title": "验证 staging",
    "prompt": "验证 staging 环境",
    "depends_on": [2],
    "step_type": "conditional",
    "branches": [
      {"name": "失败", "next_steps": [2]},
      {"name": "通过", "next_steps": [4]}
    ]
  },
  {
    "title": "部署到 production",
    "prompt": "部署到生产环境",
    "depends_on": [3]
  }
]
```

### 分支选择方法

条件节点的子 agent 必须使用 `emit_event.py` 报告选择的分支：

```bash
cd /home/ubuntu/.openclaw/workspace && \
python3 scripts/emit_event.py step.done \
  --task-id YOUR_TASK_ID \
  --result "检查结果：测试通过" \
  --branch "通过"
```

参数说明：
- `--task-id`: 当前任务 ID（从环境变量或上下文获取）
- `--result`: 步骤执行结果描述
- `--branch`: 选择的分支名称（必须匹配 branches 中定义的 name）

### 依赖图可视化

条件节点在依赖图中的特殊显示：
- **节点形状**：条件节点用菱形表示
- **分支箭头**：每个分支用不同样式的箭头表示
  - 选中的分支：蓝色实线
  - 未选中的分支：灰色虚线
  - 未决定的分支：灰色点线
- **分支标签**：分支名称显示在条件节点内部，与对应箭头对齐
- **循环箭头**：反向箭头（指向更早步骤）用折线在节点上方绕过

### 注意事项

1. **分支名称要清晰**：使用"通过/失败"、"成功/重试"等明确的名称
2. **循环要有退出条件**：确保不会无限循环（系统会在 10 次后强制停止）
3. **Prompt 要说明如何选择分支**：明确告诉子 agent 在什么条件下选择哪个分支
4. **循环会增加成本**：每次循环都会重新执行步骤，消耗 token 和时间

## 🚨 模型选择规则

**默认行为：不填 model 字段，让 Planner 自动推断。**

Planner 内置了 `_estimate_model()` 函数，会根据步骤的 title + prompt 中的关键词自动选择：
- 架构/设计/重构类关键词 → Claude Opus (T4)
- 代码/实现/测试/PR 类关键词 → Claude Sonnet (T3)
- 中文写作/翻译/调研类关键词 → Kimi (CN)
- 其他 → Gemini Flash (T2，最便宜)

**什么时候才需要显式填 model：**
- 自动推断选错了模型（比如中文代码任务被分到 Kimi 但你需要 Sonnet）
- 特殊需求（比如必须用 Opus 做深度分析）

**❌ 错误做法：** 每个步骤都填 `"model": "api-proxy-claude/claude-sonnet-4-6"`
**✅ 正确做法：** 省略 model 字段，让 Planner 自动选择

## 模型分层参考 (2026-02-23 更新)

### 分层模型（全部走 API Proxy，⛔ 禁用直连官方）

| 层级 | 模型 ID | 适用场景 | Input $/M | Output $/M |
|------|---------|---------|-----------|------------|
| T4 最终解决者 | api-proxy-claude/claude-opus-4-6-thinking | 复杂架构、深度 debug、关键决策 | 0.475 | 2.375 |
| T3 代码/复杂 | api-proxy-claude/claude-sonnet-4-6 | 代码实现、修复、测试、PR | 0.285 | 1.425 |
| T2 默认 | api-proxy/gemini-3-flash-preview | 调研、总结、格式转换、批量处理 | 0.051 | 0.405 |
| T2 备选 | api-proxy-claude/claude-haiku-4-5-20251001 | 同上，Claude 系列轻量 | 0.058 | 0.288 |
| 中文 | api-proxy/kimi-k2.5 | 中文写作、翻译、中文调研 | 0.184 | 0.920 |

### 选择原则

1. **默认用 gemini-3-flash** — 最便宜，大部分子任务够用
2. **代码任务用 sonnet-4.6** — 性价比最优的代码模型
3. **只有架构/重构/深度 debug 才上 opus** — 最后的解决者
4. **中文任务优先 Kimi** — 中文理解和生成质量更好
5. **⛔ 绝对不用 anthropic/ 前缀的直连模型** — 比 aiberm 贵 4x

### 典型场景 → 模型映射

| 场景 | 推荐模型 | timeout |
|------|---------|---------|
| 调研/搜索/总结 | api-proxy/gemini-3-flash-preview | 15 |
| 写中文文档/报告 | api-proxy/kimi-k2.5 | 15 |
| 简单计算/数据处理 | api-proxy/gemini-3-flash-preview | 5 |
| 格式转换/文本处理 | api-proxy/gemini-3-flash-preview | 5 |
| 写代码/实现功能 | api-proxy-claude/claude-sonnet-4-6 | 15 |
| 测试/验证 | api-proxy-claude/claude-sonnet-4-6 | 10 |
| 复杂代码重构 | api-proxy-claude/claude-opus-4-6-thinking | 30 |
| 架构设计/方案分析 | api-proxy-claude/claude-opus-4-6-thinking | 30 |

## 依赖模式

### Fan-out（并行展开）
多个独立步骤同时执行：
```json
[
  {"title": "调研 A", "depends_on": []},
  {"title": "调研 B", "depends_on": []},
  {"title": "调研 C", "depends_on": []}
]
```

### Fan-in（汇聚）
多个步骤完成后汇总：
```json
[
  {"title": "调研 A", "depends_on": []},
  {"title": "调研 B", "depends_on": []},
  {"title": "汇总报告", "depends_on": [0, 1]}
]
```

### Pipeline（串行流水线）
每步依赖上一步：
```json
[
  {"title": "设计方案", "depends_on": []},
  {"title": "写代码", "depends_on": [0]},
  {"title": "写测试", "depends_on": [1]}
]
```

### Diamond（菱形）
展开 → 并行 → 汇聚 → 后续：
```json
[
  {"title": "需求分析", "depends_on": []},
  {"title": "前端实现", "depends_on": [0]},
  {"title": "后端实现", "depends_on": [0]},
  {"title": "集成测试", "depends_on": [1, 2]}
]
```

### Loop（循环，2026-02-28 新增）
条件节点 + 反向分支：
```json
[
  {"title": "执行任务", "depends_on": []},
  {
    "title": "检查结果",
    "depends_on": [0],
    "step_type": "conditional",
    "branches": [
      {"name": "重试", "next_steps": [0]},
      {"name": "通过", "next_steps": [2]}
    ]
  },
  {"title": "完成", "depends_on": [1]}
]
```

## Prompt 写作规范

### 好的 prompt 包含：
1. **明确的目标** — 这一步要产出什么
2. **具体的约束** — 文件路径、格式要求、技术栈
3. **验证标准** — 怎么算完成（测试通过、文件生成等）
4. **不要做什么** — 明确边界，避免越界
5. **（条件节点）分支选择逻辑** — 在什么条件下选择哪个分支

### 示例

**普通步骤：**
```
调研 Balatro 的 Boss Blind 机制：
- 阅读 /tmp/Immolate/lib/cache.cl 中的 BOSSES 数组定义
- 阅读 /tmp/Immolate/lib/functions.cl 中的 next_boss() 函数
- 输出：Boss 选择算法的伪代码 + 每个 Boss 的效果列表
- 不要修改任何代码，只做调研
```

**条件节点：**
```
检查测试结果并选择分支：
- 读取上一步的测试输出
- 如果所有测试通过，选择'通过'分支
- 如果有测试失败，选择'重试'分支
- 使用命令报告：
  cd /home/ubuntu/.openclaw/workspace && \
  python3 scripts/emit_event.py step.done \
    --task-id YOUR_TASK_ID \
    --result "测试结果：X 个通过，Y 个失败" \
    --branch "通过" 或 "重试"
```

### 反模式

❌ "帮我搞一下 Boss 的逻辑" — 太模糊，agent 不知道要做什么
❌ "调研并实现 Boss 系统" — 一步做两件事，应该拆成两步
❌ 不给文件路径 — agent 会浪费时间找文件
❌ 不说验证标准 — agent 不知道什么时候算完成
❌ 条件节点不说明如何选择分支 — agent 不知道选哪个

## 超时预估规则

自动预估逻辑（`_estimate_timeout`）：
- **5 分钟**: compute, calculate, query, summarize, report, format
- **15 分钟**: implement, write, create, build, fix, test, debug
- **30 分钟**: refactor, research, investigate, analyze, audit, review

手动指定的场景：
- 需要大量网络请求（爬虫、API 调用）→ 适当加长
- 已知是简单任务但关键词触发了高估 → 手动降低
- 涉及大文件处理 → 适当加长

## 临时群聊（Task Chat）

每个步骤会自动创建一个临时群聊，用于：
- 实时查看子 agent 的执行进度
- 流式卡片显示 tool calls 和输出
- 任务完成后可以查看完整日志

### 临时群特性
- **自动创建** — 步骤启动时自动创建，命名格式：`Task {task_id} Step {N}: {title}`
- **自动添加成员** — Carl 会被自动添加到群聊（通过 `LARK_OWNER_ID` 环境变量）
- **只读模式** — 默认只有机器人能发消息，成员只能查看
- **流式更新** — 子 agent 的输出会实时推送到群聊卡片

### 环境变量配置
```bash
# .env
LARK_OWNER_ID=ou_35f664e694dd100adf97b867e68e1d3a  # Carl 的 open_id
```

### 已知问题（2026-02-23）
- ⚠️ 临时群创建后成员可能为空（fallback 逻辑未生效）
- 临时解决方案：手动添加成员或等待修复

## 性能数据（2026-02-23 测试）

基于 plan-0223-28 的实测数据：

| 模型 | 消息数 | Tool calls | 耗时 | 适用场景 |
|------|--------|-----------|------|---------|
| Kimi | 57 | 29 | 57s | 中文调研，需要多轮迭代 |
| Opus Thinking | 8 | 3 | 2m | 复杂架构设计，一次性输出高质量 |
| Flash | 12-27 | 5-13 | 42s-3m | 代码实现、数据分析 |

**关键发现**：
- Opus 效率最高（消息数最少），适合复杂任务
- Flash 快速执行，适合大部分常规任务
- Kimi 需要更多轮次，但中文质量好

## 推进机制（事件驱动）

**核心原理：步骤完成后立即推进，不依赖 cron 轮询。**

1. **子 agent 完成任务** → emit `step.done` 事件到 events 表
2. **check-contracts 消费事件** → 立即检查依赖关系，启动下一批 ready 步骤
3. **Cron 只是兜底** — 每 2 分钟运行一次，防止事件丢失或 check-contracts 崩溃

**时间线示例**：
- 00:00:00 — Step 1 启动
- 00:05:23 — Step 1 完成，emit step.done
- 00:05:24 — check-contracts 消费事件，发现 Step 2/3 ready，立即启动
- 00:05:25 — Step 2/3 开始运行（不需要等到 00:06:00 的 cron）

**❌ 错误说法**："等待 cron 推进" — 这会让用户以为要等 2 分钟
**✅ 正确说法**："check-contracts 会自动启动下一批步骤" — 强调实时性

## CLI 命令参考 (2026-02-28 更新)

### 基础命令

```bash
# 创建 plan
luna-os plan init <chat_id> <goal> <steps_json>

# 启动 plan
luna-os plan start <chat_id>

# 查看 plan 状态
luna-os plan show <chat_id>

# 列出所有 plans
luna-os plan list

# 取消 plan
luna-os plan cancel <chat_id>

# 暂停 plan
luna-os plan pause <chat_id>

# 恢复 plan
luna-os plan resume <chat_id>
```

### 步骤管理

```bash
# 标记步骤完成
luna-os plan step-done <chat_id> <step_num> <result>

# 标记步骤失败
luna-os plan step-fail <chat_id> <step_num> <error>
```

### 高级命令

```bash
# 重新规划（替换 pending 步骤）
luna-os plan replan <chat_id> <new_steps_json>

# 追加步骤（保留现有 pending 步骤）
luna-os plan replan <chat_id> <new_steps_json> --append

# 手动推进（检查并启动 ready 步骤）
luna-os plan advance <chat_id>

# 检查所有 plans 并推进
luna-os plan check-contracts

# 根据 task_id 查找 plan
luna-os plan find-by-task <task_id>
```

### 依赖图生成

```bash
# 生成并发送依赖图到群聊（内部使用）
# 通过 planner._send_plan_graph(plan, chat_id) 调用
```

## 注意事项

1. **并行步骤会看到完整 plan** — 每个子 agent 都知道其他步骤在做什么，不会重复劳动
2. **已完成步骤的结果会传递** — 后续步骤能看到前置步骤的产出摘要
3. **超时不会直接失败** — 超时后进入 waiting 状态，可以人工恢复
4. **model 字段传给 Gateway** — 子 agent 会用指定的模型运行
5. **临时群可能需要手动添加成员** — 如果自动添加失败，需要手动处理
6. **check-contracts 依赖环境变量** — 确保 crontab 正确加载 `.env` 文件
7. **条件节点必须报告分支选择** — 使用 `emit_event.py --branch` 参数
8. **循环有最大次数限制** — 防止无限循环（最多 10 次）

## 故障排查

### 临时群没有成员
**症状**：群聊创建了但看不到
**原因**：`LARK_OWNER_ID` 未设置或 fallback 逻辑未生效
**解决**：
1. 检查 `.env` 文件是否有 `LARK_OWNER_ID`
2. 手动添加成员：
```python
import requests, os
app_id = os.environ['LARK_APP_ID']
app_secret = os.environ['LARK_APP_SECRET']
auth_resp = requests.post(
    'https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal',
    json={'app_id': app_id, 'app_secret': app_secret}
)
tenant_token = auth_resp.json()['tenant_access_token']
url = f'https://open.larksuite.com/open-apis/im/v1/chats/{chat_id}/members'
headers = {'Authorization': f'Bearer {tenant_token}'}
body = {'id_list': ['ou_35f664e694dd100adf97b867e68e1d3a']}
requests.post(url, headers=headers, json=body, params={'member_id_type': 'open_id'})
```

### check-contracts 不运行
**症状**：plan 创建后一直是 draft 状态
**原因**：crontab 环境变量未加载
**解决**：
```bash
# Crontab 正确写法
*/2 * * * * bash -c 'set -a; source /home/ubuntu/.openclaw/workspace/.env; set +a; /home/ubuntu/.local/bin/luna-os plan check-contracts' >> /tmp/planner-check.log 2>> /tmp/planner-check-debug.log
```

### Token 统计异常
**症状**：任务的 token 数远超实际使用
**原因**：重复累加 bug（已在 PR #51 修复）
**解决**：更新到最新版本的 luna-os

### 条件节点不推进
**症状**：条件节点完成后，后续步骤没有启动
**原因**：子 agent 没有报告分支选择
**解决**：
1. 检查子 agent 是否调用了 `emit_event.py --branch`
2. 检查分支名称是否匹配 branches 定义
3. 查看 events 表确认事件是否正确记录

### 循环无限执行
**症状**：步骤一直重复执行
**原因**：条件节点总是选择循环分支
**解决**：
1. 检查条件判断逻辑是否正确
2. 确认有退出条件（如检查 execution_count）
3. 系统会在 10 次后自动停止

## 进度通知规则 (2026-02-24)

**Luna 收到 plan 进度更新时，必须用 `message` tool 发到 plan.chat_id，不能在主 session 里回复。**

**错误示例**：
```
# Luna 在主 session（Telegram）里回复
"plan-0224-31 进展顺利，Step 4 已完成"
→ 消息发到了 Telegram，而不是创建 plan 的飞书群聊
```

**正确做法**：
```python
# 1. 查询 plan 获取 chat_id
plan = store.get_plan('plan-0224-31')

# 2. 用 message tool 发到原始群聊
message(
    action="send",
    channel="feishu",
    target=f"chat:{plan.chat_id}",
    message="plan-0224-31 进展顺利，Step 4 已完成"
)

# 3. 主 session 回复 NO_REPLY
NO_REPLY
```

**为什么**：
- Plan 在群聊 A 创建，但 Luna 可能在群聊 B 或 Telegram 收到 wake event
- 主 session 的 delivery context 不等于 plan 的原始 chat_id
- 必须显式指定目标，不能依赖自动路由
