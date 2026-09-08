# GLM provider adaptation — 本地离线报告

本轮完成共享请求策略、S49 重试/用量缺失护栏、图片提示与 E2E 密钥隔离的本地修改。
六套指定离线测试、`e2e.test_safety`、两个新增测试模块全部通过，九条命令均 exit 0。
基线为 `9909dbf42a7aacdd57f69ec5811e1dee214f8cb6`，分支 `main`，工作树已有 S58 改动。
**本轮付费模型调用为 0；真实 provider 调用为 0；没有运行 `e2e.run`、探针或 `pnpm test`。**
没有读取真实 `.env` / `.e2e_key`，没有访问生产、暂存、提交、推送或部署。
官方文档只通过公开网页读取；下文生产探针事实来自用户提供的信息，不是本轮复验。

## Provider 表与配置

`keytao_bot/utils/llm_policy.py:15` 的前缀表驱动 `with_chat_policy`；旧名
`with_deepseek_chat_policy` 保留为同一函数的兼容别名，7 个原调用点无需改动。
前缀匹配会去除首尾空格并忽略大小写。

| 模型前缀 | `thinking=False` | `thinking=True` | effort / sampling | `json_output=True` |
|---|---|---|---|---|
| `deepseek-` | `extra_body.thinking={"type":"disabled"}`，删除顶层 effort | `extra_body.thinking={"type":"enabled"}` | 默认 `high`；接受 `low/high/max`；开启思考时移除 temperature/top_p/presence_penalty/frequency_penalty | `response_format={"type":"json_object"}` |
| `glm-` | `extra_body.thinking={"type":"disabled"}`，删除顶层 effort | `extra_body.thinking={"type":"enabled"}` | 默认 `high`；接受 `low/high/max`；保留原 sampling 参数 | `response_format={"type":"json_object"}` |
| 其他前缀 | 透传 | 透传 | 不添加、删除或验证供应商选项 | 不添加 JSON 格式约束 |

GLM effort 枚举依据 [GLM-5.3 官方参数表](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3)
（本轮读取日期 2026-09-08）；该页示例也同时发送 thinking、effort 和 temperature。
[Coding Plan 快速开始](https://docs.bigmodel.cn/cn/coding-plan/quick-start)列出的 OpenAI base URL 是
`https://open.bigmodel.cn/api/coding/paas/v4`。SDK 追加 `/chat/completions`；本轮没有向该端点发请求。

**文档与已提供实测的差异：** GLM-5.3 通用文档写明仅支持 enabled，并建议用 low 降低推理量；
用户提供的 Coding endpoint 探针则表明 disabled 请求返回 200 且有正文。
本轮按明确需求 A 保留 disabled 形状，不擅自换成其他控制参数。
离线检查只证明序列化请求符合该需求，不能证明 Coding 服务端实际关闭思考，也不能保证该形状今后仍被接受。
不需要付费调用即可完成此次代码/离线验收，因此未申请或执行额外探针。

配置解析保持原样，没有新增环境变量：
主模型继续使用 `OPENAI_MODEL`（`chat_adapters.py:113`）；意图模型的既有覆盖配置最终回退主模型
（`chat_routing.py:452`）；审查模型仍优先 `KEYTAO_REVIEW_MODEL`，再取 `OPENAI_MODEL`
（`keytao_review.py:498`）。没有读取本地/生产实际配置来假定覆盖值。

## GLM 各调用点的完整请求字段

下表与代码块针对各调用点**实际解析出的模型为 `glm-5.3`** 的情形。
变量仍取原调用点的运行时值；用户消息、提示词和工具结果没有改写。
SDK 参数中的 `extra_body.thinking` 会被序列化为 HTTP JSON 顶层 `thinking`，不会发送名为 `extra_body` 的 JSON 字段。

| 操作 | 原调用点 | max_tokens | temperature | 控制 |
|---|---|---|---|---|
| `command_intent` | `keytao_bot/plugins/chat_routing.py:2725` | 260 | 0.0 | disabled；JSON |
| `word_query_intent` | `keytao_bot/plugins/chat_routing.py:2789` | 180 | 0.0 | disabled；JSON |
| `entity_knowledge` | `keytao_bot/utils/keytao_review.py:4511` | 700 | 0.0 | disabled；JSON |
| `semantic_pronunciation` | `keytao_bot/utils/keytao_review.py:4738` | 450 | 0.0 | disabled；JSON |
| `memory_summary` | `keytao_bot/plugins/openai_chat.py:1371` | `MEMORY_SUMMARY_MAX_TOKENS`，既有默认 700 | 0.2 | disabled；普通文本 |
| `main_agent` | `keytao_bot/harness/orchestrator.py:1287` | `current_max_tokens` | `self._runtime.temperature` | enabled；high，S49 重试 low |
| 批量审查 | `keytao_bot/utils/keytao_batch_review.py:1404` | `current_max_tokens` | `min(config["temperature"], 0.2)` | enabled；high；JSON |

前五项是源码 AST 枚举出的全部 `thinking=False` 调用点。以下是 HTTP body 的完整字段表达式；
每段的提示词变量分别来自表中对应函数，并非抓取的生产请求。

```python
# command_intent
{
    "model": WORD_QUERY_INTENT_MODEL,
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
    "max_tokens": 260, "temperature": 0.0,
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
}

# word_query_intent
{
    "model": WORD_QUERY_INTENT_MODEL,
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
    "max_tokens": 180, "temperature": 0.0,
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
}

# entity_knowledge
{
    "model": config["model"],
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}],
    "max_tokens": 700, "temperature": 0.0,
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
}

# semantic_pronunciation
{
    "model": config["model"],
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}],
    "max_tokens": 450, "temperature": 0.0,
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
}

# memory_summary
{
    "model": OPENAI_MODEL,
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
    "max_tokens": MEMORY_SUMMARY_MAX_TOKENS, "temperature": 0.2,
    "thinking": {"type": "disabled"},
}

# main_agent
{
    "model": self._runtime.model, "messages": messages,
    "max_tokens": current_max_tokens, "temperature": self._runtime.temperature,
    "thinking": {"type": "enabled"}, "reasoning_effort": current_reasoning_effort,
    **({"tools": tools, "tool_choice": "auto"} if tools else {}),
}

# batch_review
{
    "model": config["model"],
    "messages": [{"role": "system", "content": prompt},
                 {"role": "user", "content": user_content}],
    "max_tokens": current_max_tokens, "temperature": min(config["temperature"], 0.2),
    "thinking": {"type": "enabled"}, "reasoning_effort": "high",
    "response_format": {"type": "json_object"},
}
```

主循环初始上限仍由 `max(runtime.max_tokens, min(line_count * 200 + 500, runtime.max_tokens_cap))` 决定。
批量审查初始上限仍为 `min(max(config["max_tokens"], len(items) * 500), config["max_tokens_cap"])`。
本轮没有提高 token 上限，也未把批量审查自己的重试策略改成 S49。

审计另找到共享策略之外的文本调用：`keytao_bot/utils/word_discovery.py:640`
的候选抽取直接调用模型，字段为 model、messages、temperature=0.2、max_tokens=审查配置值，未发送 thinking/effort。
它不属于现有 `thinking=False` 调用点，本轮按“same call sites”范围保持原样。
`keytao_bot/utils/image_input.py:957` 使用独立视觉 client 与其既有策略，也保持原样。

## S49 与 usage 形状

`llm_policy.py:108` 读取 `completion_tokens_details`（兼容 `output_tokens_details`），
`:115` 读取其中的 `reasoning_tokens`；支持字典及 SDK 属性对象。
缺失/null 的字段不会进入 metrics 字典，继续保留未知语义。

`orchestrator.py:541` 先要求 `finish_reason="length"`、正文为空且没有工具调用。
若计数齐全，仍按 reasoning/output 至少 95% 判断；明确报告 reasoning=0 不等于未知。
若输出计数或 reasoning 计数缺失，对表内模型保守进入有界耗尽处理；使用配置模型作为依据，
响应没有 model 字段也不会让主循环失去此护栏。未知供应商的缺失用量行为保持原样。

GLM 主循环的第一轮使用 enabled + high；首次符合耗尽条件后压缩旧上下文，
保留当前请求及同轮结果，第二轮发送 enabled + low，`max_tokens` 保持原值。
若连续两次为空，沿用现有可信记录/确定性回退与停止流程，不进入通用长度翻倍重试。
GLM 使用 low 的依据是上述官方枚举；未采用未记录的 effort 值，也未为重试改成 disabled。

`test_glm_orchestrator.py` 以 fake client 驱动真实 `_run_loop`：
两种完整计数对象形状、缺 details、空 details、null details、null reasoning、缺全部 usage，
均断言两次调用、high→low、1800→1800、第二次完整请求体及零工具派发。
可见正文、已有工具调用、正常 stop、明确零 reasoning 均有反向断言。
该测试不证明真实供应商工具续轮或端到端路由效果。

## E2E 密钥隔离与文案审计

`e2e/run.py:349` 强制使用独立 `E2E_OPENAI_API_KEY`。
缺 key 在读取 dotenv 前直接抛出 `SafetyViolation`，明确说明 bot/provider fallback 已禁用。
有 key 时先比较进程环境，然后比较 repo、CWD、`args.next_dir` 三处顶层 `.env` / `.env.*`
中的 `OPENAI_API_KEY`，包括非活动环境变体；同钥或无法读取校验文件均拒绝启动，错误不包含密钥值。
变量名大小写、首尾空白和 python-dotenv 插值均纳入比较。
dotenv 中的 bot key 只用于拒绝同钥，不能成为测试凭据来源。

解析采用与 NoneBot 一致的逐文件 dotenv 语义：每个文件单独插值后合并；
插值可使用当前文件此前定义或进程环境，不能凭空继承另一文件中的别名。
8 个新增合成配置测试覆盖缺 key、进程/各目录/变体同钥、插值、读取失败和独立 key 放行。
本轮运行这些测试只允许临时 fixture 文件，未读取实际 bot 或 next 配置。

| 审计位置 | 处理 |
|---|---|
| `chat_adapters.py:704` | 聊天提示改为“当前主模型不支持图片输入”，保留 S58 已有句末与返回形式 |
| `README.md:121` | 图片架构介绍中的两处 DeepSeek 主模型称呼改为“主模型” |
| `e2e/README.md:124` | 删去 bot key 作为测试 key 的旧说明，明确独立 E2E key 与拒绝同钥规则 |
| `image_input.py:137` | 保留：仅当管理员明确将 DeepSeek 配为视觉代理时的校验错误，未宣称当前主模型品牌 |
| `web-search/tools.py:1400` | 保留：搜索参数中的 DeepSeek API 示例，未宣称当前主模型品牌 |
| `keytao_review.py:457`、内部函数名与历史报告 | 保留兼容默认值、内部标识与历史事实；S58 报告未改 |

## 验证记录与工作树边界

先红后绿：新增 SDK `MockTransport` 矩阵在旧策略上出现 3 个 GLM 子用例错误，
断言显示缺 thinking/effort/response_format；新增 S49 模块在旧逻辑上出现 12 个子用例失败。
修复后 SDK 精确请求矩阵（含 DeepSeek 对照、unknown 透传）与主循环 fake 测试通过。
密钥护栏也由合成配置在旧实现上复现后修复。

规定命令以独立 Python 子进程运行，启动器在 `/tmp/keytao-glm-offline/run_checks.py`。
它使用清理过的进程环境与临时 `sitecustomize.py`：拒绝 socket 网络访问、阻止真实密钥文件打开，
NoneBot 的真实 dotenv 返回空配置，临时合成 dotenv 保留真实解析。
已单独验证网络与真实密钥文件读取会被拦截；没有用现有 S58 通过记录替代本轮执行。
日志在 `/tmp/keytao-glm-offline/`，逐命令退出码记录在 `results.json`。

| 实际命令 | 本轮结果 |
|---|---|
| `.venv/bin/python test_state_machine.py` | 2019/2019；exit 0 |
| `.venv/bin/python test_memory_safety.py` | 407 tests；OK；exit 0 |
| `.venv/bin/python test_security_fixes.py` | 268/268；exit 0 |
| `.venv/bin/python test_review_gate.py` | 443/443；exit 0 |
| `.venv/bin/python test_llm_policy.py` | 11 tests；OK；exit 0 |
| `.venv/bin/python test_word_discovery.py` | 290/290；exit 0 |
| `.venv/bin/python -m e2e.test_safety` | 107 tests；OK；exit 0 |
| `.venv/bin/python -m unittest test_glm_orchestrator` | 3 tests；OK；exit 0 |
| `.venv/bin/python -m unittest e2e.test_glm_key_guard` | 8 tests；OK；exit 0 |

`git diff --check` 已通过。独立静态复核未发现本轮新增阻断缺陷；这不是跨模型验证或生产证明。
本轮只修改 README、E2E README/启动护栏、共享 policy、orchestrator 的 provider/用量/重试局部、
图片提示与 policy 测试，新增两个测试模块和本报告。
预检保存了 31 个原有 dirty/untracked 文件的指纹；其中只有本任务必需的
`e2e/README.md`、`keytao_bot/harness/orchestrator.py`、`keytao_bot/plugins/chat_adapters.py`
发生变化，其余原有文件指纹一致。没有修改 `e2e/test_safety.py`、S58 测试或 `REPORT-move-to-code.md`。
最终还在内存中撤去这 3 个重叠文件的本任务局部修改，重算指纹与预检逐一相等，确认原有 S58 内容完整保留；
该核对没有回写或撤销工作树。

本报告仅覆盖本地静态与离线执行证据；生产切换、真实 GLM 回复质量、实际 thinking 行为、
工具续轮兼容性和 S58 真实 FULL 验收均未在本轮验证。工作保持未提交、未部署。
