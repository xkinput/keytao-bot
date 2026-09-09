# S59 — 常用度查询与零工具调用上限

基线：`5fbc6999dbd62e97052f6c78df5537898007c043`，`main`，初始工作树干净。
本轮为本地实现和离线验证；没有暂存、提交、推送、部署或访问生产。
**付费模型调用 0；真实 provider 调用 0。没有读取 `.e2e_key`、真实 `.env` 或生产密钥，没有运行 `pnpm test` 或 `e2e.run`。**
不需要真实模型调用即可完成本轮验证。

## 路由、证据与表格

`keytao_bot/utils/commonness_query.py:21` 接受完整的常用度查询：

- `词频排序：耶博 伊莎贝拉 夜泊 一身本领`
- `请给这几个词的词频排序：耶博 伊莎贝拉 夜泊 一身本领`
- `按常用度排序：耶博、夜泊`
- `哪个更常用：耶博，夜泊`
- `常用度对比：耶博和夜泊`
- `耶博 和 夜泊 哪个更常用？`

词表为 2–12 个不同中文词，每词 1–32 字；空白、`、`、中英文逗号和连接词 `和` 可分隔。
支持 `夜泊和耶博、伊莎贝拉` 混用形式。标点分组内若有明确空白，按空白保留字面词，
所以含“和”的完整词可使用 `共和国 夜泊` 这种空白形式；无空白的组内“和”按连接词解释。
拒绝重复、超限、单词、报告/否定前缀、含分号的附带操作。纯词表只被读取，不取得写权限。

`openai_chat.py:3479` 的专用阶段位于会话初始化之后、所有 pending/意图分类之前
（阶段表 `:5709`），直接产生回答并结束该轮。流程指标增加 `word-commonness`。
它不进入主模型、意图模型、审词模型或网页回退。

共享入口 `word_commonness.py:37` 调用审查已经使用的 `_query_commonness_reference`
（`keytao_review.py:5260`）、`_reference_commonness_result`（`:5343`）和
`_compare_reference_commonness`（`:5568`）。数据库以 SQLite 只读方式打开，只查询
`word_commonness(word, corpus_frequency, part_of_speech, dictionary_presence_count)`。
语料频次来自 jieba；词典收录是读音资料的家族数，汉典词语/成语合为一家，另有
`large_pinyin`、`cedict`，范围 0–3（`pinyin_reference_build.py:559`）。本轮未重建或修改资料库。

完全沿用本地比较规则：两侧都有频次时，比值达到 2 才给严格方向，否则为接近；
单侧频次使用原有最低频次 10 与词典证据条件；尚不能判断时用词典家族差至少 2 判方向。
语料/词典信号权重仍为 0.75/0.25，词性、分项信号、解释分数均保留；排序不擅自按分数强排。
严格比较边的稳定拓扑排序提取为 `keytao_review.py:6235` 的共享函数，原审查链排序也使用它。
接近关系不具有传递性，故表格序号只是展示顺序，不能当成精确名次差。
真实比较规则可能构成环；fixture `(100,0) / (50,3) / (None,3)` 覆盖此情形，
保留词表、不标名次、标明证据冲突，也不推荐操作方向。

原审查可选的现代语义/网页补充依赖另行取得的证据，直接查询不编造此类证据，亦不触发其模型或网络回退。
找不到记录、资料库不可用、记录非法均逐词保留“无数据”；未知字段为 `None`，显示 `—`。
合法的词典收录零值仍为 `0`，不把缺失变成零。

S59 临时 SQLite fixture 的交付格式如下，数值为测试数据，不是生产词频：

```text
常用度排序（本地语料与词典）
名次 | 词 | 语料频次 | 词典收录 | 判定
1 | 伊莎贝拉 | 1000 | 3 | 有数据
2 | 夜泊 | 400 | 3 | 有数据
3 | 耶博 | 100 | 2 | 有数据
— | 一身本领 | — | — | 无数据
按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。词频是语料内计数。
可发送以下命令生成调整计划：
- 「把 夜泊 yebo 排到 耶博 yebo 前面」
```

## 放置建议与闭环

`commonness_query.py:71` 先用现有批量按词查询，复用 `_validated_word_lookup_entries`
检查完整词集、词条类型、编码和权重。缺失词调用现有 `keytao_encode`；前缀关系需要时也查询候选，
复用 `_candidate_codes_from_encode` 与 `_shared_candidate_chain_root`，不自行拼码。
同码同类型建议经 S57 `parse_same_code_reorder` 回读完整词/码；前缀放置建议经 S50
`parse_eviction_modified_add` 回读源词和目标词。编码查证整体最多等待 5 秒，失败仍交付本地表格。

renderer 使用 `render_executable_suggestion`；最终广告抽取必须与已验证命令逐条完全相等。
`CommonnessDelivery` 是当前任务上下文中的不可变交付事实，只匹配当前 actor/会话、查询词集和完整正文，
在每次入站串行处理开始时清除。它不创建 pending 确认、不改变授权、不接受裸“确认”，也不能由模型正文生成。
交付边界只接纳这份本轮查询生成的精确正文；改字、加命令、换用户或换请求均不匹配。
用户随后发送完整命令时，原 S57/S50 重新查询、规划和使用既有确认流程。

**B 的必要边界（REBUTTED，无效命令不能广告）：**
仅共享字符前缀并不足以证明能放置：目标现有码必须在源词的已核验候选内。
同一编码若同时共享多个词条类型，S57 无类型参数不能唯一定位，故不广告。
目标没有现有码或有多个现有码时，S50 同样不能定位；两个词都只算出候选链仍不足以产生可执行相对放置。
依据：`same_code_reorder.py:132` 要求唯一同码同类型关系；
`chat_commands.py:1675` 要求目标唯一现有码，`:1826` 要求目标在源词候选中。
这些情况保留表格，不伪造建议。独立 fixture 把旧建议送入真实 S57/S50，分别复现
“未能锁定唯一”和“不是当前服务端编码链中的候选”，修复后均无广告。

## 只读工具与提示词

`keytao-lookup/tools.py:819` 注册 `keytao_word_commonness`，与确定性路由调用同一共享入口。
Schema 如下；输入在打开数据库之前再次验证，额外参数不能进入工具本体。

```json
{
  "type": "function",
  "function": {
    "name": "keytao_word_commonness",
    "parameters": {
      "type": "object",
      "properties": {
        "words": {
          "type": "array",
          "items": {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[^\\r\\n\\t]+$"},
          "minItems": 1,
          "maxItems": 12,
          "uniqueItems": true
        }
      },
      "required": ["words"],
      "additionalProperties": false
    }
  }
}
```

返回对象含 `success / method / referenceAvailable / words / comparisons / evidenceLines / ordering / orderingNote`；
每行含 `word / corpusFrequency / dictionaryPresenceCount / partOfSpeech / known / referenceAvailable /
rank / verdict / score / signals`。最多 12 行、66 对比较，全量保留并列和未知证据。
`verdict` 为 `ranked / close / unknown`，缺失名次为 `null`。
fake-client 测试验证真实编排循环发出的工具 schema、参数、工具结果消息与最终数据回答。

`orchestrator.py:109` 修改前：

```text
只有当可信结果明确提供 suggestedCommand 时，才可逐字给出这一个命令；
若没有 suggestedCommand，只说明还缺少哪项具体信息。
```

修改后：

```text
只有核验结果给出了可执行命令时，才可逐字转述那一条命令；
不得自行发明、改写或追加命令。
若没有可执行命令，只说明还缺少哪项具体信息。
```

同轮 sweep 还改写 `authorization_grammar.py:3484` 与 `harness/tools.py` 的策略失败提示。
所有运行 `SKILL.md` 和工具 schema 均有五标识扫描测试。内部执行字段保留兼容，
模型结果序列化边界 `harness/tools.py:529` 把五标识中文化为“可执行命令 / 未执行原因 /
已核验目标 / 本次操作已拒绝 / 需要补充说明”，覆盖成功、失败、未注册工具和非 JSON 结果；不改执行器原对象。
`chat_render.py` 的重绘与最终断言共同禁止 `suggestedCommand / blockReason / boundTarget /
policyBlocked / requiresTextFollowUp` 进入用户回复。D 测试经真实 `_send_event_response` 捕获实际发送内容。
新增工具名 `keytao_word_commonness` 也加入交付重绘/拒绝检查，通用回答不能复述该内部名字。

## 75 秒和整轮两次上限

**不能把聚合日志当成三个已知的主模型响应。** `observability.py:197` 在每个受观测模型请求的
`finally` 累加次数和等待时长，失败也计数；75 秒是这些请求的合计等待，并非某一主回复的独占时间。
只凭用户提供的 `flow=general, model_calls=3, tool_calls=0` 无法还原各次 finish reason、逐次耗时、
SDK 内部重试次数和当时 pending 状态；本轮没有读取生产日志来补充不存在的证据。

普通无待确认上下文下，基线源码支持以下解释（推断，不冒充生产逐次 trace）：

1. 一次 `command_intent`：`chat_routing.py:2705` 原提示明确将词义/常用度比较返回 `none` 交给主模型；`:2725` 发分类请求。
2. 一次 `main_agent` 初始回答。
3. 一次 `main_agent` 的空白/截断重试，最终输出道歉及泄露字段名。

原消息带全角冒号，不能通过裸中文词表的筛选（基线 `chat_routing.py:201, :2312, :2833`），
因此通常不是第二种裸词分类器调用；群聊在 `openai_chat.py:1396` 跳过记忆摘要，后台摘要也隔离指标。
基线普通空白与 length 重试计数分离；缺用量的非受控模型和混合响应还可能绕过 S49，fake测试已复现。
这说明三次调用的可达路径，不能证明生产恰好走了哪个重试分支。

现在 `orchestrator.py:992` 将前置分类器次数计入整轮预算：此前用了 0/1/2 次，零工具主循环最多再用 2/1/0 次。
已有一次前置调用时，首个主请求即使用 S49 压缩及支持模型的 low effort。
第一次没有完整答案或可派发工具时，最多进行一次压缩重试，token 上限不翻倍；普通空白、截断正文、
缺失 reasoning usage、混合 stop/length、工具 JSON 截断和本地被拒的工具请求都不能借此开第三次。
本地拒绝不等于真实工具派发。已经实际调用工具的回合继续原工具处理和汇总预算。
`openai_chat.py:5614` 还阻止后置补充分类在零工具且已用两次时发第三次请求。
压缩保留当前请求和同轮可信结果，明确引用/附件不授权，并保留绑定、操作者和确认约束。
这约束受观测模型调用次数，不承诺供应商单次请求耗时或 HTTP 内部重试的绝对上限。

## 验证与交付状态

S59 独立入口为 `.venv/bin/python -m e2e.s59`，只启动四个 fixture/fake-client 模块，
不注册到真实 provider rig。实际执行通过 `.tmp-work/s59-offline/run_checks.py` 的清理环境启动器：
每个子进程隔离，socket 联网被拒绝，真实 dotenv 返回空配置，真实密钥文件打开被拒绝，
临时 SQLite/合成配置仍可正常使用。日志保存在 `/tmp/keytao-s59-offline/`。

初次路由复现因缺少专用 stage 失败；新增工具/提示和调用上限测试也分别先红后绿。
独立审查修复了跨类型歧义、分叉前缀错误广告、本地拒绝解除模型预算、截断工具 JSON 不压缩四个问题。
第一轮 state suite 为 2022/2023，唯一失败是旧 fake-model 测试要求 `1800→3600`，
与 E 的固定 token 预算冲突；现改为更严格的 `1800→1800`，两次调用和诚实未写入断言均保留。
原投影测试要求政策结果中的内部标识原样送模型，也按 D 改为中文副本，保留全部业务值并断言原对象未改。

第一轮记忆安全套件运行 407 个方法，出现 16 个失败子例和 1 个字段读取错误：
11 个失败子例和 1 个错误仍从模型消息读取旧英文键；1 个要求旧 `1800→3600` 预算；
3 个是最终拒绝异常的旧文本兼容；1 个是提示中的“执行器”及后续要求出现英文标识的旧断言。
最终提示改用“核验结果”，拒绝异常保留原兼容文本；测试只更新模型可见字段、固定预算和禁止字段的断言，
原业务值、授权、无写入和派发集合断言不变。10 个原失败方法单独复验全部通过。
首轮七条命令的退出记录和完整日志独立保存在 `results-first.json` 与 `*-first.log`，没有合并成最终通过记录。
其后的七命令全绿记录保存为 `results-second.json` 与 `*-second.log`；追加新增工具名的交付检查后，
再次冻结全部 20 个源码/测试/启动器文件并重跑七条命令，最终报告采用这一次完整执行。

可复现的本轮命令：

```sh
.venv/bin/python .tmp-work/s59-offline/run_checks.py
.venv/bin/python .tmp-work/s59-offline/run_checks.py -m e2e.s59
.venv/bin/python .tmp-work/s59-offline/run_checks.py -m unittest test_glm_orchestrator
```

首条在清理环境中分别启动下表的七条字面命令；第二条 S59 再以独立 Python 进程运行
路由/交付 9、证据 9、工具/提示 6、调用上限 11 个测试，共 35 个。第三条为原 GLM 请求矩阵 3 个测试。
S59 包含三词有数据、一词无数据、零模型路由、同码及缺失源词的 S50 命令解析闭环、
无共同链/超时、跨类型/分叉前缀拒绝、交付五字段及新增工具名防漏、前置/后置模型总预算等控制。

最终七条规定命令在同一冻结版本上全部 exit 0，启动器总 exit 0：

| 字面命令 | 最终结果 |
|---|---|
| `.venv/bin/python test_state_machine.py` | 2023/2023，0 failed |
| `.venv/bin/python test_memory_safety.py` | 407 tests，OK；200.692 秒 |
| `.venv/bin/python test_security_fixes.py` | 268/268，0 failed |
| `.venv/bin/python test_review_gate.py` | 443/443，ALL TESTS PASSED |
| `.venv/bin/python test_llm_policy.py` | 11 tests，OK |
| `.venv/bin/python test_word_discovery.py` | 290/290，ALL TESTS PASSED |
| `.venv/bin/python -m e2e.test_safety` | 107 tests，OK |
| `.venv/bin/python -m e2e.s59` | 35 tests，四个模块均 exit 0；fixture/fake-model only |
| `.venv/bin/python -m unittest test_glm_orchestrator` | 3 tests，OK |

最终退出记录为 `/tmp/keytao-s59-offline/results.json`；S59 独立结果为 `e2e.s59.log`。
`source-freeze.json` 覆盖 20 个源码/测试/启动器文件，结束时逐一重算无差异。
`git diff --check` 通过，HEAD 仍为 `5fbc6999dbd62e97052f6c78df5537898007c043`，暂存区为空。
独立审查为同模型审查，不称为跨模型或生产验证。

本轮完成的是本地实现、静态审查、指定离线套件和 S59 fixture 验收。
**真实模型调用、付费调用、生产访问均为 0；所有改动仅保留在工作树，未提交、未部署。**
真实 provider 的回复质量、实际单次耗时以及生产中三次调用的逐次细节没有被本轮验证。
