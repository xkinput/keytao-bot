# S70：寒暄回复与交付边界

状态：本地修复及最终离线验收全部完成并通过；未提交、未推送、未部署。

## 范围与基线

- HEAD：`f1cffe8`；开始时工作区干净。
- 只修改 bot；不提交、不推送、不部署，不运行 `pnpm test`。
- 仅运行离线套件和 fixture/fake-model；不读取生产密钥，不调用真实模型或生产 API。

## 复现、路由条件与根因

### 修复前的红测

在修改产品代码前，以 fake client 输出事故原文，依次经过真实 `AgentOrchestrator.run` 和 `_prepare_user_facing_reply`。命令：

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s70_smalltalk.SmallTalkTests
```

当时准确结果：

```text
AssertionError: '用户' unexpectedly found in '用户发的这条消息只是「早」，是在打招呼，没有任何词库操作请求，也没有对应的可执行命令。\n不需要补充信息，等有具体需求再说即可。'

----------------------------------------------------------------------
Ran 1 test in 0.052s

FAILED (failures=1)
```

原始红测日志：`/tmp/keytao-s60-offline/m-unittest-test_s70_smalltalk.SmallTalkTests.log`。后续把该回归加深到真实阶段链及交付前再次注入旁白，没有用单纯字符串单测替代事故路径。

确定的诱因是旧 `READ_ONLY_TURN_GUIDANCE` 对所有没有写入授权的回合要求“说明你理解的当前请求”“若没有可执行命令，只说明还缺少哪项具体信息”。问候也会收到这些指令。旧交付检查只约束候选、命令和内部实现片段，没有拒绝普通中文的意图旁白。主提示词还有“中文短词默认查词”，lookup 技能的候选回复表又把所有“其他”消息导向选码提示。这些是源码和假模型复现证据；没有访问事故当时的生产模型轨迹。

### FIXED：按缺少操作意图路由，不按问候名单

`keytao_bot/plugins/chat_routing.py:490` 增加独立的可空 `has_operation_intent`，由现有语义路由 JSON 携带；解析在 `:2547`，语义要求在 `:2795`。它与快捷操作 `intent` 分开：`none` 只表示没有快捷处理项，功能问答、新的复杂操作仍可为 `has_operation_intent=true`。

`keytao_bot/plugins/openai_chat.py:3294` 只有在该字段明确为 `false`、快捷意图为 `none`、其他本轮分类没有正向操作、没有查询目标，且当前消息没有明确写入语法或操作语法缺口时，才进入纯聊天路径。缺失、非布尔值均保留 `None`，不会当作“没有操作”。真实查询、功能说明、草稿操作和信息不全的操作继续原流程。

该条件没有“早／谢谢／晚安”等词表。`natural_reply` 中的少量语意匹配只在已经判定没有操作意图后选择措辞，不能让任何消息取得聊天路由资格。未列举的“今天的云像棉花糖”走同一路径。正向操作语法还能否决错误的聊天元数据。

在 `openai_chat.py:5801`，已确认无操作意图的消息跳过裸词解析，避免把“晚安，明早见”按两个词查询；`:5971` 跳过后续查词追加。`harness/orchestrator.py:627`、`:814` 在此类回合不暴露工具，避免模型自行查词后把寒暄变成候选报告。

## 提示词与交付边界

### 提示词清单

| 位置 | 修复 |
|---|---|
| `keytao_bot/plugins/chat_prompt.py:67`、`:113` | 短消息先看查词意图；无操作意图以喵喵身份直接聊一句。禁止第三人称用户旁白、无命令报告、操作尾注。 |
| `keytao_bot/harness/orchestrator.py:109` | 删除泛化的“说明理解／没有命令就说明缺项”；仅对真实操作请求询问具体缺项。 |
| `keytao_bot/skills/keytao-lookup/SKILL.md:7`、`:382` | 查词展示和候选追问只用于对应请求；“其他”拆为选择不清和转而聊天，后者自然回应。 |
| `keytao_bot/skills/keytao-draft/SKILL.md:10` | 限定草稿模板、补充信息和尾注的适用范围，普通聊天不套用。 |
| `keytao_bot/harness/authorization_grammar.py:3531` | 工具拒绝的 `modelInstruction` 只对真实操作询问缺项，不旁述意图。 |
| `keytao_bot/harness/tools.py:2317` | 修复同一条动态提示词的第二处生成点。 |
| `keytao_bot/plugins/chat_routing.py:2795` | 分类只输出独立操作意图元数据；按完整消息语意判断，寒暄不能抵消同时提出的操作。 |

也检查了 review、docs、web-search、date-time 技能；其失败/缺项提示已有具体工具或查询条件，没有另一处泛化的“没有命令就解释缺项”要求，未作无关修改。

### FIXED：最终发送前整段重绘

`keytao_bot/utils/conversational_reply.py:10` 按用户旁白、消息分类、没有命令/操作、决定不调用工具、等待具体需求等类别检查；`:18` 同时拒绝命令、候选、草稿、证据、服务选项和操作尾注。命中后在 `:41` 整段重绘，保留一行自然回复；已经自然的聊天回复保留内容并合成一行。

回复在 `openai_chat.py:2576` 的共享交付边界再检查，不依赖模型遵守提示词。喵喵的自然回复来自小组措辞，避开历史中的上一条回复；`ConversationalReply` 标记保证同一次交付重复检查不会重新换词。不会为重绘发起模型调用。

QQ/Telegram 沿用当前阶段链的分类结果；Web/web-anon 原本绕过阶段链，现在由共享 `get_ai_response_core` 的 `:1614` 入口设置当前消息，并在没有传入分类结果时复用同一分类器。Web 因此新增一次语义分类步骤；群聊不额外分类。真实 `ServerBackedQueryReply` 在 `:1731` 保留其可信结果。Web fake fixture 调用真实 core 和最终交付函数，未启动 HTTP 服务。

### 早：修复前后

输入：`喵喵 早`。fake model 固定返回生产事故原文：

```text
用户发的这条消息只是「早」，是在打招呼，没有任何词库操作请求，也没有对应的可执行命令。
不需要补充信息，等有具体需求再说即可。
```

修复前该原文原样发出；修复后最终交付：

```text
早呀，喵～
```

同一历史中再次收到 `早`，同一个 fake model 输出被重绘为：

```text
早上好呀，今天也精神满满喵！
```

`/tmp/keytao-s70/final/replay.json` 保存完整回放。晚安、在吗、谢谢分别得到“晚安喵，做个好梦～”“在呢，喵～”“不客气喵～”；未列举的聊天原本自然时保留“软乎乎的，看得本喵都想咬一口～”。

### 回归与审查闭合

`test_s70_smalltalk.py` 共 15 项测试：指定寒暄及普通闲聊；事故原文及交付前注入的同类旁白；重复问候；真实草稿工具路径；只读编码问句；信息不全的操作；`none` 下的绑定/功能问答；明确操作否决错误元数据；晚安复合句；Web/web-anon；真实 JSON 分类解析；unknown/true 优先；已有无关候选保持不动；正常“用户体验”措辞。

独立只读审查指出的 `none` 误判、Web 漏口、用户旁白变体及晚安复合句均已修复并增加回归；增量复核未发现剩余阻断项。审查没有运行真实模型或 HTTP 服务。

首轮六套中的状态机曾为 `2019/2023`：3 项是提示词字面约定，保留原有词义/批量查询说明并加上意图限定后恢复；1 项暴露把快捷 `none` 当无操作意图的问题，改用独立可空字段后恢复。没有删改现有测试或放宽其断言。该首轮混合代码状态不算最终验收。

## 最终验收

最终验收全部通过；所有命令退出码 **0**。六套分别为 **2023 / 407 / 268 / 443 / 11 / 290** 项通过，`e2e.test_safety` **107/107**，S54/S58/S59/S62–S70 指定 fixture 合计 **345/345**（其中 S70 **15/15**）。

本节只采用源码冻结后的最终运行。完整日志和逐条命令在 `/tmp/keytao-s70/final/`、`results.json`。驱动：

```sh
.venv/bin/python /tmp/keytao-s70/run_acceptance.py
```

`source-before.json` 与 `source-after.json` 的源码、技能提示词和测试 SHA-256 完全一致；后续只补写报告。驱动最终准确输出：

```text
{"sourceUnchanged": true, "allPassed": true, "runnerExits": [0, 0, 0]}
```

### 六套离线套件与安全套件

实际入口为下列命令，由仓库 runner 在七个独立、空凭据且有网络/密钥读取 guard 的子进程执行：

```sh
.venv/bin/python e2e/offline_checks.py
```

`test_state_machine.py.log` 的准确尾部：

```text
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================
```

`test_memory_safety.py.log` 的准确尾部：

```text
----------------------------------------------------------------------
Ran 407 tests in 214.102s

OK
```

`test_security_fixes.py.log` 的准确尾部：

```text
============================================================
Results: 268/268 passed, 0 failed
============================================================
```

`test_review_gate.py.log` 的准确尾部：

```text
============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED
```

`test_llm_policy.py.log` 的准确尾部：

```text
----------------------------------------------------------------------
Ran 11 tests in 0.121s

OK
```

`test_word_discovery.py.log` 的准确尾部：

```text
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED
```

`m-e2e.test_safety.log` 的准确尾部：

```text
----------------------------------------------------------------------
Ran 107 tests in 0.642s

OK
```

### S54 / S58 / S59 / S62–S70

为避免 fixture 模块替身相互污染，下列每个模块均独立运行；命令模板是实际执行的形式，模块名逐项列在准确尾部之前：

```sh
.venv/bin/python e2e/offline_checks.py -m unittest MODULE
```

`test_s54_multiword`：

```text
----------------------------------------------------------------------
Ran 11 tests in 0.081s

OK
```

`test_s54_renderer`：

```text
----------------------------------------------------------------------
Ran 6 tests in 0.085s

OK
```

`test_s54_selection`：

```text
----------------------------------------------------------------------
Ran 20 tests in 0.034s

OK
```

`test_s58_advertisement`：

```text
----------------------------------------------------------------------
Ran 25 tests in 0.257s

OK
```

`test_s58_bridge`：

```text
----------------------------------------------------------------------
Ran 9 tests in 0.056s

OK
```

`test_s58_explicit_destination`：

```text
----------------------------------------------------------------------
Ran 5 tests in 0.015s

OK
```

`test_s58_generic_bridge`：

```text
----------------------------------------------------------------------
Ran 15 tests in 0.140s

OK
```

`test_s58_grammar`：

```text
----------------------------------------------------------------------
Ran 8 tests in 0.032s

OK
```

`test_s59_commonness_evidence`：

```text
----------------------------------------------------------------------
Ran 9 tests in 0.014s

OK
```

`test_s59_commonness_route`：

```text
----------------------------------------------------------------------
Ran 9 tests in 0.066s

OK
```

`test_s59_general_cap`：

```text
----------------------------------------------------------------------
Ran 11 tests in 0.102s

OK
```

`test_s59_tool_prompt`：

```text
----------------------------------------------------------------------
Ran 6 tests in 0.068s

OK
```

`test_s62_adversarial`：

```text
----------------------------------------------------------------------
Ran 6 tests in 0.063s

OK
```

`test_s62_commonness`：

```text
----------------------------------------------------------------------
Ran 4 tests in 0.001s

OK
```

`test_s62_draft_flow`：

```text
----------------------------------------------------------------------
Ran 14 tests in 0.097s

OK
```

`test_s62_incident`：

```text
----------------------------------------------------------------------
Ran 8 tests in 0.194s

OK
```

`test_s62_reading_resolution`：

```text
----------------------------------------------------------------------
Ran 7 tests in 0.069s

OK
```

`test_s62_selection`：

```text
----------------------------------------------------------------------
Ran 8 tests in 0.036s

OK
```

`test_s62_stale_draft`：

```text
----------------------------------------------------------------------
Ran 4 tests in 0.080s

OK
```

`test_s63_fresh_selection`：

```text
----------------------------------------------------------------------
Ran 16 tests in 0.130s

OK
```

`test_s63_general_reply`：

```text
----------------------------------------------------------------------
Ran 9 tests in 0.035s

OK
```

`test_s64_existing_query`：

```text
----------------------------------------------------------------------
Ran 3 tests in 0.160s

OK
```

`test_s64_explicit_submit`：

```text
----------------------------------------------------------------------
Ran 16 tests in 0.142s

OK
```

`test_s65_commonness_render`：

```text
----------------------------------------------------------------------
Ran 10 tests in 0.638s

OK
```

`test_s66_readings`：

```text
----------------------------------------------------------------------
Ran 15 tests in 0.358s

OK
```

`test_s67_ranking`：

```text
----------------------------------------------------------------------
Ran 13 tests in 0.056s

OK
```

`test_s68_selection`：

```text
----------------------------------------------------------------------
Ran 13 tests in 0.364s

OK
```

`test_s69_fly_ticket`：

```text
----------------------------------------------------------------------
Ran 15 tests in 0.121s

OK
```

`test_s70_smalltalk`：

```text
----------------------------------------------------------------------
Ran 15 tests in 0.156s

OK
```

### S63 BCC 独立入口

同样由驱动创建空环境，仅传入 PATH/LANG/TMPDIR/PYTHONUNBUFFERED；复现命令：

```sh
env -i PATH=/Users/rea/code/keytao-org/keytao-bot/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 TMPDIR=/tmp PYTHONUNBUFFERED=1 /Users/rea/code/keytao-org/keytao-bot/.venv/bin/python -m e2e.s63
```

三个子进程的准确结果尾部和最终摘要：

```text
----------------------------------------------------------------------
Ran 22 tests in 0.181s

OK

----------------------------------------------------------------------
Ran 10 tests in 1.011s

OK

----------------------------------------------------------------------
Ran 3 tests in 0.032s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

### 静态检查与回放

`git diff --check` 退出码 0，无输出。冻结后的 fake 回放由 `/tmp/keytao-s70/replay.py` 在空环境、`-I` 和仓库 I/O guard 下执行，退出码 0；产物为 `/tmp/keytao-s70/final/replay.json`，寒暄工具调用列表为 `[]`。


## 预算与交付状态

付费模型调用 **0**，真实 provider 调用 **0**，生产密钥读取 **0**。不运行 `pnpm test`，不调用生产 API 或主机。全部 Python 测试在空凭据环境运行，离线 runner 的 I/O guard 禁止网络和真实密钥文件读取；fake key 是测试内字面量。

HEAD 保持 `f1cffe8`；仅本地未提交修改，未 stage、commit、push、deploy，未生产验证。未验证真实模型的分类准确率、QQ/Telegram 实际发送或 HTTP 服务；本报告只主张源码、离线套件和 fixture/fake-model 证据。
