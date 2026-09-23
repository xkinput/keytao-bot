# S70b：恢复自然问候，查询语法优先

本地实现与离线验收已完成：模型的“无操作意图”不再足以把裸词查询变成闲聊。自然回复、重复问候换措辞、无尾注、无模型调用的交付重画，以及真正聊天时不暴露工具，均已恢复。六套离线检查、safety 和全部指定场景回归通过，结果见末尾独立验收节。

## 范围与初始状态

- 仓库：`/Users/rea/code/keytao-org/keytao-bot`。
- `git log -2 --oneline`：`3fb4865 revert: S70 natural-chat routing swallowed dictionary queries`，上一条为 `95cacc2 fix: answer a greeting instead of reporting that it found no command`。
- 初始工作区干净。通过补丁恢复 S70 产品代码和假模型测试，没有执行会创建提交的 revert，也没有恢复旧报告的验收结论。
- 本轮属于本地逻辑修复。无外部写入授权，没有 commit、push、deploy 或生产访问。

## 根因与复现

先恢复 S70，再新增真实 stage 链上的假模型测试。命令：

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s70b_query_precedence
```

最初输出为 `Ran 2 tests in 0.103s` / `FAILED (failures=5)`。失败子场景为 `练枪 得吃`、`@喵喵 练枪 得吃`、`蛋粉 单份`、`敲不死`、`一力降十会（yi2 li4 xiang2 shi2 hui4）`。所有失败都是查询词列表为空，例如 `[] != ['练枪', '得吃']`；聊天矩阵通过。原始日志：`/tmp/keytao-s70b/red.log`。

根因有两层：S70 在裸词 parser 之前仅凭 `has_operation_intent=False` 设置聊天标记，使查询主路径和补充查询都跳过；即使解除这层阻断，旧单词入口仍需要第二个词查询分类器点头。因此同时修复聊天认领条件和单词语法的确定性放行，不依赖调整分类提示词来保证正确性。

## 最终优先级与判别器

1. 已有明确操作、候选选择、草稿流程等保持原有处理顺序和授权检查。
2. 对未被处理的消息，只有**整条消息**属于明确聊天形式，才允许设置聊天标记。分类器还须返回 `intent=none`、`has_operation_intent=False`；缓存中不能有相反任务判定，不能有已解析的查询目标，也不能匹配写入或未闭合操作语法。单独一个“无意图”判定永远不够。
3. 其余单词语法直接返回查询目标；S54 多词列表和 S66 括号读音继续进入既有逐词查询/审词/候选持久化路径。模型的否定判定不能阻断这些确定性匹配。
4. 问候与词混合时按查询处理，例如 `谢谢 练枪` 查两个词。完整操作、功能问句、比较句、转述句不被聊天认领，仍由各自原有流程处理。无法确认为聊天的词形输入优先查询。

`keytao_bot/utils/conversational_reply.py:22` 的判别器逐个完整分句匹配，分句间可用空白或中文/英文标点：

- 小型会话行为词表：问候、感谢、告别、应答、在吗类呼叫，以及连续笑声；可有少量语气词。例如 `早`、`晚安`、`谢谢`、`辛苦了`、`收到`、`明早见`、`哈哈哈`。只有全部分句匹配才认领，包含某个问候词并不够。
- 保守的完整句式：如“今天/昨天/明天/今晚的……像/看起来……”和“我[今天/现在]觉得/感觉/有点……”，且不含查询、词条、编码、草稿、修改、绑定、功能问句等任务标记。因此 `今天的云像棉花糖` 可以聊天，`我感觉编码有问题` 不行。这不是任意中文句子的语义分类器。
- 空白、`，`、`、`、`；` 直接分词；独立 `和` 可作连接词。无空白 `和` 仅在两侧均至少两个字时拆分；不明确时按整个单词查询，以免把 `共和国` 拆坏。已有明确空白边界时保留词内的 `和`，如 `和平 共处`。括号读音由既有 S66 parser 负责。

词表为何允许：它只为“不执行工具”的聊天分支提供正向证据；不会授予 create、submit、delete、候选绑定或确认能力。写入授权的 parser、actor、记录、摘要及确认规则没有变化。`test_s70b_query_precedence.py:177` 对这些聊天消息同时断言 `message_authorizes_mutation=False`。

## 产品修改与保留项

| 位置 | 修改或保留行为 |
|---|---|
| `keytao_bot/plugins/openai_chat.py:3294` | 聊天认领必须额外满足整条聊天形式；主查询、补充查询、Web core 和交付使用同一认领状态 |
| `keytao_bot/plugins/chat_routing.py:2914` / `:2928` | 共享裸词边界，排除操作片段、转述与比较句；单词正向语法先于模型回退 |
| `keytao_bot/plugins/chat_commands.py:3452` | 多词列表沿用逐词查询管线，支持所需分隔符，完整寒暄才退出 |
| `keytao_bot/harness/orchestrator.py:109` / `:815` | 修复 READ_ONLY_TURN_GUIDANCE；聊天请求不暴露 tools |
| `keytao_bot/plugins/chat_prompt.py:67` / `:114` | 查询默认优先，自然聊天直接面对用户，一行、无尾注 |
| `keytao_bot/skills/keytao-lookup/SKILL.md:7` | 查询语法优先；完整聊天不套查询或候选模板 |
| `keytao_bot/skills/keytao-draft/SKILL.md:10` | 仅操作目标不完整时询问缺项，聊天不追加草稿尾注 |
| `keytao_bot/harness/tools.py:2316` / `keytao_bot/harness/authorization_grammar.py:3531` | 恢复生成点的自然回复指导；只修改指导文案，没有修改写授权 |
| `keytao_bot/utils/conversational_reply.py` | 恢复第三人称旁白、“无可执行命令”、操作尾注的整句重画，重复问候换措辞，无额外模型调用 |

## 矩阵与额外反例

所有行都使用真实 stage 链、真实 core/orchestrator、假传输；command classifier 对**所有输入**返回 `has_operation_intent=False`，word classifier 也始终拒绝。每次验证 command classifier 确实被调用。查询断言精确目标、最终正文与零生成调用；聊天断言一行自然正文、零工具执行、生成请求中无 tools。

| 输入 | 实际路由 / 查询目标 | 结果 |
|---|---|---|
| `练枪 得吃` | query：练枪、得吃 | PASS |
| `蛋粉 单份` | query：蛋粉、单份 | PASS |
| `敲不死` | query：敲不死 | PASS |
| `一力降十会（yi2 li4 xiang2 shi2 hui4）` | query：一力降十会 | PASS |
| `早` | chat：早呀，喵～ | PASS |
| `晚安` | chat：晚安喵，做个好梦～ | PASS |
| `谢谢` | chat：不客气喵～ | PASS |
| `在吗` | chat：在呢，喵～ | PASS |
| `辛苦了` | chat：不客气喵～ | PASS |
| `晚安，明早见` | chat：晚安喵，做个好梦～ | PASS |
| `今天的云像棉花糖` | chat：喵，我在听呢。 | PASS |
| `哈哈哈` | chat：哈哈，本喵也乐了～ | PASS |

上表聊天回复来自合成旁白输入触发的交付重画，不是模型生成质量评测。原有 S70 测试另验证自然且相关的“软乎乎的，看得本喵都想咬一口～”原样保留。逐行原始观察：`/tmp/keytao-s70b/matrix.json`；其中 `abcd` 仅为测试编码，不代表真实词库事实。

额外通过：带 `@喵喵` 的原始事故消息；九种分隔形式；`谢谢 练枪` 等混合输入；`共和国` 与 `和平 共处` 不拆坏；词库未收录时逐词审词并保存两词候选；`加词 加亮` 仍进入审词并建立候选记录；`查词 谢谢` 明确查询；Web/Web-anon 查询仍有工具可用且不被聊天重画；不完整操作仍询问目标；旧候选不劫持感谢；重复问候换措辞；合成旁白在最终交付再次注入仍被无模型调用重画。

## 中间检查与修正

- 首轮状态机发现共享 facade 导入、提示词兼容文案、`他说加入` 转述，以及 `这个和电机哪个常用` 比较句边界。修复后原有断言全部通过，未放宽断言。首轮完整日志：`/tmp/keytao-s70b/first-state-machine.log`。
- S65 静态检查把新路由正则中的冗余“更常用”字面量识别成硬编码常用度结论；删除该冗余分支，保留“哪个”等比较句结构，原检查通过。日志：`/tmp/keytao-s70b/s65-regex-guard.log`。
- 为定位内存安全套件的长执行曾开启 30 秒 traceback 诊断；栈落在既有授权交叉矩阵，常规套件随后完整通过。诊断日志也打印了 407 项 OK，但诊断会话未自行退出，最终对该会话发送 Ctrl-C 收尾（exit 130）。诊断输出不作为通过证据，也不把它算作常规套件失败。

## 零付费调用与验证边界

**付费模型调用 0；真实 provider 调用 0；生产密钥读取 0。** 常规检查通过既有 `e2e/offline_checks.py`：白名单子进程环境、禁网、禁止真实凭据文件、dotenv 空返回、禁止未隔离子进程。未运行其会探测真实凭据路径的 self-test。S63 使用空环境和已有 fixture-only 入口；其 shell fixture 仅调用临时 `uv` 替身，未启动真实 bot、Docker、下载或安装。独立矩阵重放也先安装同一 offline guard。

没有运行 `pnpm test`、付费 E2E runner、真实模型、生产接口、commit、push 或 deploy。这里是离线逻辑与交付验证，不是生产验证。

## Final acceptance / 最终验收

全部要求通过：六套离线检查、`e2e.test_safety`、当前全部 S54/S58/S59/S62–S69 模块，以及 S70/S70b。额外运行字面短语路由回归。各命令 exit 0；`Results` 是脚本断言数，`Ran ... tests` 是 unittest 方法数，矩阵 subTest 不混入方法计数。

### 六套离线检查与 safety

完整启动命令：

```sh
.venv/bin/python e2e/offline_checks.py
```

该入口逐一执行下面七项，最终启动器 exit 0。最后的词形/分隔符收窄之后，另完整重跑 `test_state_machine.py` 和全部指定场景模块，均通过。没有以启动成功或旧报告代替测试结果。

`.venv/bin/python test_state_machine.py`

```text
Results: 2023/2023 passed, 0 failed
```

`.venv/bin/python test_memory_safety.py`

```text
Ran 407 tests in 217.035s
OK
```

`.venv/bin/python test_security_fixes.py`

```text
Results: 268/268 passed, 0 failed
```

`.venv/bin/python test_review_gate.py`

```text
Results: 443/443 passed
```

`.venv/bin/python test_llm_policy.py`

```text
Ran 11 tests in 0.107s
OK
```

`.venv/bin/python test_word_discovery.py`

```text
Results: 290/290 passed
```

`.venv/bin/python -m e2e.test_safety`

```text
Ran 107 tests in 0.719s
OK
```

七项原始日志、退出码与时长：`/tmp/keytao-s70b/final-suites/` 及其中的 `results.json`。

### 指定场景及自然聊天回归

每个普通模块在独立进程执行，避免不同假模型 harness 的模块替身互相污染：

```sh
.venv/bin/python e2e/offline_checks.py -m unittest MODULE
```

下表给出每个 MODULE 的原始尾部结果。S63 的三个 BCC 模块使用既有专用入口：

```sh
env -i PATH=/Users/rea/code/keytao-org/keytao-bot/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 TMPDIR=/tmp .venv/bin/python -m e2e.s63
```

| MODULE / fixture entry | Tail（均 exit 0） |
|---|---|
| `test_s54_multiword` | `Ran 11 tests in 0.094s / OK` |
| `test_s54_renderer` | `Ran 6 tests in 0.096s / OK` |
| `test_s54_selection` | `Ran 20 tests in 0.038s / OK` |
| `test_s58_advertisement` | `Ran 25 tests in 0.283s / OK` |
| `test_s58_bridge` | `Ran 9 tests in 0.065s / OK` |
| `test_s58_explicit_destination` | `Ran 5 tests in 0.017s / OK` |
| `test_s58_generic_bridge` | `Ran 15 tests in 0.157s / OK` |
| `test_s58_grammar` | `Ran 8 tests in 0.042s / OK` |
| `test_s59_commonness_evidence` | `Ran 9 tests in 0.015s / OK` |
| `test_s59_commonness_route` | `Ran 9 tests in 0.063s / OK` |
| `test_s59_general_cap` | `Ran 11 tests in 0.121s / OK` |
| `test_s59_tool_prompt` | `Ran 6 tests in 0.082s / OK` |
| `test_s62_adversarial` | `Ran 6 tests in 0.074s / OK` |
| `test_s62_commonness` | `Ran 4 tests in 0.001s / OK` |
| `test_s62_draft_flow` | `Ran 14 tests in 0.114s / OK` |
| `test_s62_incident` | `Ran 8 tests in 0.215s / OK` |
| `test_s62_reading_resolution` | `Ran 7 tests in 0.079s / OK` |
| `test_s62_selection` | `Ran 8 tests in 0.040s / OK` |
| `test_s62_stale_draft` | `Ran 4 tests in 0.092s / OK` |
| `test_s63_fresh_selection` | `Ran 16 tests in 0.153s / OK` |
| `test_s63_general_reply` | `Ran 9 tests in 0.041s / OK` |
| `test_s64_existing_query` | `Ran 3 tests in 0.183s / OK` |
| `test_s64_explicit_submit` | `Ran 16 tests in 0.159s / OK` |
| `test_s65_commonness_render` | `Ran 10 tests in 0.763s / OK` |
| `test_s66_readings` | `Ran 15 tests in 0.406s / OK` |
| `test_s67_ranking` | `Ran 13 tests in 0.066s / OK` |
| `test_s68_selection` | `Ran 13 tests in 0.408s / OK` |
| `test_s69_fly_ticket` | `Ran 15 tests in 0.136s / OK` |
| `test_s70_smalltalk` | `Ran 15 tests in 0.173s / OK` |
| `test_s70b_query_precedence` | `Ran 8 tests in 0.275s / OK` |
| `test_sep20_phrase_routing` | `Ran 4 tests in 0.217s / OK` |
| `e2e.s63` | `Ran 22 tests in 0.197s / OK / Ran 10 tests in 1.176s / OK / Ran 3 tests in 0.036s / OK` |

以上合计 **357 个 unittest 方法**，其中原 S70 为 15 项、新 S70b 为 8 项。S63 入口内三模块分别为 22、10、3 项，最终摘要为 `passed: true`、`paidModelCalls: 0`、`realProviderCalls: 0`。

完整命令索引与日志：`/tmp/keytao-s70b/regressions/results.json`、该目录的逐模块日志。执行脚本：`/tmp/keytao-s70b/run_regressions.py`。最终矩阵独立重放：`/tmp/keytao-s70b/replay_matrix.py` → `matrix.json`。

### 静态与交付状态

`git diff --check` 通过；12 个产品/测试文件的 SHA-256 与 `/tmp/keytao-s70b/source-freeze.json` 一致。旧测试断言未修改或削弱。HEAD 保持 `3fb4865`，索引为空，改动仅在工作区，**未提交、未推送、未部署，未做生产验证**。付费调用与生产密钥读取均为零。
