# S66: parenthesised readings, tone corrections, repeated mentions

S66 已在本地完成。T1、T2 精确回放均走确定性审词，模型调用为 0；同一会话 T1 → T2 →「加入并提交」也已验证，假工具写入并提交两个词。最终六套离线检查、`e2e.test_safety`、S54/S62/S63/S64/S65/S66 全部通过。

开始时 `git log -1 --oneline`：

```text
788ea0c fix: every commonness verdict shows the evidence it was decided on
```

开始工作区干净；当前改动未暂存、未提交、未推送、未部署。

## 语法与路由

新语法由 `keytao_bot/utils/reading_request.py:20`、`:30` 完整匹配消息：

```text
[ADD] WORD（READING） SEP WORD(READING)
[ADD] WORD（READING：CODE） SEP WORD（READING）
ADD := 加词 | 添加词 | 新增词 | 添加
SEP := ， | 、 | ； | , | ; | 和 | whitespace
```

- 支持数字声调、标调字母和无声调拼音；括号与冒号经 NFKC 统一，容忍括号内两端空格，保留读音中的标调和数字。
- 仍须消费整条消息；否定、转述、额外操作、未闭合括号、重复词项不会由新语法取得候选票据。
- 裸词列表与加词列表共用现有逐词查询、审词、`select_candidate_inventory`、批量渲染和确认执行链。语法本身只创建可供下一轮选择的候选，不直接写入。
- 每词将原始标注作为 `requested_reading` 传给 `keytao_prepare_reviewed_add`；只去掉标注两端空白。最多 10 词，在工具调用前检查上限。
- 可选 `CODE` 必须在该读音的可信候选链中；不匹配时展示已核验读音链并要求澄清，不忽略用户编码、不生成新编码。

代码依据：`chat_routing.py:2745` 在命令意图模型前识别新形状；`chat_commands.py:3480`、`:3580` 将多词读音逐词带入现有管线；`:3747` 校验括号中的编码。

## 声调差异与音节冲突

**对根因的补充说明（有源码依据）：** `utils/pinyin_reference.py:76` 在本次修改前已经去掉声调数字和声调符号，`utils/keytao_review.py:602` 使用它比较音节。因此，事故里的声调阻断来自未命中确定性路由后的模型回答，并非审词工具原先强制声调相同。本次保留这条音节比较规则，补上解释和审计上下文。

`keytao_review.py:631` 只在音节序列相同后比较声调；`xiang2` 与 `xiáng` 是等价写法，不声称发生更正。声调不同或缺失时，继续使用审词结果中的读音和原候选链，输出一行更正说明，且不新增读音阻断。原有常用度、来源和人工审核规则照常适用。

fixture 给出的已审读音为 `yī lì xiáng shí huì`、`dà lì chū qí jì`；对应 T1 的实际输出包括：

```text
审词：读音 yī lì xiáng shí huì；来源 用户明确指定读音 + 编码服务候选组；
读音按标准 yī/xiáng/huì（你标注的 yi2、xiang3、hui 已按此更正）
```

T2 的第一词仍含 `yi2`，因此也有按 fixture 已审 `yī` 更正的说明。说明内容由审词读音生成，没有对事故词句写例外。上述读音和候选编码是离线审词载荷，用于验证数据流，不是本次实时查询到的语言学或生产词库事实。

审词结果保留 `requestedReading`、`readingCorrection`；批量候选 scope 保留 `requestedReading` 和完整审词提示，写入备注保留「用户标注读音 …」。候选能力及写入能力始终绑定已审读音。依据：`keytao_review.py:4251`、`chat_render.py:1542`、`chat_commands.py:3802`。

音节不同的初次括号标注仍先问一次，同时列出已逐字验证的默认读音链和用户读音链，不创建该词的加词票据。明确回复「一力降十会 读音 yi1 li4 jiang4 shi2 hui4」后，由既有显式读音处理器选择编码服务已返回的分组，不再问同一问题。若用户音节根本不在编码服务候选中，只展示真实返回且通过逐字核验的链，不伪造该未知读音的编码。一个词冲突时，其余已解决词仍得到批量候选和加入页脚。

依据：`chat_commands.py:3737`；`keytao_review.py:3761`、`:3849`。S66 覆盖两链展示、一次追问、后续显式选择、未知音节、不匹配编码和混合批次。

## 重复提及与连续回合

`authorization_grammar.py:320`、`:2275` 将正文任意位置的独立 `@喵喵`、`喵喵`、`@键道`、`键道` 自身提及/触发 token 去掉；路由和授权归一化共用这一逻辑。保留正文中的其他用户提及及「喵喵叫」等词内部字符串。T2 的中间提及及三次提及用例均已回放。

T1 已生成候选后，T2 是一次新的明确读音审词。`openai_chat.py:3876` 让新标注替换未执行的多词候选，避免把它作为旧票据的确认语句。此分支不会删除带执行标记的记录、服务端风险确认票据或存在活动操作的记录。

## 修复前后回放

| 回合 | 修复前（用户提供的生产日志） | 修复后（本地 fixture） |
| --- | --- | --- |
| T1 原文 | `general`；3 model calls；0 tools；询问要做什么 | `word-discovery`；0 model calls；双词审词、全部候选、加入/加入并提交页脚 |
| T2 原文 | `general`；4 model calls；2 tools | `word-discovery`；0 model calls；两词 `requested_reading`、已审批量票据、更正说明 |
| T3 加入并提交 | `pending-confirmation`；0 model calls；写入提交 | `pending-confirmation`；0 model calls；假工具写入并提交两词 |

修复前精确失败回放：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s66_readings

Ran 2 tests in 0.069s

FAILED (failures=3)
```

两个原文都触发被 mock 阻断的模型入口，重复提及断言失败。原始日志 `/tmp/keytao-s66/red.log`；这是尝试进入模型的离线证据，并没有发生真实模型请求。

最终 S66 15 个测试方法，另含 126 组语法矩阵子场景。生产阶段链、真实工具分发器/执行器、审词函数、候选封印、最终广告校验实际执行；编码、来源、占用、写入、提交的外部边界使用 fixture。路由与审词模型构造器均设为调用即失败。T3 检查了 actor、已审读音能力、版本及 warning/snapshot/audit digest，并断言只写入和提交一次。跨 actor 确认不产生写入。

机器可读回放：

- `/tmp/keytao-s66/t1-replay.json`
- `/tmp/keytao-s66/t2-replay.json`
- `/tmp/keytao-s66/t1-t2-t3-replay.json`

## 最终离线验证

以下 `Results` 为自定义套件的断言计数；`Ran ... tests` 为 unittest 方法数，不混称。

六套和 safety 的最终复核命令：

```text
.venv/bin/python e2e/offline_checks.py
```

七个独立进程全部 exit 0。各日志的准确结果尾部：

```text
test_state_machine.py
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================

test_memory_safety.py
----------------------------------------------------------------------
Ran 407 tests in 208.413s

OK

test_security_fixes.py
============================================================
Results: 268/268 passed, 0 failed
============================================================

test_review_gate.py
============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED

test_llm_policy.py
----------------------------------------------------------------------
Ran 11 tests in 0.091s

OK

test_word_discovery.py
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED

-m e2e.test_safety
----------------------------------------------------------------------
Ran 107 tests in 0.599s

OK
```

日志归档 `/tmp/keytao-s66/final-logs/`；七套退出码索引为该目录下的 `six-and-safety-results.json`，定向模块保留对应 `*-results.json`。负向 fixture 中的拒绝、error 日志属于预期行为，以最终断言结果与进程退出码为准。

S54 / S66（exit 0）：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s54_multiword test_s54_renderer test_s54_selection test_s66_readings

Ran 52 tests in 0.445s

OK
```

S62 全模块（exit 0）：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness

Ran 51 tests in 0.320s

OK
```

S63 其余路由、S64、S65（exit 0）：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply test_s64_existing_query test_s64_explicit_submit test_s65_commonness_render

Ran 54 tests in 0.980s

OK
```

S63 的 importer/shell fixture 使用其既有专用隔离入口（exit 0），没有放开普通 launcher 的子进程限制：

```text
env -i PATH="$PWD/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" LANG=en_US.UTF-8 TMPDIR=/tmp .venv/bin/python -m e2e.s63

test_s63_bcc
Ran 22 tests in 0.199s

OK

test_s63_bcc_ingest
Ran 10 tests in 1.130s

OK

test_s63_bcc_delivery
Ran 3 tests in 0.034s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

S66 独立执行（exit 0）：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s66_readings

Ran 15 tests in 0.342s

OK
```

一次尝试将全部定向模块合并调用，因 launcher 将模块名拼接为日志文件名而触发 `OSError: [Errno 63] File name too long`，测试尚未启动；随后拆成以上较短的模块组运行，全部通过。没有修改 launcher 或放宽隔离。

`git diff --check` 无输出、exit 0。最终 HEAD 保持 `788ea0c`。

## 验证边界

**ZERO paid model calls：付费模型调用 0、真实 provider 调用 0、生产密钥读取 0。** 未运行 `pnpm test`、真实模型 E2E、SSH、部署、线上 API 或生产数据操作。

常规检查由现有 `e2e/offline_checks.py` 清空继承凭据，阻断网络、真实凭据文件和未隔离子进程；未运行其凭据路径 self-test。S63 使用空环境和既有 fixture 入口，其临时 shell 替身不启动真实 bot、builder、ingest、Docker 或安装程序。

证据层次：静态源码/diff + fixture/fake-tool 定向执行 + 指定离线套件。生产事实仅来自本任务提供的事故记录；未进行生产验证。改动只在本地，未提交、未推送、未部署。
