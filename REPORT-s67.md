# S67：词频表来源、行语义与有效调序广告

S67 的 A–D 已完成本地修复。起始 `git log -1 --oneline`：

```text
5fa88d0 fix: parenthesised readings route deterministically; tone-only corrections are a note
```

分支 `main`，起始工作区干净。未暂存、未提交、未推送、未部署。指定测试的最终结果见文末；线上改版效果没有验证。

## 缺陷与选择

| 项目 | 结论 | 修复与源码证据 |
| --- | --- | --- |
| A | FIXED | `keytao_bot/utils/commonness_query.py:87` 明示「BCC 现代四频道未收录」；回退值按 `corpusSource` 加来源，例如「jieba 参考 10」。未安装/不可用不冒充未收录。 |
| B | FIXED | `commonness_query.py:150` 使用该对词真实的 `front_more_common / behind_more_common`，不把显示行号当比较结论；`:165` 仅在目标同码同类型位置唯一且胜者当前权重大于败者时广告。 |
| C | FIXED | `commonness_query.py:76` 将「判定」改为「数据收录」；`:98` 统一回答「该词有什么可用来源」，不再汇总其与其他词的关系。 |
| D | FIXED | `keytao_bot/utils/keytao_review.py:5803` 保留双方实际采用的最强频道，最多两个频道；`:5818` 共用表格 formatter；`:5852` 的比例仍以未舍入信号计算。 |

选择「数据收录」的理由：表格一行代表一个词；多词查询的一行可能同时参与可判定、接近、未知的不同关系，不能用其中一种覆盖整行。现代 BCC 有收录时显示「现代 BCC 收录」；没有现代记录时列出 jieba、词典、历史 BCC 来源；只有历史时明确「仅历史 BCC 收录」；所有来源均无记录或不可用时「无数据」。比较对象的接近/未知及胜负仍保留于两词比较和排序说明，不改变底层工具的 `verdict` 字段。

四类行的 fixture 均覆盖：双方现代有记录、单边现代有记录、双方现代未收录但有历史/参考、全部无记录；另覆盖参考库不可用。未来旧参考来源通过 `corpusSource` 展示，已有无该字段的旧契约按 jieba 解释；测试包含显式的 `legacy-fixture` 来源。

精度规则沿用 `bcc_reference.py:163` 的 `format_frequency`：每百万频次通常保留至两位小数并去除尾零；小于 0.01 使用两位有效数字，避免真实小值变成零。因此 384.1023 在表格与摘要中均为 384.1，0.0944 均为 0.09，比例仍为真实比值。历史频次也复用同一格式。跨字/词的比较保留排名单位，平局转多领域时保留实际破平频道。

`_compare_reference_commonness` 与 `lookup_word_commonness` 的 AST 已逐一与 `5fa88d0` 比较，完全相同；没有更改比较阈值、历史破平规则、排名算法或工具 schema。产品技能 `keytao_bot/skills/keytao-lookup/SKILL.md:231` 已同步上述输出契约。

## 实时链核验与空操作检查

在修改产品代码前完成核验。时间：2026-09-22T14:28:08.636769+08:00。公开只读接口：

[GET /api/phrases/by-code?code=yeoiav&page=1](https://keytao.vercel.app/api/phrases/by-code?code=yeoiav&page=1)

```json
{"phrases":[{"word":"咽","code":"yeoiav","type":"Single","weight":10},{"word":"嘢","code":"yeoiav","type":"Single","weight":11}],"pagination":{"page":1,"pageSize":6,"total":2,"totalPages":1}}
```

调用前只读检查了相邻仓库 `keytao-next/app/api/phrases/by-code/route.ts:26`：只对 Finish 词条做数据库 SELECT/count，`:33` 明确 `weight: 'asc'`，没有浏览器、worker、stream 等持久资源分配。接口返回完整一页、总数 2，咽已在嘢前面，原广告确为空操作。

Web 阅读工具无法访问该 JSON URL，随后使用一次成功的 `env -i ... curl --disable --noproxy '*'` 无凭据 GET；没有读取环境密钥、cookie、认证文件或生产 DB。原始响应保留在 `/tmp/keytao-s67/live-yeoiav.json`。这是用户明确要求的实时链核验；所有测试依旧离线。

修复复用当前回合已校验的 `keytao_lookup_by_words_batch` 权重，不增加新的读链接口调用。已一致时选择**省略广告**。逆序 11/10 才显示同一条可执行命令。相等权重、缺失权重、重复位置、类型歧义、关系接近/未知或整组方向冲突时不广告。已有词在较短候选前缀上时也不广告移动；相反方向与缺词的有效放置仍保留。入口依旧经过原有命令 parser 与本回合交付封印（`commonness_query.py:254`），后续用户真正执行时仍走原有服务端核验。

核验只证明查询当时的实时 Finish 链。没有读取 Rea 的未提交草稿，也没有声称此快照保证未来执行时顺序不变。

## 数据来源和两条生产消息的前后回放

新增 `e2e/fixtures/bcc/s67.json` 来自本机 `data/pinyin_reference.db` 的只读七词切片，包含咽、嘢、强、鎗、之乎、敲不死、曰。按单字/词分别取对应 BCC 表，保留原计数、ppm、rank 和 jieba 记录；12 个 dataset 的 total_count / row_count 与既有 S63 fixture 逐项校验相等。没有写入本地原始数据库或读取生产数据库。

下列 BEFORE 在产品代码修改前通过原 `5fa88d0` 函数真实生成；AFTER 经过同一 fixture、真实查询 stage 和最终用户交付检查。编码/查词边界使用 fake tools；已核验的咽 10、嘢 11 是事故链 fixture。其余编码返回空候选，仅用于隔离这条广告。不是改版后的生产运行记录。

完整机器可读回放：`/tmp/keytao-s67/before.json`、`/tmp/keytao-s67/after.json`。

### 喵喵 词频排序：咽 嘢 强 鎗

BEFORE：

```text
常用度排序（本地语料与词典）
名次 | 词 | 语料频次 | 词典收录 | 判定 | 历史补充
1 | 强 | BCC 多领域 628,669（每百万 975.64）；新闻 2,849,945（每百万 1705.12）；文学 61,523（每百万 384.1）；口语 86,326（每百万 411.44） | 1 | 有数据 | 古代汉语 412,893（每百万 288.23）；近代汉语 496,682（每百万 346.34）
2 | 咽 | BCC 多领域 19,037（每百万 29.54）；新闻 9,439（每百万 5.65）；文学 8,431（每百万 52.64）；口语 2,199（每百万 10.48） | 1 | 有数据 | 古代汉语 95,231（每百万 66.48）；近代汉语 31,094（每百万 21.68）
3 | 嘢 | BCC 多领域 8（每百万 0.01）；新闻 57（每百万 0.03） | 1 | 无法判断 | 古代汉语 16（每百万 0.01）；近代汉语 未收录
4 | 鎗 | 10 | 0 | 无法判断 | 古代汉语 14,578（每百万 10.18）；近代汉语 4,791（每百万 3.34）
按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。词频是语料内计数。
BCC 取四个现代频道的最高每百万频次；字与词按各自表内排名比较。未收录不代表实际零次，单边收录不决定高低。
古代汉语、近代汉语不计入现代频次；仅在双方现代四频道均未收录且词典与 jieba 无明确方向时破平。
可发送以下命令生成调整计划：
- 「把 咽 yeoiav 排到 嘢 yeoiav 前面」
```

AFTER：

```text
常用度排序（本地语料与词典）
名次 | 词 | 语料频次 | 词典收录 | 数据收录 | 历史补充
1 | 强 | BCC 多领域 628,669（每百万 975.64）；新闻 2,849,945（每百万 1705.12）；文学 61,523（每百万 384.1）；口语 86,326（每百万 411.44） | 1 | 现代 BCC 收录 | 古代汉语 412,893（每百万 288.23）；近代汉语 496,682（每百万 346.34）
2 | 咽 | BCC 多领域 19,037（每百万 29.54）；新闻 9,439（每百万 5.65）；文学 8,431（每百万 52.64）；口语 2,199（每百万 10.48） | 1 | 现代 BCC 收录 | 古代汉语 95,231（每百万 66.48）；近代汉语 31,094（每百万 21.68）
3 | 嘢 | BCC 多领域 8（每百万 0.01）；新闻 57（每百万 0.03） | 1 | 现代 BCC 收录 | 古代汉语 16（每百万 0.01）；近代汉语 未收录
4 | 鎗 | BCC 现代四频道未收录；jieba 参考 10 | 0 | jieba、历史 BCC 收录 | 古代汉语 14,578（每百万 10.18）；近代汉语 4,791（每百万 3.34）
按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。词频是语料内计数。
数据收录仅说明该词的可用来源，不代表两词能否判定高低。
BCC 取四个现代频道的最高每百万频次；字与词按各自表内排名比较。未收录不代表实际零次，单边收录不决定高低。
古代汉语、近代汉语不计入现代频次；仅在双方现代四频道均未收录且词典与 jieba 无明确方向时破平。
```

### 喵喵 词频排序：之乎 鎗 敲不死

BEFORE：

```text
常用度排序（本地语料与词典）
名次 | 词 | 语料频次 | 词典收录 | 判定 | 历史补充
1 | 之乎 | — | 0 | 有数据 | 古代汉语 未收录；近代汉语 1,311（每百万 1.57）
2 | 鎗 | 10 | 0 | 有数据 | 古代汉语 14,578（每百万 10.18）；近代汉语 4,791（每百万 3.34）
— | 敲不死 | — | — | 无数据 | 古代汉语 未收录；近代汉语 未收录
按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。词频是语料内计数。
BCC 取四个现代频道的最高每百万频次；字与词按各自表内排名比较。未收录不代表实际零次，单边收录不决定高低。
古代汉语、近代汉语不计入现代频次；仅在双方现代四频道均未收录且词典与 jieba 无明确方向时破平。
```

AFTER：

```text
常用度排序（本地语料与词典）
名次 | 词 | 语料频次 | 词典收录 | 数据收录 | 历史补充
1 | 之乎 | BCC 现代四频道未收录 | 0 | 仅历史 BCC 收录 | 古代汉语 未收录；近代汉语 1,311（每百万 1.57）
2 | 鎗 | BCC 现代四频道未收录；jieba 参考 10 | 0 | jieba、历史 BCC 收录 | 古代汉语 14,578（每百万 10.18）；近代汉语 4,791（每百万 3.34）
— | 敲不死 | BCC 现代四频道未收录 | — | 无数据 | 古代汉语 未收录；近代汉语 未收录
按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。词频是语料内计数。
数据收录仅说明该词的可用来源，不代表两词能否判定高低。
BCC 取四个现代频道的最高每百万频次；字与词按各自表内排名比较。未收录不代表实际零次，单边收录不决定高低。
古代汉语、近代汉语不计入现代频次；仅在双方现代四频道均未收录且词典与 jieba 无明确方向时破平。
```

### 曰 / 强比较摘要

BEFORE：

```text
「强」较「曰」更常用：「强」BCC 多领域 628,669（每百万 975.6388）；新闻 2,849,945（每百万 1705.12）；文学 61,523（每百万 384.1023）；口语 86,326（每百万 411.4387）；「曰」BCC 多领域 10,180（每百万 15.7985）；新闻 11,382（每百万 6.8098）；文学 8,511（每百万 53.1361）；所用频次比 32.09×
```

AFTER：

```text
「强」较「曰」更常用：「强」BCC 新闻 2,849,945（每百万 1705.12）；文学 61,523（每百万 384.1）；「曰」BCC 新闻 11,382（每百万 6.81）；文学 8,511（每百万 53.14）；所用频次比 32.09×
```

保留新闻（强的最高 ppm）和文学（曰的最高 ppm），相同频道名最多两个，表格仍保留所有观测频道。

逆序 fixture 的广告尾部：

```text
可发送以下命令生成调整计划：
- 「把 咽 yeoiav 排到 嘢 yeoiav 前面」
```

**数值澄清（不反驳 C 的缺陷）**：本地同源完整切片中，咽/嘢按现有「各自最高现代 ppm」规则得到的是 `52.636681967369505 / 0.03410305789735371 = 1543.46×`，不是 2954×。两者仍明确可判定，C 的结论成立；没有以已经舍入的表格数字计算倍数，也没有为了复现 2954 而改算法。

## 离线复现及验收

修复前命令：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s67_ranking

Ran 9 tests in 0.053s

FAILED (failures=12)
```

launcher 的该次子进程 exit 1，原日志 `/tmp/keytao-s67/red.log`。失败直接覆盖裸 10、已经在前仍广告、未知/接近关系仍广告、旧混合列名、之乎空频次，以及摘要列出四频道。最终 S67 扩为 13 个测试方法，补上相同权重、重复/缺字段、当前较短编码、反向移动、极小频次、多领域破平和参考库不可用。

S59 原来的裸数字预期与 S65 原来的四位 ppm 预期按照新契约更新；保留了其路由、交付、真实比例、来源证据和授权断言。

### 最终六套及 safety

```text
.venv/bin/python e2e/offline_checks.py
```

七个独立子进程全部 exit 0。下列为各日志的准确尾部；自定义套件的 Results 是断言数，unittest 的 Ran 是测试方法数。

`.venv/bin/python test_state_machine.py`（exit 0）：

```text
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================
```

`.venv/bin/python test_memory_safety.py`（exit 0）：

```text
Ran 407 tests in 208.152s

OK
```

`.venv/bin/python test_security_fixes.py`（exit 0）：

```text
============================================================
Results: 268/268 passed, 0 failed
============================================================
```

`.venv/bin/python test_review_gate.py`（exit 0）：

```text
============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED
```

`.venv/bin/python test_llm_policy.py`（exit 0）：

```text
Ran 11 tests in 0.104s

OK
```

`.venv/bin/python test_word_discovery.py`（exit 0）：

```text
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED
```

`.venv/bin/python -m e2e.test_safety`（exit 0）：

```text
Ran 107 tests in 0.615s

OK
```

### 定向模块

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s54_multiword test_s54_renderer test_s54_selection test_s66_readings

Ran 52 tests in 0.439s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_commonness_evidence test_s59_commonness_route

Ran 18 tests in 0.072s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_tool_prompt

Ran 6 tests in 0.079s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_general_cap

Ran 11 tests in 0.113s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness

Ran 51 tests in 0.319s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply test_s64_existing_query test_s64_explicit_submit test_s65_commonness_render test_s67_ranking

Ran 67 tests in 0.993s

OK
```

exit 0。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s67_ranking

Ran 13 tests in 0.059s

OK
```

exit 0。

### S63 数据导入及 shell fixture 的专用隔离入口

```text
env -i PATH="$PWD/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" LANG=en_US.UTF-8 TMPDIR=/tmp .venv/bin/python -m e2e.s63
```

exit 0。三个子模块的准确尾部：

```text
test_s63_bcc
Ran 22 tests in 0.182s

OK

test_s63_bcc_ingest
Ran 10 tests in 1.151s

OK

test_s63_bcc_delivery
Ran 3 tests in 0.035s

OK

{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

日志归档：`/tmp/keytao-s67/final-logs/`。其中 `six-and-safety-results.json` 保留七套退出码，各组 `*-results.json` 保留命令与退出码，`summary.json` 记录准确尾部。所有日志都是本次最终代码的执行结果。

一次把全部 S59 模块放在同一 Python 进程运行，装载阶段触发 `AttributeError: module 'nonebot' has no attribute 'init'`，尚未执行测试：route 测试引入的 `test_state_machine` 使用 nonebot 替身，tool_prompt / general_cap 需要真实 nonebot 初始化。依据现有 `e2e/s59.py` 的分进程约定，随后将两类模块分开运行，全部通过；没有修改其初始化、禁网守卫或断言来绕过失败。原失败日志：`/tmp/keytao-s67/s59-combined-import-failure.log`。

`git diff --check` 无输出、exit 0。最终 `git log -1` 仍为 `5fa88d0`；所有修改只在工作区。

## 预算与验证边界

**ZERO paid model calls：付费模型调用 0，真实模型/provider 调用 0，生产密钥读取 0。**

- 常规测试和消息前后回放使用既有 `e2e/offline_checks.py` 的隔离环境/审计守卫，禁止网络、真实凭据文件和未隔离子进程。未运行会尝试读取凭据路径的 self-test。S67 的模型入口另设调用即失败。
- S63 通过 `env -i` 和既有 fixture 专用入口执行；临时 shell 替身未启动真实 bot、builder、ingest、Docker 或安装程序。
- 只读访问本机参考库抽取静态 fixture；本任务唯一成功的线上业务请求是用户明确要求的、无需认证的 `yeoiav` 公开 GET，已在测试前单独核验并记录。
- 未运行 `pnpm test`、真实 provider E2E、SSH、生产 DB 查询、迁移、提交、推送或部署。未修改原始参考库、生产状态和旁仓库文件。
- 证据等级：静态源码/AST/diff；fixture/fake-tool 定向执行；六套离线与 safety；公开实时链事实。没有把离线测试称作已部署或生产修复验证。
