# LLM 供应商选型对比（2026-09）

> 存放位置说明：本文放在 `docs/research/`，而非任务里写的 `docs/` 根目录 —— 本仓库已有 `docs/research/mimo-vision-2026-07-31.md` 这一「调研笔记按 `docs/research/<主题>-<日期>.md` 归档」的既有约定，`docs/incidents/` 则用于事故记录，遵循现有约定不新开目录。
>
> 调研日期：2026-09-08。汇率：**USD→CNY = 6.71**（[tradingeconomics.com/china/currency](https://tradingeconomics.com/china/currency)，2026-09-08 读取）。

## 1. 一句话结论

**主力换成智谱 GLM-5.3-Flash（¥22.95/千轮，AA 智能指数 42，工具调用榜前三），DeepSeek V4 Flash 降级为兜底通道** —— 但如果本 bot 的流量本来就集中在晚间和周末（DeepSeek 高峰期仅工作日 9:00–12:00、14:00–18:00，周末全天低谷价），那这次涨价对你的实际账单影响远小于新闻标题，**不换也是一个正当选项**，先去账单里核实峰谷分布再决定。

## 2. 价格与质量总表

价格单位：**元/百万 tokens**；「缓存」= 缓存命中（cache-hit）输入价。AA = Artificial Analysis Intelligence Index v4.2（[artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models)，2026-09-08 读取）。

| 模型 | 输入 | 缓存命中 | 输出 | 上下文 | AA | 大陆直连 | 支付宝/微信 |
|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash**（智谱） | 0.8（限时 0.4） | 0.23（限时 0.115） | 2.8（限时 1.4） | 1M | **42** | ✅ 官方 | ✅ |
| **Qwen3.8-Flash**（百炼） | 0.8 | 0.1 | 2.7 | 1M | 42* | ✅ 官方 | ✅ |
| **DeepSeek V4 Flash** 低谷 | 1.5 | 0.05 | 4.5 | 1M | 35 | ✅ 官方 | ✅ |
| DeepSeek V4 Flash 高峰 | 3.0 | 0.10 | 9.0 | 1M | 35 | ✅ 官方 | ✅ |
| DeepSeek V4 Pro 低谷 | 4.5 | 0.15 | 13.5 | 1M | 36 | ✅ 官方 | ✅ |
| **GLM-5.3**（智谱旗舰） | 8 | 2 | 28 | 1M | **45** | ✅ 官方 | ✅ |
| **Kimi K3** | 20 | 2 | 100 | 1M | 44 | ✅ 官方 | ✅ |
| Kimi K2.6 | 6.5 | 1.1 | 27 | 256K | 未核实 | ✅ 官方 | ✅ |
| MiniMax-M3（≤512K） | 2.1 | 0.42 | 8.4 | 1M | 30 | ✅ 官方 | ✅ |
| Doubao-Seed-2.1-turbo | 3 | 0.6 †  | 15 | 256K | 未核实 | ✅ 官方 | ✅ |
| Doubao-Seed-2.0-mini（≤32K 档） | 0.2 † | 0.04 † | 2 † | 256K | 未核实 | ✅ 官方 | ✅ |
| 腾讯 Hunyuan-a13b | 0.5 | 未核实 | 2 | 未核实 | 未核实 | ✅ 官方 | ✅ |
| 百度 ERNIE 5.1 | 4 † | 未核实 | 18 † | 未核实 | 未核实 | ✅ 官方 | ✅ |
| Gemini 3.1 Flash-Lite | 1.68（$0.25） | 0.168（$0.025） | 10.07（$1.50） | — | 23‡ | ❌ | ❌ 需外卡 |
| Gemini 3.8 Flash | 5.03（$0.75） | 0.503（$0.075） | 25.16（$3.75） | — | 41 | ❌ | ❌ 需外卡 |
| GPT-5.4-nano | 1.34（$0.20） | 0.134（$0.02） | 8.39（$1.25） | — | 未核实 | ❌ | ❌ 需外卡 |
| GPT-5.4-mini | 5.03（$0.75） | 0.503（$0.075） | 30.20（$4.50） | — | 未核实 | ❌ | ❌ 需外卡 |
| Claude Haiku 4.5 | 6.71（$1） | 0.671（$0.10） | 33.55（$5） | — | 18 | ❌ | ❌ 需外卡 |

\* AA 榜上的条目名为 **Qwen3.8-Flash-Next**（42 分），与百炼售卖的 `qwen3.8-flash` 是否同一模型**未核实**，此分数存疑。
† 来源为二手（新闻/社区），火山引擎官方价格页 `docs.volcengine.com/docs/82379/1544106` 反复抓取只返回导航骨架、表格未渲染，**Doubao 全部数字均未经一手核实**。
‡ 榜上为 Gemini 3.5 Flash-Lite（23 分），3.1 Flash-Lite 未单列。

**DeepSeek 这次到底改了什么**：2026-08-13 公告，2026-08-17 00:00 起启用峰谷计费，高峰时段为周一至周五 9:00–12:00 与 14:00–18:00，低谷价为高峰价的一半；V4 Pro 高峰输出从 6 元涨到 27 元（+350%），V4 Flash 缓存命中价涨幅最高被报道为 1100%。一周后 2026-08-23 00:00 起再次调整：**周六周日全天统一按低谷价**。

## 3. 按本工作负载的成本估算

**口径**：生产单轮 = 55k 输入（85% 命中缓存 → 46.75k 命中 + 8.25k 未命中）+ 2k 输出。测试台单跑 = 1.6M tokens（90% 命中缓存 1.44M / 5% 未命中输入 0.08M / 5% 输出 0.08M）。

```
千轮成本(元) = 46.75 × P_缓存 + 8.25 × P_输入 + 2 × P_输出
单跑成本(元) =  1.44 × P_缓存 + 0.08 × P_输入 + 0.08 × P_输出
（P 均为元/百万 tokens）
```

| 模型 | 千轮成本 | 测试台单跑 |
|---|---|---|
| **GLM-5.3-Flash**（限时 5 折，9/9 截止） | **¥11.48** | ¥0.31 |
| **Qwen3.8-Flash** | **¥16.68** | ¥0.42 |
| **GLM-5.3-Flash**（原价） | **¥22.95** | ¥0.62 |
| **DeepSeek V4 Flash（低谷/周末）** | **¥23.71** | ¥0.55 |
| GPT-5.4-nano | ¥34.12 | ¥0.97 |
| Gemini 3.1 Flash-Lite | ¥41.82 | ¥1.18 |
| DeepSeek V4 Flash（高峰） | ¥47.43 | ¥1.10 |
| MiniMax-M3 | ¥53.76 | ¥1.45 |
| DeepSeek V4 Pro（低谷） | ¥71.14 | ¥1.66 |
| Doubao-Seed-2.1-turbo † | ¥82.80 | ¥2.30 |
| Gemini 3.8 Flash | ¥115.42 | ¥3.14 |
| GPT-5.4-mini | ¥125.44 | ¥3.54 |
| Claude Haiku 4.5 | ¥153.83 | ¥4.19 |
| Kimi K2.6 | ¥159.05 | ¥4.26 |
| GLM-5.3 | ¥215.50 | ¥5.76 |
| Kimi K3 | ¥458.50 | ¥12.48 |

注意成本结构：85% 命中率下，**未命中输入 + 输出占了账单的 70–90%**，缓存价只占一小块。所以「缓存价便宜」的 DeepSeek（0.05）反而输给缓存价贵 4.6 倍但输出价便宜 60% 的 GLM-5.3-Flash。降本的真正杠杆是**压输出长度和减少未命中**，不是挑缓存单价最低的家。

## 4. 三个最优选择

**最便宜可用 —— Qwen3.8-Flash（¥16.68/千轮）**
百炼官方页一手核实（输入 0.8 / 缓存 0.1 / 输出 2.7，1M 上下文，Function Call 支持，2026-08-27 12:00 起降价）。阿里云本来就是这台服务器的宿主，同区内网延迟最低，账号和发票链路已经在。唯一风险是 AA 分数无法与售卖型号对上，中文质量需自己跑一轮回归再切。

**性价比最高 —— GLM-5.3-Flash（原价 ¥22.95/千轮，AA 42）**
比 DeepSeek V4 Flash 便宜 3%，智能指数高 7 分（42 vs 35），AA 的三项 agentic 评测全面领先（AA-Briefcase 1455 vs 1259、GDPval-AA v2 1669 vs 1468、AutomationBench 60% vs 54%）；工具调用综合榜排第三（33.8，仅次于 Kimi K3 和 GLM-5.3）。1M 上下文、缓存存储限时免费。**限时 5 折到 2026-09-09 24:00**——如果要试，今天就该建号跑测试台，5 折期内单跑仅 ¥0.31。

**最聪明 —— GLM-5.3（AA 45，¥215.50/千轮）**
榜首梯队里唯一大陆官方可直连的选择，比同档 Kimi K3（AA 44，¥458.50）便宜 53%。不适合当主力，适合做「难轮升级」路由：常规轮走 Flash，工具循环失败或用户显式要求深度推理时升到 GLM-5.3。

**建议落地形态**：主力 GLM-5.3-Flash + 难轮升级 GLM-5.3 + DeepSeek V4 Flash 保留为故障兜底（key 不注销，权重降到 0）。Qwen3.8-Flash 作为第二兜底 —— 同厂商同区，DeepSeek 和智谱同时挂掉的概率极低。

## 5. 具体注意事项

**可达性**：DeepSeek / 百炼 / 火山方舟 / Kimi / 智谱 / MiniMax / 腾讯 / 百度全部为大陆境内官方服务，无需代理。OpenAI 自 2024-07 起按 IP 封锁大陆与香港，大陆不在其 supported countries 名单内；Gemini 与 Anthropic 同样不对大陆开放且需境外卡 —— 三家国际厂商在本工作负载下**直接出局**，表中列出仅作价格锚点。

**计费**：所列国产平台均支持支付宝/微信充值、纯按量后付费、无月费无闲置费。智谱、百炼、腾讯、火山均有新用户免费额度（百炼每模型 100 万 tokens；腾讯 100 万 tokens/有效期 1 年）。国际三家需 Visa/Master 外卡。

**工具调用差异**：GLM-5.3 系列官方明确支持 Function Calling + MCP；Qwen3.8-Flash 官方标注支持 Function Call；MiniMax-M3 支持 `tools`/`tool_choice`。~30 个 tool schema 的多步循环对 schema 遵从度敏感，**切换前必须跑一次完整测试台**（成本 ¥0.3–0.6，几乎免费，没有理由跳过）。

**推理强度控制**：DeepSeek 支持 `reasoning_effort`（如 `"high"`）与 `thinking: {type: enabled}`；Qwen3.8-Flash 有独立 thinking 模式（思维链上限 262144 tokens）。换供应商时这个参数名不统一，是最容易漏掉的适配点。

**缓存机制差异（重要）**：百炼的**隐式缓存自动开启且无法关闭**，命中价通常为输入价的 20%（qwen3.8-flash 实际为 0.1/0.8 = 12.5%）；显式缓存命中按输入价 10% 计费，但**创建缓存要按输入价 125% 收费**——55k 系统提示每次重建都是净亏，务必用隐式缓存、别手动开显式。智谱缓存存储限时免费。MiniMax 缓存写入单独收费 ¥2.625/M。Claude 的 cache write 溢价（$1.25 vs $1）本表未计入，实际会更贵。

**限流**：Kimi 按累计充值分 6 档（¥0 档只有 3 RPM / 150 万 TPD，¥50 档即升到 100 RPM / 200 万 TPM 且 TPD 无限）—— 单轮 55k tokens 的负载在 Tier 0 上跑不动，要先充值。其余厂商的按量档限流未核实，上线前需实测。

## 6. 来源列表

一手（官方，2026-09-08 读取）：
- DeepSeek 价格：https://api-docs.deepseek.com/zh-cn/quick_start/pricing
- DeepSeek 推理参数：https://api-docs.deepseek.com/zh-cn/guides/reasoning_model
- DeepSeek 涨价公告：https://api-docs.deepseek.com/zh-cn/news/news260813
- 智谱价格表：https://docs.bigmodel.cn/cn/guide/start/pricing.md
- 智谱模型总览：https://docs.bigmodel.cn/cn/guide/start/model-overview
- 百炼价格表：https://help.aliyun.com/zh/model-studio/model-pricing
- 百炼 qwen3.8-flash：https://help.aliyun.com/zh/model-studio/qwen3-8-flash
- 百炼上下文缓存计费：https://help.aliyun.com/zh/model-studio/context-cache
- Kimi 定价（K3 / K2.6）：https://platform.kimi.com/docs/pricing/chat-k3 、 https://platform.kimi.com/docs/pricing/chat-k26.md
- Kimi 限流分档：https://platform.kimi.com/docs/pricing/limits.md
- MiniMax 按量计费：https://platform.minimaxi.com/docs/guides/pricing-paygo
- 腾讯混元计费：https://cloud.tencent.com/document/product/1729/97731
- 火山引擎模型市场（含价格，无缓存价）：https://ai.volcengine.com/model
- OpenAI 价格：https://developers.openai.com/api/docs/pricing
- Gemini 价格：https://ai.google.dev/gemini-api/docs/pricing
- Anthropic 价格：https://claude.com/pricing
- Artificial Analysis 榜单：https://artificialanalysis.ai/leaderboards/models
- AA 对比页（GLM-5.3-Flash vs DeepSeek V4 Flash）：https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-deepseek-v4-flash
- 汇率：https://tradingeconomics.com/china/currency

二手（已在正文标注，未获一手核实）：
- 火山方舟 Doubao 各档价格与缓存命中价：搜索摘要引自 https://docs.volcengine.com/docs/82379/1544106 与 https://www.volcengine.com/article/2602982 —— **本机对该页多次抓取仅返回导航骨架，表格未渲染**
- DeepSeek 周末低谷价调整（2026-08-23 生效）：https://www.ithome.com/0/993/095.htm
- DeepSeek 涨价幅度（V4 Pro 输出 6→27 元、Flash 缓存 +1100%）：https://m.21jingji.com/article/20260823/herald/7689cba6304e220f4d616798549da51d.html
- GLM-5.3-Flash 限时 5 折截止 2026-09-09：https://qcode.cc/glm-5-3-flash-pricing
- 工具调用综合榜（Kimi K3 35.7 / GLM-5.3 35.3 / GLM-5.3-Flash 33.8）：https://llm-stats.com/leaderboards/best-ai-for-tool-calling
- 百度 ERNIE 5.1 价格：https://www.llmabacus.com/models/ernie-5-1
- 腾讯 Hunyuan-TurboS 价格：https://www.txybk.com/hunyuan
- OpenAI 大陆封锁政策：https://www.scmp.com/tech/policy/article/3267971/tech-war-openai-further-block-access-mainland-china-hong-kong-based-developers

抓取失败记录（环境 TLS 限制）：`www.alibabacloud.com/help/zh/model-studio/model-pricing`、`platform.openai.com/docs/pricing`、`cloud.tencent.com/document/product/1729/97731`（首次 socket closed，重试成功）曾返回 `Socket is closed`；`docs.volcengine.com/docs/82379/1544106` 可连通但表格未渲染，Doubao 数据因此降级为二手。
