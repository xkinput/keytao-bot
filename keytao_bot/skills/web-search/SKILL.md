---
name: web-search
description: 通用网络搜索和网页正文抓取工具。当用户询问实时信息、站外资料、新闻动态、外部网页内容、明确要求搜索，或你对答案没有把握时必须调用。不要凭记忆编造搜索结果。优先返回多个候选来源，并保留原始链接。
---

# 通用网络搜索

为 AI 助手提供网络搜索和网页正文抓取能力，用于补充键道站内工具无法覆盖的外部信息。

## 适用场景

当用户出现以下意图时，应调用此 skill：

- 明确要求搜索：`搜一下`、`查下网上怎么说`、`帮我找资料`
- 需要实时信息：新闻、版本发布、近期动态、公告
- 需要外部网页来源：博客、文档、论坛、GitHub、官网页面
- 用户发来 URL，需要阅读网页内容
- 你对回答没有把握，或问题涉及近期/可能变化的信息
- 当前仓库内的其他 skill 无法回答的问题

## 何时不要调用

- 纯闲聊
- 键道词条编码查询：应调用 `keytao-lookup`
- 键道官方文档规则说明：应调用 `keytao-docs`
- 键道词库增删改草稿操作：应调用 `keytao-draft`

## 工具定义

### web_search

执行多来源通用网络搜索，返回标题、摘要和链接；支持顺手抓取前几个结果的正文。

参数：

- `query` (string, required): 搜索词或完整问题
- `max_results` (integer, optional): 返回结果数量，默认 5，范围 1-10
- `fetch_top_n` (integer, optional): 抓取前 N 个结果正文，默认 0，范围 0-3。当搜索摘要不足以回答时设置为 1-3。

返回格式：

```json
{
  "success": true,
  "query": "nonebot2 function calling",
  "provider": "multi",
  "providersTried": ["duckduckgo-html", "duckduckgo-lite", "bing"],
  "results": [
    {
      "title": "NoneBot2 Documentation",
      "url": "https://nonebot.dev/",
      "snippet": "Cross-platform Python chatbot framework...",
      "provider": "duckduckgo-lite"
    }
  ],
  "fetchedPages": [
    {
      "title": "NoneBot2 Documentation",
      "url": "https://nonebot.dev/",
      "text": "页面正文节选..."
    }
  ],
  "count": 1
}
```

### web_fetch

抓取指定网页正文，适合用户直接发 URL，或搜索结果摘要不够时继续读取原文。

参数：

- `url` (string, required): 要抓取的网页 URL
- `max_chars` (integer, optional): 返回正文字符数，默认 4000，范围 800-12000

返回格式：

```json
{
  "success": true,
  "url": "https://example.com/article",
  "title": "Article title",
  "description": "meta description",
  "text": "网页正文节选...",
  "truncated": true
}
```

## 使用要求

- 明确要求搜索、实时信息、外部事实、近期版本、价格、新闻、规则变化、官网文档时，必须先调用 `web_search`
- 不确定、不懂、记忆里没有、或用户问“现在/最新/最近”时，先搜索；拿到有价值的新内容后再回答
- 用户给 URL，或搜索摘要不足以判断时，调用 `web_fetch`
- 回答必须反馈信息来源：至少列出 1-3 个标题或链接；不要把链接吞掉
- 如果搜索结果不确定，应明确说明“这是搜索结果，不是确定事实”
- 如果多个来源互相矛盾，说明分歧并保守回答
- 搜索失败时不要编造；说明搜索失败原因，并可让用户稍后重试或换关键词
