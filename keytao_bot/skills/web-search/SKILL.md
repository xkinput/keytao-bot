---
name: web-search
description: 通用网络搜索工具。当用户询问实时信息、站外资料、新闻动态、外部网页内容或明确要求“帮我搜一下”时必须调用。不要凭记忆编造搜索结果。优先返回多个候选来源，并保留原始链接。
---

# 通用网络搜索

为 AI 助手提供基础网络搜索能力，用于补充键道站内工具无法覆盖的外部信息。

## 适用场景

当用户出现以下意图时，应调用此 skill：

- 明确要求搜索：`搜一下`、`查下网上怎么说`、`帮我找资料`
- 需要实时信息：新闻、版本发布、近期动态、公告
- 需要外部网页来源：博客、文档、论坛、GitHub、官网页面
- 当前仓库内的其他 skill 无法回答的问题

## 何时不要调用

- 纯闲聊
- 键道词条编码查询：应调用 `keytao-lookup`
- 键道官方文档规则说明：应调用 `keytao-docs`
- 键道词库增删改草稿操作：应调用 `keytao-draft`

## 工具定义

### web_search

执行一次通用网络搜索，返回标题、摘要和链接。

参数：

- `query` (string, required): 搜索词或完整问题
- `max_results` (integer, optional): 返回结果数量，默认 5，范围 1-10

返回格式：

```json
{
  "success": true,
  "query": "nonebot2 function calling",
  "provider": "duckduckgo",
  "results": [
    {
      "title": "NoneBot2 Documentation",
      "url": "https://nonebot.dev/",
      "snippet": "Cross-platform Python chatbot framework..."
    }
  ],
  "count": 1
}
```

## 使用要求

- 优先保留原始来源链接，不要把链接吞掉
- 如果搜索结果不确定，应明确说明“这是搜索结果，不是确定事实”
- 如果用户要更深入的内容，可以基于结果链接继续回答，但不要编造未搜索到的信息