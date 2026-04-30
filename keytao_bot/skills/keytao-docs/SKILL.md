---
name: keytao-docs
description: 键道输入法文档查询工具。当用户询问概念性问题时调用。使用场景：编码规则、学习方法、安装部署、概念解释、Lua功能（时间日期数字简码提示单字模式）。通过 GitHub Code Search API 在官方文档仓库中全文搜索，无需手动维护关键词映射。
---

# 展示格式规范

## 展示要求
- 直接展示文档内容（content 字段）
- 内容已清理 frontmatter，直接使用即可
- 如果内容过长，展示核心部分
- 结尾附上来源链接，**必须原封不动使用 sources 字段中的 URL，禁止自行推断或修改路径**
- 如果 sources 为空或工具返回 success=false，只说"未找到相关文档，请访问 https://keytao-docs.vercel.app 查阅"，**不得捏造任何链接**

## ⚠️ 严禁行为
- 禁止根据文档标题猜测 URL（例如不能因为标题含"lua"就写 `/guide/advanced/lua-features`）
- 禁止在 sources 字段之外自行生成任何 keytao-docs.vercel.app 链接
- 禁止在工具未返回结果时假装找到了文档

## 展示示例
```
【键道编码规则】

[文档内容...]

---
📖 完整文档：{sources[0] 原值，不修改}
```

# 键道文档查询

为 AI 助手提供查询键道输入法官方文档的能力，从 GitHub 源码仓库实时获取最新的文档内容。

## 数据来源

通过 **GitHub Code Search API** 在官方文档仓库中全文搜索，再拉取匹配的 markdown 原文：
- 搜索接口：`GET https://api.github.com/search/code?q={query}+repo:xkinput/keytao-docs+extension:md`
- 原文拉取：`https://raw.githubusercontent.com/xkinput/keytao-docs/main/{path}`

无需手动维护关键词映射，任何文档内容均可搜索到。

## 使用场景

当用户提出以下类型问题时，应该调用此 skill：

- "键道的规则是什么？"
- "键道的编码规则"
- "键道怎么学？"  
- "键道有什么学习教程？"
- "键道的字根怎么记？"
- "怎么安装键道输入法？"
- "单字怎么打？"
- "词组编码规则"
- "什么是顶功？"
- "键道有哪些 Lua 功能？"
- "怎么输入今天的日期？"
- "键道简码提示怎么用？"
- "单字模式怎么开启？"

## 工具定义

### keytao_fetch_docs

从键道文档 GitHub 仓库获取相关内容。

**参数：**
- `query` (string, required): 要查询的问题或关键词，如"规则"、"学习"、"字根"、"安装"等

**返回格式：**
```json
{
  "success": true,
  "query": "规则",
  "content": "【键道形码】\n\n# 键道形码\n\n...",
  "sources": ["https://keytao-docs.vercel.app/guide/learn-xkjd/stroke-rules"],
  "matched_keywords": ["规则"],
  "hint": "更多详细信息请访问: https://keytao-docs.vercel.app"
}
```

## 支持的查询关键词

无限制——底层使用 GitHub Code Search 全文搜索，直接传入用户的自然语言问题即可，无需转换为特定关键词。

典型示例：
- `顶功` → 返回顶功上屏规则文档
- `Lua 时间日期` → 返回 lua-features.md 中时间/日期功能说明
- `简码提示` → 返回简码提示相关内容
- `单字模式` → 返回 keytao_filter 单字模式说明

## 提示

- 官网地址: https://keytao.vercel.app
- 文档地址: https://keytao-docs.vercel.app
- iOS 元书输入法码表包稳定下载链接（自动指向最新版）：https://keytao.vercel.app/api/install/ios-latest
- 该工具从 GitHub 实时获取最新文档内容
- 每次查询最多返回3个相关文档片段
- 内容自动清理 markdown frontmatter，保留核心内容
