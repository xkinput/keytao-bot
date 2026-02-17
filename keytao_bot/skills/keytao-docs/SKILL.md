---
name: keytao-docs
description: 键道输入法文档查询工具。从 GitHub 仓库获取真实的键道文档内容，支持查询编码规则、学习方法、安装指南等。
version: "2.0.0"
author: xkinput
---

# 键道文档查询

为 AI 助手提供查询键道输入法官方文档的能力，从 GitHub 源码仓库实时获取最新的文档内容。

## 数据来源

从 GitHub 仓库实时获取：https://github.com/xkinput/keytao-docs

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

- **规则/编码**：获取音码和形码编码规则
- **学习/教程/入门**：获取学习方法和入门指南
- **安装/下载**：获取安装部署文档
- **字根/笔画**：获取字根和笔画说明
- **单字**：获取单字编码规则
- **词组**：获取词组编码规则
- **顶功**：获取顶功上屏机制说明
- **简码**：获取简码使用说明

## 提示

- 官网地址: https://keytao.vercel.app
- 文档地址: https://keytao-docs.vercel.app
- 该工具从 GitHub 实时获取最新文档内容
- 每次查询最多返回3个相关文档片段
- 内容自动清理 markdown frontmatter，保留核心内容
