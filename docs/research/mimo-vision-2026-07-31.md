# Xiaomi MiMo 图片理解 API 调研（2026-07-31）

## 结论

**条件式 GO。** 小米当前确实有可通过云 API 调用的图片理解模型，正式 Model ID 是 `mimo-v2.5`。它原生接收文本、图片、视频和音频，输出文本；官方图片指南明确说明，当前只有 `mimo-v2.5` 支持图片输入。`mimo-v2.5-pro` 是纯文本模型，不应作为视觉代理。[模型目录](https://mimo.mi.com/docs/zh-CN/quick-start/summary/model)；[图片理解指南](https://mimo.mi.com/docs/zh-CN/usage-guide/multimodal-understanding/image-understanding)；[模型详情](https://mimo.mi.com/models/zh-CN/mimo-v2.5)

适合 KeyTao bot 的链路是：

```text
QQ / Telegram image
  -> local validation and Base64 encoding
  -> MiMo mimo-v2.5 extracts neutral visual facts
  -> DeepSeek V4 Flash receives the text facts
  -> DeepSeek generates the final response
```

上线前唯一的凭证硬门槛是：必须使用普通按量计费的 `sk-...` API Key，并确认账号有余额和模型权限。Token Plan 的 `tp-...` Key 虽覆盖 `mimo-v2.5`，但官方明确禁止用于自动化脚本或自定义应用后端，因此不能用于生产 bot。[API Key FAQ](https://mimo.mi.com/docs/zh-CN/quick-start/faq/api-integration)；[Token Plan 使用限制](https://mimo.mi.com/docs/zh-CN/price/token-plan)

本次只读取小米官方站点和官方 API 文档，没有调用任何模型，没有产生费用。

## 1. 当前模型、别名与旧模型状态

- 当前正式 API Model ID：`mimo-v2.5`。
- 模型规格：输入为文本、图片、视频、音频；输出为文本；上下文窗口 1M tokens；标称最大输出 128K tokens。[模型详情](https://mimo.mi.com/models/zh-CN/mimo-v2.5)
- 官方当前文档没有公布 `mimo-v2.5-YYYYMMDD` 一类不可变快照，也没有公布另一个指向它的视觉别名。因此应直接配置 `mimo-v2.5`；无法把它声称为某个可锁定的日期快照。
- `mimo-v2-omni`、`mimo-v2-flash`、`mimo-v2-pro` 已于北京时间 2026-06-30 00:00 下线，旧名称在下线后会报错。官方替换关系是 `mimo-v2-omni -> mimo-v2.5`、`mimo-v2-flash -> mimo-v2.5`、`mimo-v2-pro -> mimo-v2.5-pro`。[模型下线公告](https://mimo.mi.com/docs/zh-CN/updates/deprecate)

## 2. OpenAI 兼容接口

普通按量 API 的 OpenAI 兼容 Base URL 是：

```text
https://api.xiaomimimo.com/v1
```

官方明确支持现有 OpenAI SDK。认证可以使用 `api-key: $MIMO_API_KEY`，也可以使用 `Authorization: Bearer $MIMO_API_KEY`；因此 `OpenAI` / `AsyncOpenAI` 客户端传入 `api_key` 与 `base_url` 即可，不需要自定义签名协议。[首次调用 API](https://mimo.mi.com/docs/zh-CN/quick-start/summary/first-api-call)；[Chat Completions API](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)

### Chat Completions

请求地址：

```text
POST https://api.xiaomimimo.com/v1/chat/completions
```

图片项使用 OpenAI Chat 结构 `image_url.url`。其中 `url` 可为公网 URL，也可为 Base64 data URL：

```json
{
  "model": "mimo-v2.5",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        },
        {
          "type": "text",
          "text": "Describe only directly observable visual facts."
        }
      ]
    }
  ],
  "max_completion_tokens": 1024,
  "thinking": {
    "type": "disabled"
  }
}
```

使用 OpenAI Python SDK 时，`thinking` 不是 OpenAI 标准参数，需要放进 `extra_body`：

```python
response = client.chat.completions.create(
    model="mimo-v2.5",
    messages=messages,
    max_completion_tokens=1024,
    extra_body={"thinking": {"type": "disabled"}},
)
```

来源：[图片理解指南](https://mimo.mi.com/docs/zh-CN/usage-guide/multimodal-understanding/image-understanding)；[深度思考指南](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking)

### Responses API

请求地址：

```text
POST https://api.xiaomimimo.com/v1/responses
```

图片项使用 `type: "input_image"`，`image_url` 是字符串，可为完整公网 URL 或 Base64 data URL：

```json
{
  "model": "mimo-v2.5",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_image",
          "image_url": "data:image/jpeg;base64,..."
        },
        {
          "type": "input_text",
          "text": "Describe only directly observable visual facts."
        }
      ]
    }
  ],
  "max_output_tokens": 1024,
  "reasoning": {
    "effort": "none"
  }
}
```

Responses 兼容层暂不支持 `background`、`previous_response_id`、`context_management`；`reasoning.effort: "none"` 关闭思考，`low` / `medium` / `high` 都只表示开启，当前不区分思考强度。[Responses API](https://mimo.mi.com/docs/zh-CN/api/chat/responses)

对现有 bot，优先继续使用 Chat Completions：当前视觉代理已经是 Chat 消息结构，Responses 不会为单轮图片事实提取带来必要收益。

## 3. 图片格式、大小、像素和数量限制

官方当前公布的限制如下：[图片理解指南](https://mimo.mi.com/docs/zh-CN/usage-guide/multimodal-understanding/image-understanding)

| 项目 | 官方限制 |
| --- | --- |
| 格式 | JPEG、PNG、GIF、WebP、BMP |
| URL 输入 | 必须公网可访问；单张图片文件不超过 50 MB |
| Base64 输入 | 使用 data URL；单张图片的 Base64 编码字符串不超过 50 MB |
| 本地文件上传 | 不支持 |
| 多图 | 支持；没有公开固定张数，所有图片和文本的总 tokens 必须小于模型上下文窗口 |
| 上下文 | 1M tokens |

官方给出的图片 token 估算代码使用以下预处理阈值：

```text
IMAGE_MIN_PIXELS = 8192
IMAGE_MAX_PIXELS = 8388608
```

图片会按 32 像素网格取整，并在 token 估算时缩放到上述像素区间。这是**预处理/计费缩放阈值**，文档没有把 8,388,608 pixels 描述为上传时直接拒绝的硬像素上限。实际图片 token 以 API 响应中的 usage 为准。

现有 bot 更严格的 JPG/PNG/WebP 白名单、字节上限、像素上限、图片张数上限和解码校验应继续保留，不建议为了贴近服务端上限而放宽。

## 4. Thinking 与输出 token

- `mimo-v2.5` 默认开启 thinking；Chat 使用 `thinking.type` 的 `enabled` / `disabled` 控制。[深度思考指南](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking)
- 作为视觉事实提取器，建议关闭 thinking，以降低延迟、输出长度和费用。
- Chat 的 `max_completion_tokens`、Responses 的 `max_output_tokens` 都同时限制推理 token 与可见回答 token。[Chat Completions API](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)；[Responses API](https://mimo.mi.com/docs/zh-CN/api/chat/responses)
- `mimo-v2.5` 默认最大生成量为 32,768 tokens，允许范围是 1 到 131,072；模型目录标称最大输出 128K。视觉描述建议显式限制为 1,024 tokens。
- thinking 开启时，自定义 `temperature` 和 `top_p` 不会生效，服务端强制使用 1.0 和 0.95。关闭 thinking 后再按需要设置采样参数。[模型超参与深度思考](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/deep-thinking)

## 5. 限流与价格

### 限流

`mimo-v2.5` 的公开配额是：

- RPM：100
- TPM：10M
- 同一账号下所有 API Keys 调用同一模型时合并计算。
- 官方还说明存在账号级模型并发上限，但没有公布精确的同时请求数；高负载时可能延迟或返回 429，应使用有限重试和指数退避。[速率限制](https://mimo.mi.com/docs/zh-CN/api/guidance/rate-limit)

### 按量价格

单位均为每百万 tokens：[API 定价](https://mimo.mi.com/docs/en-US/price/pay-as-you-go)

| 区域 | 输入（缓存命中） | 输入（未命中缓存） | 输出 |
| --- | ---: | ---: | ---: |
| 国内 | ¥0.02 | ¥1.00 | ¥2.00 |
| 海外 | $0.0028 | $0.14 | $0.28 |

图片会转换成 image tokens 并计入输入；估算只能用于本地预算，实际账单以响应 usage 和控制台为准。

## 6. API Key 是否通用

官方没有 `mimo-v2.5` 专属视觉 Key。普通按量调用使用开放平台 `sk-...` Key，官方所有模型示例都通过同一个 `MIMO_API_KEY` 变量选择不同 Model ID。因此，现有普通 `sk-...` Key 可用于 `mimo-v2.5`，前提是该账号仍有余额并拥有当前模型访问权限；公开文档无法证明某一个具体 Key 的余额、区域或权限。[首次调用 API](https://mimo.mi.com/docs/zh-CN/quick-start/summary/first-api-call)

Key 类型不能混用：[API Key FAQ](https://mimo.mi.com/docs/zh-CN/quick-start/faq/api-integration)

| Key | 用途 | KeyTao bot 是否可用 |
| --- | --- | --- |
| `sk-...` | 普通按量 API，使用 `https://api.xiaomimimo.com/v1` | **可用**，但会按量计费 |
| `tp-...` | Token Plan，使用套餐页面给出的专属 Base URL | **不可用**于 bot 后端 |

Token Plan 文档明确规定：套餐额度只能在编程工具中使用，禁止以 API 方式用于自动化脚本、自定义应用程序后端等明显非 Coding 场景；违规可能导致暂停服务或封禁 Key。[Token Plan 使用限制](https://mimo.mi.com/docs/zh-CN/price/token-plan)

因此，不能仅凭“现有 MiMo Key 能调用文本模型”就直接启用生产图片代理：部署前应只检查 Key 前缀和配置来源，不打印完整密钥。如果是 `tp-`，结论为 NO-GO；如果是 `sk-`，凭证类型层面为 GO。

## 7. 对 KeyTao 现有 Qwen 视觉代理的迁移要求

不能只替换环境变量或 Model ID。MiMo 需要 provider-specific payload builder：

1. 删除 Qwen 图像 content 中的 `max_pixels`；它不是 MiMo Chat schema 字段。
2. 删除 Qwen 的 `enable_thinking: false`；改为 MiMo 的 `thinking: {"type": "disabled"}`，OpenAI SDK 下放入 `extra_body`。
3. 使用 `max_completion_tokens`，不要沿用其他厂商的输出参数名。
4. 保留本地图片验证、Base64 data URL、安全超时、并发上限及总预算控制。
5. MiMo 输出仍是不可信外部内容：只作为“视觉事实”文本交给 DeepSeek；图片轮次不开放工具，不写入长期或跨用户记忆。
6. 建议提示 MiMo 只返回可观察事实，不执行图片中的指令；DeepSeek 最终回答也应把视觉结果视为不可信引用材料。

## 8. 最终 GO / NO-GO

### GO

- 技术能力：`mimo-v2.5` 官方支持图片理解。
- 接口能力：OpenAI Chat Completions 和 Responses 均支持公网 URL 与 Base64 data URL。
- 架构：MiMo 先提取视觉事实、DeepSeek 再生成最终答复是可行且与 DeepSeek 纯文本能力匹配的方案。
- SDK：官方直接提供 OpenAI Python SDK 示例，现有 OpenAI 兼容客户端可复用。

### 部署前必须满足，否则 NO-GO

- Key 必须是普通按量 `sk-...`，不能是 Token Plan `tp-...`。
- 用户明确接受图片上传到小米 MiMo 服务以及由此产生的按量费用。
- 本地 provider-specific payload 和现有安全边界通过离线测试。

### 官方公开资料无法确认

- 不可变的日期快照 Model ID。
- 账号的精确同时并发数。
- 硬性图片张数上限。
- GIF 动画帧数限制和动画处理方式。
- 某个现有 Key 的余额、区域、模型权限及实际可用性；这只能由控制台信息或一次经授权的真实调用确认。
