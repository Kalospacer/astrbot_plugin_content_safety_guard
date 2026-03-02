# Content Safety Guard（内容安全守卫）

AstrBot 内容安全插件：支持输入前置拦截、输出重生、黑名单和多种审核策略。

## 当前行为概览

- 用户输入检查（`check_input=true`）在 **LLM 请求前** 执行，不通过会直接拦截，不调用模型。
- 模型输出检查（`check_output=true`）在 **LLM 响应后** 执行，不通过会按 `max_retries` 重试重生。
- 可选“重复回复拦截”：如果回复与同会话上一条回复完全一致，会触发重生。
- 可选“占位回复开关”：拦截后可回复提示，也可静默拦截。
- 可选“仅群聊生效”及“群聊非管理员 `/` 指令拦截”。

## 审核策略

按顺序执行，任一命中即不通过：

1. 关键词 / 正则匹配（带文本归一化：去零宽字符、去中文字符间空格、统一小写）
2. 百度 AIP 文本审核（可选）
3. LLM 自审查（可选）

## 配置说明

### 基本

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `max_retries` | `2` | 输出不合规时的最大重试次数 |
| `check_input` | `false` | 是否在请求前检查用户输入 |
| `check_output` | `true` | 是否检查模型输出 |
| `block_duplicate_reply` | `true` | 是否拦截与上一条模型回复完全一致的输出 |
| `reply_placeholder_on_block` | `true` | 拦截后是否回复占位消息（false=静默） |
| `group_only` | `true` | 是否仅在群聊生效 |
| `block_non_admin_slash_in_group` | `true` | 群聊中是否拦截非管理员 `/` 开头消息 |

### 关键词

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `keywords.enable` | `true` | 是否启用关键词检查 |
| `keywords.extra_keywords` | `[]` | 自定义关键词/正则列表 |

### 百度 AIP

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `baidu_aip.enable` | `false` | 是否启用百度审核 |
| `baidu_aip.app_id` | `""` | App ID |
| `baidu_aip.api_key` | `""` | API Key |
| `baidu_aip.secret_key` | `""` | Secret Key |

### LLM 自审查

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `llm_audit.enable` | `false` | 是否启用 LLM 审查 |
| `llm_audit.provider_id` | `""` | 默认审查提供商（下拉选择） |
| `llm_audit.input_provider_id` | `""` | 输入前置审查提供商（可单独配置） |
| `llm_audit.output_provider_id` | `""` | 输出审查提供商（可单独配置） |
| `llm_audit.prompt` | 内置模板 | 审查提示词模板（需返回 JSON） |

### 提示词与拦截文案

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `safety_prompt` | 内置模板 | 输出重生提示词（含 `{reason}`） |
| `prevention_prompt` | 内置模板 | 注入 system_prompt 的预防提示（含 `{keywords}`） |
| `block_message` | `抱歉，我无法回答...` | 输出重试失败后的兜底消息 |
| `input_block_message` | `你的消息包含...` | 输入拦截提示（仅 `reply_placeholder_on_block=true`） |

### 黑名单

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `blacklist.enable` | `false` | 是否启用自动拉黑 |
| `blacklist.duration` | `60` | 拉黑时长（分钟，`0`=永久） |
| `blacklist.max_violations` | `3` | 触发拉黑的违规次数 |
| `blacklist.message` | 内置文案 | 黑名单提示（仅首次且需占位回复开关开启） |

## 管理指令

- `/csgbl add <user_id> [分钟]`：手动拉黑（`0`=永久）
- `/csgbl del <user_id>`：解封并清除记录
- `/csgbl ls`：查看黑名单、剩余时长、到期时间
- `/csgbl clear`：清空黑名单与违规记录

## 安装

```bash
cd /path/to/astrbot/data/stars
git clone https://github.com/Kalospacer/astrbot_plugin_content_safety_guard.git
```

## 许可证

MIT
