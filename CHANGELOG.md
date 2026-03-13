# Changelog

## v1.2.6

- 修复前置输入拦截在 `reply_placeholder_on_block=true` 时只设置提示消息、未终止事件传播，导致主模型仍继续请求的问题。
- 将 `llm_audit.mode` 的配置界面从自由输入改为下拉选择，提供 `separate` 和 `combined_post` 两个模式选项。
- 优化白名单与 LLM 审查相关配置说明，便于在 AstrBot 面板中直接理解实际行为。
