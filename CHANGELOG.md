# Changelog

## v1.2.6

- 修复前置输入拦截在 `reply_placeholder_on_block=true` 时只设置提示消息、未终止事件传播，导致主模型仍继续请求的问题。
- 将 `llm_audit.mode` 的配置界面从自由输入改为下拉选择，提供 `separate` 和 `combined_post` 两个模式选项。
- 新增 `check_input_original_only` 配置项；开启后，用户输入审查只检查原始用户消息文本，不再把记忆/RAG 等插件注入到 `request.prompt` 的内容一并当作用户输入审查。
- 修复与记忆注入类插件联动时，前置输入审核可能因为注入内容命中关键词而误判的问题。
- 优化白名单与 LLM 审查相关配置说明，便于在 AstrBot 面板中直接理解实际行为。
