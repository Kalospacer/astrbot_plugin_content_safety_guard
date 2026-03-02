import json
import re
from typing import Any

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import logger


# ═══════════════════════════════════════════════════════════════
# 全角 → 半角 映射表
# ═══════════════════════════════════════════════════════════════
_FULLWIDTH = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９"
_HALFWIDTH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_FW2HW_TABLE = str.maketrans(_FULLWIDTH, _HALFWIDTH)


@register(
    name="astrbot_plugin_content_safety_guard",
    author="Kalo",
    desc="内容安全守卫 - 拦截不合规的LLM回复，引导模型重新生成合规内容，替代内置内容安全模块",
    version="1.1.0",
    repo="https://github.com/Kalospacer/astrbot_plugin_content_safety_guard",
)
class ContentSafetyGuardPlugin(Star):
    """内容安全守卫插件

    替代 AstrBot 内置的内容安全模块。当 LLM 回复不合规时，
    不是简单地屏蔽，而是将拦截原因注入提示词，引导模型重新生成合规回复。
    重试完毕后接回原工作流正常发送。

    检测策略（按顺序执行，任一命中即判定不安全）:
    1. 关键词/正则匹配（带文本归一化预处理）
    2. 百度 AIP 内容审核
    3. LLM 自审查（用 LLM 自身判断回复是否合规）

    工作流程:
    1. [on_llm_request] 保存原始请求上下文 + 可选的用户输入检查
    2. [on_llm_response] 检查 LLM 回复 → 不合规则重试 → 替换为合规内容
    """

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}

        # ─── 基本配置 ───
        self.max_retries: int = self.config.get("max_retries", 2)
        self.check_input: bool = self.config.get("check_input", False)
        self.check_output: bool = self.config.get("check_output", True)

        # ─── 关键词配置 ───
        keywords_cfg = self.config.get("keywords", {})
        self.keywords_enabled: bool = keywords_cfg.get("enable", True)
        self.keywords_list: list[str] = keywords_cfg.get("extra_keywords", [])

        # ─── 百度 AIP 配置 ───
        baidu_cfg = self.config.get("baidu_aip", {})
        self.baidu_enabled: bool = baidu_cfg.get("enable", False)
        self.baidu_client: Any = None
        if self.baidu_enabled:
            try:
                from aip import AipContentCensor

                self.baidu_client = AipContentCensor(
                    baidu_cfg.get("app_id", ""),
                    baidu_cfg.get("api_key", ""),
                    baidu_cfg.get("secret_key", ""),
                )
            except ImportError:
                logger.warning(
                    "[ContentSafetyGuard] 使用百度内容审核需要先 pip install baidu-aip"
                )
                self.baidu_enabled = False

        # ─── LLM 自审查配置 ───
        llm_audit_cfg = self.config.get("llm_audit", {})
        self.llm_audit_enabled: bool = llm_audit_cfg.get("enable", False)
        self.llm_audit_provider: str = llm_audit_cfg.get("provider_id", "")
        self.llm_audit_model: str = llm_audit_cfg.get("model", "")
        self.llm_audit_prompt: str = llm_audit_cfg.get(
            "prompt",
            (
                "你是一名内容安全审查员。请判断以下文本是否包含不适当内容"
                "（包括但不限于：暴力、色情、违法犯罪、歧视、自残自杀引导等）。\n"
                '请严格仅返回如下 JSON 格式，不要包含任何额外文字：\n'
                '{"safe": true/false, "reason": "简要说明原因"}\n\n'
                "待审查文本：\n```\n{text}\n```"
            ),
        )

        # ─── 提示词配置 ───
        self.safety_prompt_template: str = self.config.get(
            "safety_prompt",
            (
                "你之前的回复未通过内容安全审核，被拦截了。\n"
                "拦截原因：{reason}\n\n"
                "请重新回复用户的消息，保持有用和完整，但务必确保内容合规、安全，"
                "不要包含任何敏感或不适当的内容。"
            ),
        )
        self.block_message: str = self.config.get(
            "block_message", "抱歉，我无法回答这个问题。请换一个话题吧～"
        )
        self.input_block_message: str = self.config.get(
            "input_block_message", "你的消息包含不适当的内容，已被屏蔽。"
        )

        logger.info(
            f"[ContentSafetyGuard] 已加载 v1.1.0 | "
            f"关键词: {'启用' if self.keywords_enabled else '禁用'}({len(self.keywords_list)}个) | "
            f"百度AIP: {'启用' if self.baidu_enabled else '禁用'} | "
            f"LLM审查: {'启用' if self.llm_audit_enabled else '禁用'}"
            f"{(' [' + (self.llm_audit_provider or '默认') + ']') if self.llm_audit_enabled else ''} | "
            f"最大重试: {self.max_retries} | "
            f"检查输入: {self.check_input} | 检查输出: {self.check_output}"
        )

    # ══════════════════════════════════════════════════════════════
    # 文本归一化
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_text(text: str) -> str:
        """归一化文本，防止通过空格/特殊字符/全角字符等手段绕过关键词检测。

        处理策略:
        - 去除零宽字符（ZWJ/ZWNJ/ZWSP/BOM/Soft-Hyphen 等）
        - 去除中文字符间的空白（防止 "自 杀" 绕过 "自杀"）
        - 全角字母/数字转半角
        - 统一为小写
        """
        if not text:
            return ""
        # 去除零宽字符
        s = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u180e]", "", text)
        # 去除中文字符间插入的空格干扰
        s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
        # 全角 → 半角
        s = s.translate(_FW2HW_TABLE)
        # 小写
        s = s.lower()
        return s

    # ══════════════════════════════════════════════════════════════
    # 内容安全检查
    # ══════════════════════════════════════════════════════════════

    def _check_keywords(self, text: str, normalized: str) -> tuple[bool, str]:
        """关键词 / 正则检查（对归一化后的文本进行匹配）"""
        if not self.keywords_enabled or not self.keywords_list:
            return True, ""

        for keyword in self.keywords_list:
            try:
                if re.search(keyword, normalized) or re.search(keyword, text):
                    return False, f"匹配到敏感词规则: {keyword}"
            except re.error:
                if keyword.lower() in normalized:
                    return False, f"匹配到敏感词: {keyword}"
        return True, ""

    def _check_baidu_aip(self, text: str) -> tuple[bool, str]:
        """百度 AIP 内容审核"""
        if not self.baidu_enabled or not self.baidu_client:
            return True, ""

        try:
            res = self.baidu_client.textCensorUserDefined(text)
            if "conclusionType" not in res:
                return False, "百度审核服务返回异常"
            if res["conclusionType"] != 1:
                if "data" in res:
                    parts = []
                    for item in res["data"]:
                        if isinstance(item, dict):
                            msg = item.get("msg", "")
                            if msg:
                                parts.append(msg)
                    reason = (
                        f"百度审核不通过: {'; '.join(parts)}"
                        if parts
                        else "百度审核不通过"
                    )
                else:
                    reason = f"百度审核不通过: {res.get('conclusion', '未知原因')}"
                return False, reason
        except Exception as e:
            logger.error(f"[ContentSafetyGuard] 百度AIP调用失败: {e}")
        return True, ""

    async def _check_llm_audit(self, text: str) -> tuple[bool, str]:
        """LLM 自审查 — 用 LLM 自身判断文本是否合规"""
        if not self.llm_audit_enabled:
            return True, ""

        try:
            # 确定使用的 Provider
            provider_id = self.llm_audit_provider
            if not provider_id:
                prov = self.context.get_using_provider()
                if not prov:
                    logger.warning("[ContentSafetyGuard] LLM审查: 找不到可用的 Provider")
                    return True, ""
                provider_id = prov.meta().id

            audit_prompt = self.llm_audit_prompt.format(text=text)

            # 可选指定模型
            kwargs: dict[str, Any] = {}
            if self.llm_audit_model:
                kwargs["model"] = self.llm_audit_model

            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=audit_prompt,
                system_prompt="你是内容安全审查员。只返回 JSON，不要附加任何解释。",
                **kwargs,
            )

            result_text = (resp.completion_text or "").strip()
            return self._parse_llm_audit_result(result_text)

        except Exception as e:
            logger.error(f"[ContentSafetyGuard] LLM审查调用失败: {e}")
            # 调用失败时放行，避免误杀
            return True, ""

    @staticmethod
    def _parse_llm_audit_result(text: str) -> tuple[bool, str]:
        """解析 LLM 审查返回的 JSON 结果"""
        if not text:
            return True, ""

        # 提取 JSON
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                is_safe = data.get("safe", True)
                reason = str(data.get("reason", ""))
                if isinstance(is_safe, bool):
                    if not is_safe:
                        return False, f"LLM审查不通过: {reason}" if reason else "LLM审查不通过"
                    return True, ""
                # 非 bool 类型，尝试解析
                if str(is_safe).lower() in ("false", "0", "no"):
                    return False, f"LLM审查不通过: {reason}" if reason else "LLM审查不通过"
                return True, ""
            except (json.JSONDecodeError, ValueError):
                pass

        # JSON 解析失败，简单启发式判断
        lowered = text.lower()
        if any(w in lowered for w in ("不安全", "不合规", "unsafe", '"safe": false', '"safe":false')):
            return False, f"LLM审查不通过: {text[:100]}"
        return True, ""

    async def check_content_safety(self, text: str) -> tuple[bool, str]:
        """组合安全检查：关键词 → 百度AIP → LLM审查，任一不通过即返回

        Returns:
            (is_safe, reason): 安全返回 (True, ""), 不安全返回 (False, reason)
        """
        if not text or not text.strip():
            return True, ""

        normalized = self._normalize_text(text)

        # 1. 关键词检查（最快，零成本）
        is_safe, reason = self._check_keywords(text, normalized)
        if not is_safe:
            return False, reason

        # 2. 百度 AIP 检查
        is_safe, reason = self._check_baidu_aip(text)
        if not is_safe:
            return False, reason

        # 3. LLM 自审查（最慢，但语义理解最强）
        is_safe, reason = await self._check_llm_audit(text)
        if not is_safe:
            return False, reason

        return True, ""

    # ══════════════════════════════════════════════════════════════
    # 事件钩子
    # ══════════════════════════════════════════════════════════════

    @filter.on_llm_request()
    async def on_llm_request_hook(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        """LLM 请求前钩子

        1. 保存原始请求的 system_prompt，供重试时复用
        2. 可选：检查用户输入内容安全
        """
        # 保存原始 system_prompt，供 on_llm_response 重试时使用
        if self.check_output:
            event.set_extra("_csg_system_prompt", request.system_prompt or "")

        # 用户输入内容安全检查
        if not self.check_input:
            return

        text = request.prompt or event.get_message_str()
        if not text or not text.strip():
            return

        is_safe, reason = await self.check_content_safety(text)
        if is_safe:
            return

        logger.info(f"[ContentSafetyGuard] 用户输入未通过安全检查: {reason}")
        event.set_result(self.input_block_message)
        event.stop_event()

    @filter.on_llm_response()
    async def on_llm_response_hook(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """LLM 响应后钩子

        检查 LLM 回复内容安全：
        - 通过 → 放行，工作流继续
        - 不通过 → 将拦截原因注入提示词，调用 LLM 重新生成
        - 重试全部失败 → 替换为安全默认消息
        """
        if not self.check_output:
            return

        # 跳过流式响应的 chunk（仅检查完整响应）
        if response.is_chunk:
            return

        # 获取回复文本
        text = response.completion_text
        if not text or not text.strip():
            return

        # 内容安全检查
        is_safe, reason = await self.check_content_safety(text)
        if is_safe:
            return

        logger.info(f"[ContentSafetyGuard] LLM回复未通过安全检查: {reason}")

        # 获取当前 Provider，用于重试
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.warning(
                    "[ContentSafetyGuard] 无法获取当前 Provider，直接替换为安全消息"
                )
                response.completion_text = self.block_message
                return
            provider_id = provider.meta().id
        except Exception as e:
            logger.error(f"[ContentSafetyGuard] 获取 Provider 失败: {e}")
            response.completion_text = self.block_message
            return

        # 获取保存的上下文
        original_message = event.get_message_str()
        system_prompt = event.get_extra("_csg_system_prompt", "")

        # ─── 重试循环 ───
        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"[ContentSafetyGuard] 第 {attempt}/{self.max_retries} 次重新生成"
            )

            # 构建安全引导提示
            safety_guidance = self.safety_prompt_template.format(reason=reason)
            retry_prompt = f"{original_message}\n\n[系统安全提示] {safety_guidance}"

            try:
                new_resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=retry_prompt,
                    system_prompt=system_prompt if system_prompt else None,
                )

                new_text = new_resp.completion_text
                if not new_text or not new_text.strip():
                    logger.warning(
                        f"[ContentSafetyGuard] 第 {attempt} 次重试返回空文本"
                    )
                    continue

                # 检查新回复
                new_is_safe, new_reason = await self.check_content_safety(new_text)
                if new_is_safe:
                    logger.info(
                        f"[ContentSafetyGuard] 第 {attempt} 次重试通过安全检查 ✓"
                    )
                    # 替换原回复文本，工作流继续
                    response.completion_text = new_text
                    return
                else:
                    logger.info(
                        f"[ContentSafetyGuard] 第 {attempt} 次重试仍不通过: {new_reason}"
                    )
                    reason = new_reason  # 更新原因，供下次重试使用

            except Exception as e:
                logger.error(
                    f"[ContentSafetyGuard] 第 {attempt} 次重试 LLM 调用失败: {e}"
                )

        # 所有重试都失败，使用安全兜底消息
        logger.warning(
            f"[ContentSafetyGuard] {self.max_retries} 次重试均失败，使用安全默认消息"
        )
        response.completion_text = self.block_message
