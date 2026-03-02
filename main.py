import asyncio
import json
import re
import time
from typing import Any

from pathlib import Path

from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import logger


@register(
    name="astrbot_plugin_content_safety_guard",
    author="Kalo",
    desc="内容安全守卫 - 拦截不合规的LLM回复，引导模型重新生成合规内容，替代内置内容安全模块",
    version="1.2.0",
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
    1. [on_llm_request] 保存原始请求上下文（system_prompt + 用户文本）
    2. [on_llm_response] 同时审查用户输入 + AI 回复 → 用户不通过直接拦截 / AI 不通过引导重试
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
                "请判断以下文本是否包含以下要素：{keywords}\n"
                "请严格仅返回如下 JSON 格式，不要包含任何额外文字：\n"
                '{"safe": true/false, "reason": "简要说明原因"}\n\n'
                "待审查文本：\n{text}"
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
        self.prevention_prompt: str = self.config.get(
            "prevention_prompt",
            (
                "【内容安全约束】你的回复不得包含以下要素：{keywords}。"
                "包含此类内容的回复将会被系统拦截。"
            ),
        )

        # ─── 黑名单配置 ───
        blacklist_cfg = self.config.get("blacklist", {})
        self.blacklist_enabled: bool = blacklist_cfg.get("enable", False)
        self.blacklist_duration: int = blacklist_cfg.get("duration", 60)  # 分钟，0=永久
        self.blacklist_max_violations: int = blacklist_cfg.get("max_violations", 3)
        self.blacklist_message: str = blacklist_cfg.get(
            "message", "你已因多次发送违规内容被临时限制使用，请稍后再试。"
        )

        # ─── 黑名单运行时状态（持久化到 data 目录）───
        self._blacklist: dict[str, float] = {}  # sender_id → 解封时间戳
        self._violations: dict[str, int] = {}  # sender_id → 累计违规次数
        self._data_file: Path | None = None
        self._cleanup_task: asyncio.Task | None = None
        if self.blacklist_enabled:
            try:
                self._data_file = StarTools.get_data_dir() / "blacklist.json"
                self._load_blacklist()
            except Exception as e:
                logger.warning(f"[ContentSafetyGuard] 加载黑名单数据失败: {e}")
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_bans())

        logger.info(
            f"[ContentSafetyGuard] 已加载 v1.1.0 | "
            f"关键词: {'启用' if self.keywords_enabled else '禁用'}({len(self.keywords_list)}个) | "
            f"百度AIP: {'启用' if self.baidu_enabled else '禁用'} | "
            f"LLM审查: {'启用' if self.llm_audit_enabled else '禁用'}"
            f"{(' [' + (self.llm_audit_provider or '默认') + ']') if self.llm_audit_enabled else ''} | "
            f"最大重试: {self.max_retries} | "
            f"检查输入: {self.check_input} | 检查输出: {self.check_output} | "
            f"黑名单: {'启用' if self.blacklist_enabled else '禁用'}"
            f"{(f'({self.blacklist_max_violations}次违规/{self.blacklist_duration}分钟)') if self.blacklist_enabled else ''}"
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
        - 统一为小写
        """
        if not text:
            return ""
        # 去除零宽字符
        s = re.sub(
            r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u180e]", "", text
        )
        # 去除中文字符间插入的空格干扰
        s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
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
                    logger.warning(
                        "[ContentSafetyGuard] LLM审查: 找不到可用的 Provider"
                    )
                    return True, ""
                provider_id = prov.meta().id

            audit_prompt = self.llm_audit_prompt.format(
                text=text,
                keywords="、".join(self.keywords_list)
                if self.keywords_list
                else "无指定",
            )

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
                        return (
                            False,
                            f"LLM审查不通过: {reason}" if reason else "LLM审查不通过",
                        )
                    return True, ""
                # 非 bool 类型，尝试解析
                if str(is_safe).lower() in ("false", "0", "no"):
                    return (
                        False,
                        f"LLM审查不通过: {reason}" if reason else "LLM审查不通过",
                    )
                return True, ""
            except (json.JSONDecodeError, ValueError):
                pass

        # JSON 解析失败，简单启发式判断
        lowered = text.lower()
        if any(
            w in lowered
            for w in ("不安全", "不合规", "unsafe", '"safe": false', '"safe":false')
        ):
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

    def _check_fast(self, text: str) -> tuple[bool, str]:
        """快速检查：关键词 + 百度AIP（不含 LLM 审查，零延迟）"""
        if not text or not text.strip():
            return True, ""
        normalized = self._normalize_text(text)
        is_safe, reason = self._check_keywords(text, normalized)
        if not is_safe:
            return False, reason
        is_safe, reason = self._check_baidu_aip(text)
        if not is_safe:
            return False, reason
        return True, ""

    async def _check_llm_audit_combined(
        self, user_text: str, ai_text: str
    ) -> tuple[str, str]:
        """组合 LLM 审查 — 一次调用同时审查用户输入和 AI 回复

        Returns:
            ("pass", "")       — 均安全
            ("user", reason)   — 用户输入不安全
            ("ai", reason)     — AI 回复不安全
        """
        if not self.llm_audit_enabled:
            return "pass", ""

        try:
            provider_id = self.llm_audit_provider
            if not provider_id:
                prov = self.context.get_using_provider()
                if not prov:
                    logger.warning(
                        "[ContentSafetyGuard] LLM审查: 找不到可用的 Provider"
                    )
                    return "pass", ""
                provider_id = prov.meta().id

            keywords_str = (
                "、".join(self.keywords_list) if self.keywords_list else "无指定"
            )
            prompt = (
                f"请分别判断以下对话中的【用户消息】和【AI回复】是否包含以下要素：{keywords_str}\n"
                "请严格仅返回如下 JSON 格式，不要包含任何额外文字：\n"
                '{"user_safe": true/false, "ai_safe": true/false, "reason": "简要说明原因"}\n\n'
                f"【用户消息】：\n{user_text}\n\n"
                f"【AI回复】：\n{ai_text}"
            )

            kwargs: dict[str, Any] = {}
            if self.llm_audit_model:
                kwargs["model"] = self.llm_audit_model

            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt="你是内容安全审查员。只返回 JSON，不要附加任何解释。",
                **kwargs,
            )

            result_text = (resp.completion_text or "").strip()
            return self._parse_combined_audit_result(result_text)

        except Exception as e:
            logger.error(f"[ContentSafetyGuard] LLM组合审查调用失败: {e}")
            return "pass", ""

    @staticmethod
    def _parse_combined_audit_result(text: str) -> tuple[str, str]:
        """解析组合审查返回的 JSON 结果"""
        if not text:
            return "pass", ""

        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                user_safe = data.get("user_safe", True)
                ai_safe = data.get("ai_safe", True)
                reason = str(data.get("reason", ""))
                # 归一化为 bool
                if not isinstance(user_safe, bool):
                    user_safe = str(user_safe).lower() not in ("false", "0", "no")
                if not isinstance(ai_safe, bool):
                    ai_safe = str(ai_safe).lower() not in ("false", "0", "no")
                if not user_safe:
                    return (
                        "user",
                        f"LLM审查不通过(用户输入): {reason}"
                        if reason
                        else "LLM审查不通过(用户输入)",
                    )
                if not ai_safe:
                    return (
                        "ai",
                        f"LLM审查不通过(AI回复): {reason}"
                        if reason
                        else "LLM审查不通过(AI回复)",
                    )
                return "pass", ""
            except (json.JSONDecodeError, ValueError):
                pass

        # 解析失败，启发式归因到 AI
        lowered = text.lower()
        if any(
            w in lowered
            for w in ("不安全", "不合规", "unsafe", '"safe": false', '"safe":false')
        ):
            return "ai", f"LLM审查不通过: {text[:100]}"
        return "pass", ""

    # ══════════════════════════════════════════════════════════════
    # 黑名单管理
    # ══════════════════════════════════════════════════════════════

    def _load_blacklist(self) -> None:
        """从磁盘加载黑名单数据"""
        if not self._data_file or not self._data_file.exists():
            return
        try:
            data = json.loads(self._data_file.read_text(encoding="utf-8"))
            now = time.time()
            # 加载时过滤已过期条目
            for uid, expiry in data.get("blacklist", {}).items():
                if expiry == float("inf") or expiry > now:
                    self._blacklist[uid] = expiry
            self._violations = data.get("violations", {})
            logger.info(
                f"[ContentSafetyGuard] 已加载黑名单数据: "
                f"{len(self._blacklist)} 个封禁, {len(self._violations)} 个违规记录"
            )
        except Exception as e:
            logger.warning(f"[ContentSafetyGuard] 解析黑名单文件失败: {e}")

    def _save_blacklist(self) -> None:
        """将黑名单数据保存到磁盘"""
        if not self._data_file:
            return
        try:
            data = {
                "blacklist": {uid: exp for uid, exp in self._blacklist.items()},
                "violations": dict(self._violations),
            }
            self._data_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"[ContentSafetyGuard] 保存黑名单数据失败: {e}")

    def _is_blacklisted(self, sender_id: str) -> bool:
        """检查用户是否在黑名单中（同时清理过期条目）"""
        if not self.blacklist_enabled or not sender_id:
            return False
        if sender_id not in self._blacklist:
            return False
        expiry = self._blacklist[sender_id]
        if expiry == float("inf") or time.time() < expiry:
            return True
        # 已过期，移除
        del self._blacklist[sender_id]
        self._violations.pop(sender_id, None)
        self._save_blacklist()
        logger.info(f"[ContentSafetyGuard] 用户 {sender_id} 封禁已到期，已自动解封")
        return False

    def _add_violation(self, sender_id: str, reason: str) -> bool:
        """记录违规并判断是否需要拉黑。返回 True 表示已被拉黑。"""
        if not self.blacklist_enabled or not sender_id:
            return False
        count = self._violations.get(sender_id, 0) + 1
        self._violations[sender_id] = count
        if count >= self.blacklist_max_violations:
            if self.blacklist_duration > 0:
                expiry = time.time() + self.blacklist_duration * 60
            else:
                expiry = float("inf")
            self._blacklist[sender_id] = expiry
            duration_text = (
                f"{self.blacklist_duration} 分钟"
                if self.blacklist_duration > 0
                else "永久"
            )
            logger.warning(
                f"[ContentSafetyGuard] 用户 {sender_id} 累计违规 {count} 次，"
                f"已加入黑名单（{duration_text}）。原因: {reason}"
            )
            self._save_blacklist()
            return True
        logger.info(
            f"[ContentSafetyGuard] 用户 {sender_id} 违规 {count}/{self.blacklist_max_violations}"
        )
        self._save_blacklist()
        return False

    async def _cleanup_expired_bans(self):
        """后台协程：定期清理过期的黑名单条目"""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                uid
                for uid, expiry in self._blacklist.items()
                if expiry != float("inf") and now >= expiry
            ]
            for uid in expired:
                del self._blacklist[uid]
                self._violations.pop(uid, None)
                logger.info(f"[ContentSafetyGuard] 用户 {uid} 封禁已到期，已自动解封")
            if expired:
                self._save_blacklist()

    async def terminate(self):
        """插件卸载时取消后台任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    # ══════════════════════════════════════════════════════════════
    # 事件钩子
    # ══════════════════════════════════════════════════════════════

    @filter.on_llm_request()
    async def on_llm_request_hook(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        """LLM 请求前钩子 — 黑名单拦截 + 保存上下文 + 注入安全约束到 system_prompt"""
        # ─── 黑名单检查（最高优先级）───
        sender_id = event.get_sender_id()
        if self._is_blacklisted(sender_id):
            logger.info(f"[ContentSafetyGuard] 黑名单用户 {sender_id} 请求已拦截")
            event.set_result(self.blacklist_message)
            event.stop_event()
            return

        event.set_extra("_csg_system_prompt", request.system_prompt or "")
        event.set_extra("_csg_user_text", request.prompt or event.get_message_str())

        # 注入屏蔽词约束到 system_prompt，从源头引导 LLM 规避敏感内容
        if self.keywords_list:
            prevention = self.prevention_prompt.format(
                keywords="、".join(self.keywords_list)
            )
            if request.system_prompt:
                request.system_prompt = f"{request.system_prompt}\n\n{prevention}"
            else:
                request.system_prompt = prevention

    @filter.on_llm_response()
    async def on_llm_response_hook(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """LLM 响应后钩子 — 同时审查用户输入和 AI 回复

        流程:
        1. 快速检查（关键词 + 百度AIP）用户输入 → 不通过直接拦截
        2. 快速检查 AI 回复 → 不通过进入重试
        3. LLM 组合审查（一次调用同时审查双方）→ 用户不通过拦截 / AI 不通过重试
        4. 重试时仅检查 AI 新回复（用户输入已通过）
        """
        if not (self.check_input or self.check_output):
            return

        if response.is_chunk:
            return

        ai_text = response.completion_text
        if not ai_text or not ai_text.strip():
            return

        user_text = event.get_extra("_csg_user_text", "") if self.check_input else ""

        # ─── 快速检查（关键词 + 百度AIP）───
        if user_text and user_text.strip():
            is_safe, reason = self._check_fast(user_text)
            if not is_safe:
                logger.info(f"[ContentSafetyGuard] 用户输入未通过快速检查: {reason}")
                self._add_violation(event.get_sender_id(), reason)
                event.set_result(self.input_block_message)
                event.stop_event()
                return

        ai_fail_reason = ""
        if self.check_output:
            is_safe, reason = self._check_fast(ai_text)
            if not is_safe:
                ai_fail_reason = reason

        # ─── LLM 审查（一次调用同时审查用户输入 + AI 回复）───
        if not ai_fail_reason and self.llm_audit_enabled:
            has_user = bool(self.check_input and user_text and user_text.strip())
            has_ai = self.check_output

            if has_user and has_ai:
                target, reason = await self._check_llm_audit_combined(
                    user_text, ai_text
                )
                if target == "user":
                    logger.info(f"[ContentSafetyGuard] 用户输入未通过LLM审查: {reason}")
                    self._add_violation(event.get_sender_id(), reason)
                    event.set_result(self.input_block_message)
                    event.stop_event()
                    return
                elif target == "ai":
                    ai_fail_reason = reason
            elif has_ai:
                is_safe, reason = await self._check_llm_audit(ai_text)
                if not is_safe:
                    ai_fail_reason = reason
            elif has_user:
                is_safe, reason = await self._check_llm_audit(user_text)
                if not is_safe:
                    logger.info(f"[ContentSafetyGuard] 用户输入未通过LLM审查: {reason}")
                    self._add_violation(event.get_sender_id(), reason)
                    event.set_result(self.input_block_message)
                    event.stop_event()
                    return

        # 全部通过
        if not ai_fail_reason:
            return

        logger.info(f"[ContentSafetyGuard] LLM回复未通过安全检查: {ai_fail_reason}")

        # ─── 获取 Provider ───
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

        original_message = event.get_message_str()
        system_prompt = event.get_extra("_csg_system_prompt", "")
        reason = ai_fail_reason

        # ─── 重试循环（仅检查 AI 新回复，用户输入已通过）───
        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"[ContentSafetyGuard] 第 {attempt}/{self.max_retries} 次重新生成"
            )

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

                new_is_safe, new_reason = await self.check_content_safety(new_text)
                if new_is_safe:
                    logger.info(
                        f"[ContentSafetyGuard] 第 {attempt} 次重试通过安全检查 ✓"
                    )
                    response.completion_text = new_text
                    return
                else:
                    logger.info(
                        f"[ContentSafetyGuard] 第 {attempt} 次重试仍不通过: {new_reason}"
                    )
                    reason = new_reason

            except Exception as e:
                logger.error(
                    f"[ContentSafetyGuard] 第 {attempt} 次重试 LLM 调用失败: {e}"
                )

        logger.warning(
            f"[ContentSafetyGuard] {self.max_retries} 次重试均失败，使用安全默认消息"
        )
        response.completion_text = self.block_message
