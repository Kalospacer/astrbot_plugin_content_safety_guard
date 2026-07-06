import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

TEMP_DATA_DIR = (
    Path(tempfile.gettempdir()) / "astrbot_plugin_content_safety_guard_tests"
)
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODULE_PATH = Path(__file__).resolve().parents[1] / "main.py"
README_PATH = Path(__file__).resolve().parents[1] / "README.md"
METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata.yaml"


class DummyCommandGroup:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def command(self, _name):
        return lambda func: func


class DummyFilter:
    class EventMessageType:
        GROUP_MESSAGE = "group"

    class PermissionType:
        ADMIN = "admin"

    @staticmethod
    def command_group(_name):
        return lambda func: DummyCommandGroup(func)

    @staticmethod
    def permission_type(_permission):
        return lambda func: func

    @staticmethod
    def event_message_type(*_args, **_kwargs):
        return lambda func: func

    @staticmethod
    def on_llm_request():
        return lambda func: func

    @staticmethod
    def on_llm_response():
        return lambda func: func


class DummyStar:
    def __init__(self, context):
        self.context = context


class DummyStarTools:
    @staticmethod
    def get_data_dir() -> Path:
        return TEMP_DATA_DIR


def register(**_kwargs):
    return lambda cls: cls


def build_stub_modules() -> dict[str, ModuleType]:
    logger = MagicMock()
    astrbot_module = ModuleType("astrbot")
    api_module = ModuleType("astrbot.api")
    star_module = ModuleType("astrbot.api.star")
    event_module = ModuleType("astrbot.api.event")
    provider_module = ModuleType("astrbot.api.provider")

    star_module.Context = object
    star_module.Star = DummyStar
    star_module.StarTools = DummyStarTools
    star_module.register = register

    event_module.AstrMessageEvent = object
    event_module.filter = DummyFilter()

    provider_module.LLMResponse = object
    provider_module.ProviderRequest = object

    api_module.logger = logger
    astrbot_module.api = api_module

    return {
        "astrbot": astrbot_module,
        "astrbot.api": api_module,
        "astrbot.api.star": star_module,
        "astrbot.api.event": event_module,
        "astrbot.api.provider": provider_module,
    }


def load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "content_safety_guard_main", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    with patch.dict(sys.modules, build_stub_modules()):
        spec.loader.exec_module(module)

    return module


module = load_plugin_module()
ContentSafetyGuardPlugin = module.ContentSafetyGuardPlugin
AUDIT_SYSTEM_PROMPT = module.AUDIT_SYSTEM_PROMPT


class DummyProvider:
    def __init__(self, provider_id: str = "dummy-provider"):
        self._provider_id = provider_id

    def meta(self):
        return SimpleNamespace(id=self._provider_id)


class DummyContext:
    def __init__(self, provider_id: str = "dummy-provider"):
        self.provider = DummyProvider(provider_id)
        self.llm_calls: list[dict] = []

    def get_using_provider(self, *_args, **_kwargs):
        return self.provider

    async def llm_generate(self, **kwargs):
        self.llm_calls.append(kwargs)
        return SimpleNamespace(completion_text='{"safe": true, "reason": ""}')


class DummyEvent:
    def __init__(
        self,
        sender_id: str = "10001",
        platform_name: str = "aiocqhttp",
        unified_msg_origin: str = "aiocqhttp:group:session-1",
    ):
        self.sender_id = sender_id
        self.platform_name = platform_name
        self._extras: dict[str, object] = {}
        self.result = None
        self.stopped = False
        self.message = "hello"
        self.unified_msg_origin = unified_msg_origin

    def is_private_chat(self):
        return False

    def get_sender_id(self):
        return self.sender_id

    def get_platform_name(self):
        return self.platform_name

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_result(self, value):
        self.result = value

    def stop_event(self):
        self.stopped = True

    def get_message_str(self):
        return self.message

    def get_session_id(self):
        return "session-1"


class DummyRequest:
    def __init__(self, prompt: str = "hello", system_prompt: str = ""):
        self.prompt = prompt
        self.system_prompt = system_prompt


def build_plugin(config: dict | None = None) -> ContentSafetyGuardPlugin:
    plugin = ContentSafetyGuardPlugin(DummyContext(), config or {})
    plugin._cleanup_task = None
    return plugin


def run(coro):
    return asyncio.run(coro)


def test_register_metadata_and_docs_are_unified():
    metadata_text = METADATA_PATH.read_text(encoding="utf-8-sig")
    readme_text = README_PATH.read_text(encoding="utf-8")

    assert module.PLUGIN_DISPLAY_NAME == "阿瓦隆 Avalon"
    assert module.PLUGIN_AUTHOR == "Kalospacer"
    assert module.PLUGIN_VERSION == "1.2.6"
    assert "LLM 自审查" in module.PLUGIN_DESC
    assert AUDIT_SYSTEM_PROMPT == "你是内容安全审查员。只返回 JSON，不要附加任何解释。"
    assert f'display_name: "{module.PLUGIN_DISPLAY_NAME}"' in metadata_text
    assert f"version: {module.PLUGIN_VERSION}" in metadata_text
    assert module.PLUGIN_DESC in metadata_text
    assert readme_text.splitlines()[0] == f"# {module.PLUGIN_DISPLAY_NAME}"


def test_whitelist_matches_unified_msg_origin_sid():
    plugin = build_plugin(
        {
            "whitelist": {
                "enable": True,
                "sids": ["aiocqhttp:group:session-1"],
            }
        }
    )
    event = DummyEvent(unified_msg_origin="aiocqhttp:group:session-1")
    assert plugin._is_whitelisted(event) is True


def test_blacklist_has_priority_over_whitelist_in_request_hook():
    plugin = build_plugin(
        {
            "whitelist": {
                "enable": True,
                "sids": ["aiocqhttp:group:session-1"],
            }
        }
    )
    plugin.blacklist_enabled = True
    plugin._blacklist["10001"] = float("inf")

    event = DummyEvent(
        sender_id="10001",
        platform_name="aiocqhttp",
        unified_msg_origin="aiocqhttp:group:session-1",
    )
    request = DummyRequest()
    run(plugin.on_llm_request_hook(event, request))

    assert event.stopped is True


def test_separate_mode_input_audit_uses_shared_system_prompt():
    plugin = build_plugin(
        {
            "check_input": True,
            "llm_audit": {
                "enable": True,
                "mode": "separate",
                "provider_id": "audit-provider",
            },
        }
    )
    event = DummyEvent(sender_id="10002")
    request = DummyRequest(prompt="need audit")

    run(plugin.on_llm_request_hook(event, request))

    assert plugin.context.llm_calls
    assert plugin.context.llm_calls[0]["system_prompt"] == AUDIT_SYSTEM_PROMPT


def test_combined_post_skips_input_stage_llm_audit():
    plugin = build_plugin(
        {
            "check_input": True,
            "llm_audit": {
                "enable": True,
                "mode": "combined_post",
                "provider_id": "audit-provider",
            },
        }
    )
    event = DummyEvent(sender_id="10003")
    request = DummyRequest(prompt="need audit")

    run(plugin.on_llm_request_hook(event, request))

    assert plugin.context.llm_calls == []


def test_default_request_hook_injects_prevention_prompt():
    plugin = build_plugin(
        {
            "keywords": {
                "enable": True,
                "plain_keywords": ["risk"],
            },
            "prevention_prompt": "BLOCK {keywords}",
        }
    )
    event = DummyEvent(sender_id="10003")
    request = DummyRequest(system_prompt="BASE")

    run(plugin.on_llm_request_hook(event, request))

    assert request.system_prompt == "BASE\n\nBLOCK risk"


def test_disabled_request_hook_skips_prevention_prompt_injection():
    plugin = build_plugin(
        {
            "inject_prevention_prompt": False,
            "keywords": {
                "enable": True,
                "plain_keywords": ["risk"],
            },
            "prevention_prompt": "BLOCK {keywords}",
        }
    )
    event = DummyEvent(sender_id="10003")
    request = DummyRequest(system_prompt="BASE")

    run(plugin.on_llm_request_hook(event, request))

    assert request.system_prompt == "BASE"


def test_original_input_only_ignores_prompt_injection_for_input_check():
    plugin = build_plugin(
        {
            "check_input": True,
            "check_input_original_only": True,
            "keywords": {
                "enable": True,
                "plain_keywords": ["喵"],
            },
        }
    )
    event = DummyEvent(sender_id="10003")
    event.message = "居"
    request = DummyRequest(prompt="<RAG-Faiss-Memory>喵</RAG-Faiss-Memory>\n居")

    run(plugin.on_llm_request_hook(event, request))

    assert event.stopped is False
    assert event.result is None
    assert event.get_extra("_csg_user_text") == "居"


def test_default_input_check_still_uses_request_prompt():
    plugin = build_plugin(
        {
            "check_input": True,
            "keywords": {
                "enable": True,
                "plain_keywords": ["喵"],
            },
        }
    )
    event = DummyEvent(sender_id="10003")
    event.message = "居"
    request = DummyRequest(prompt="<RAG-Faiss-Memory>喵</RAG-Faiss-Memory>\n居")

    run(plugin.on_llm_request_hook(event, request))

    assert event.stopped is True
    assert event.result == plugin.input_block_message


def test_combined_post_does_not_block_user_branch_when_check_input_disabled():
    plugin = build_plugin(
        {
            "check_input": False,
            "llm_audit": {
                "enable": True,
                "mode": "combined_post",
                "provider_id": "audit-provider",
            },
        }
    )

    async def fake_llm_generate(**kwargs):
        plugin.context.llm_calls.append(kwargs)
        return SimpleNamespace(
            completion_text='{"user_safe": false, "ai_safe": true, "reason": "user risk"}'
        )

    plugin.context.llm_generate = fake_llm_generate
    event = DummyEvent(sender_id="10004")
    event.set_extra("_csg_user_text", "need audit")
    response = SimpleNamespace(completion_text="normal reply", is_chunk=False)

    run(plugin.on_llm_response_hook(event, response))

    assert plugin.context.llm_calls
    assert response.completion_text == "normal reply"
    assert event.result is None
    assert event.stopped is False
    assert plugin._violations == {}


def test_combined_post_prefers_ai_branch_when_input_blocking_disabled():
    plugin = build_plugin(
        {
            "check_input": False,
            "llm_audit": {
                "enable": True,
                "mode": "combined_post",
                "provider_id": "audit-provider",
            },
        }
    )

    async def fake_llm_generate(**kwargs):
        plugin.context.llm_calls.append(kwargs)
        return SimpleNamespace(
            completion_text='{"user_safe": false, "ai_safe": false, "reason": "both risk"}'
        )

    plugin.context.llm_generate = fake_llm_generate

    target, reason = run(plugin._check_llm_audit_combined("user text", "ai text"))

    assert target == "ai"
    assert "AI回复" in reason


def test_combined_audit_uses_shared_system_prompt_constant():
    plugin = build_plugin(
        {
            "llm_audit": {
                "enable": True,
                "mode": "combined_post",
                "provider_id": "audit-provider",
            }
        }
    )

    run(plugin._check_llm_audit_combined("user text", "ai text"))

    assert plugin.context.llm_calls
    assert plugin.context.llm_calls[0]["system_prompt"] == AUDIT_SYSTEM_PROMPT


def test_combined_audit_uses_configurable_prompt_template():
    plugin = build_plugin(
        {
            "llm_audit": {
                "enable": True,
                "mode": "combined_post",
                "provider_id": "audit-provider",
                "combined_prompt": "KW={keywords}|U={user_text}|A={ai_text}",
            }
        }
    )

    run(plugin._check_llm_audit_combined("user text", "ai text"))

    assert plugin.context.llm_calls
    assert plugin.context.llm_calls[0]["prompt"] == "KW=无指定|U=user text|A=ai text"


def run_all_tests() -> None:
    test_functions = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in sorted(test_functions, key=lambda func: func.__name__):
        test()
    print(f"{len(test_functions)} tests passed")


if __name__ == "__main__":
    run_all_tests()
