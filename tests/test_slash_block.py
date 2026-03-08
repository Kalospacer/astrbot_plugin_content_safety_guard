from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_MAIN = ROOT / "astrbot_plugin_content_safety_guard" / "main.py"


def install_astrbot_stubs() -> None:
    if "astrbot.api.star" in sys.modules:
        return

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    star_mod = types.ModuleType("astrbot.api.star")
    event_mod = types.ModuleType("astrbot.api.event")
    provider_mod = types.ModuleType("astrbot.api.provider")

    class DummyLogger:
        def info(self, *args, **kwargs) -> None:
            pass

        def warning(self, *args, **kwargs) -> None:
            pass

        def error(self, *args, **kwargs) -> None:
            pass

    class DummyContext:
        pass

    class DummyStar:
        def __init__(self, context=None) -> None:
            self.context = context

    class DummyStarTools:
        @classmethod
        def get_data_dir(cls):
            return ROOT / "data"

    def register(**kwargs):
        def decorator(obj):
            return obj

        return decorator

    class DummyCommandGroup:
        def command(self, *args, **kwargs):
            return simple_decorator

    def simple_decorator(*args, **kwargs):
        def decorator(obj):
            return obj

        return decorator

    def command_group(*args, **kwargs):
        def decorator(obj):
            return DummyCommandGroup()

        return decorator

    class DummyEventMessageType:
        GROUP_MESSAGE = "group"

    class DummyPermissionType:
        ADMIN = "admin"

    class DummyFilterModule:
        EventMessageType = DummyEventMessageType
        PermissionType = DummyPermissionType
        event_message_type = staticmethod(simple_decorator)
        permission_type = staticmethod(simple_decorator)
        on_llm_request = staticmethod(simple_decorator)
        on_llm_response = staticmethod(simple_decorator)

    DummyFilterModule.command_group = staticmethod(command_group)

    class DummyAstrMessageEvent:
        pass

    class DummyLLMResponse:
        pass

    class DummyProviderRequest:
        pass

    api_mod.logger = DummyLogger()
    star_mod.Context = DummyContext
    star_mod.Star = DummyStar
    star_mod.StarTools = DummyStarTools
    star_mod.register = register
    event_mod.filter = DummyFilterModule()
    event_mod.AstrMessageEvent = DummyAstrMessageEvent
    provider_mod.LLMResponse = DummyLLMResponse
    provider_mod.ProviderRequest = DummyProviderRequest

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.provider"] = provider_mod


install_astrbot_stubs()

spec = importlib.util.spec_from_file_location("content_safety_guard_main", PLUGIN_MAIN)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ContentSafetyGuardPlugin = module.ContentSafetyGuardPlugin


class DummyEvent:
    def __init__(
        self,
        *,
        raw_message: str,
        trimmed_message: str,
        is_admin: bool = False,
    ) -> None:
        self._raw_message = raw_message
        self._trimmed_message = trimmed_message
        self._is_admin = is_admin
        self.stopped = False

    def is_private_chat(self) -> bool:
        return False

    def get_message_outline(self) -> str:
        return self._raw_message

    def get_message_str(self) -> str:
        return self._trimmed_message

    def get_sender_id(self) -> str:
        return "user-1"

    def get_self_id(self) -> str:
        return "bot-1"

    def is_admin(self) -> bool:
        return self._is_admin

    def get_group_id(self) -> str:
        return "group-1"

    def stop_event(self) -> None:
        self.stopped = True


def make_plugin() -> ContentSafetyGuardPlugin:
    plugin = ContentSafetyGuardPlugin.__new__(ContentSafetyGuardPlugin)
    plugin.block_non_admin_slash_in_group = True
    return plugin


class SlashBlockTests(unittest.IsolatedAsyncioTestCase):
    async def test_block_group_slash_uses_raw_message_text(self) -> None:
        plugin = make_plugin()
        event = DummyEvent(raw_message="/help", trimmed_message="help")

        await plugin.block_group_slash_for_non_admin(event)

        self.assertTrue(event.stopped)

    async def test_block_group_slash_after_leading_mention(self) -> None:
        plugin = make_plugin()
        event = DummyEvent(
            raw_message="[At:bot-1] /new",
            trimmed_message="/new",
        )

        await plugin.block_group_slash_for_non_admin(event)

        self.assertTrue(event.stopped)

    async def test_does_not_block_mid_sentence_slash(self) -> None:
        plugin = make_plugin()
        event = DummyEvent(
            raw_message="hello /new",
            trimmed_message="hello /new",
        )

        await plugin.block_group_slash_for_non_admin(event)

        self.assertFalse(event.stopped)

    async def test_block_group_slash_skips_admin(self) -> None:
        plugin = make_plugin()
        event = DummyEvent(raw_message="/help", trimmed_message="help", is_admin=True)

        await plugin.block_group_slash_for_non_admin(event)

        self.assertFalse(event.stopped)
