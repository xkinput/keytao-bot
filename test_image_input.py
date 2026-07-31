"""Tests for bounded image ingestion and vision proxy serialization."""

import asyncio
import base64
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI
from PIL import Image

_MODULE_PATH = Path(__file__).parent / "keytao_bot" / "utils" / "image_input.py"
_SPEC = importlib.util.spec_from_file_location("keytao_image_input_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
image_input = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = image_input
_SPEC.loader.exec_module(image_input)

ImageAttachment = image_input.ImageAttachment
ImageInputError = image_input.ImageInputError
VisionConfigurationError = image_input.VisionConfigurationError
VisionRuntimeConfig = image_input.VisionRuntimeConfig
VisionServiceError = image_input.VisionServiceError
deduplicate_image_attachments = image_input.deduplicate_image_attachments
detect_image_media_type = image_input.detect_image_media_type
extract_image_attachments = image_input.extract_image_attachments
prepare_image_attachments = image_input.prepare_image_attachments
request_vision_description = image_input.request_vision_description


def _make_png(width=16, height=16):
    output = BytesIO()
    Image.new("RGB", (width, height), color=(32, 64, 96)).save(output, format="PNG")
    return output.getvalue()


_PNG_BYTES = _make_png()


def _config(**overrides):
    values = {
        "enabled": True,
        "api_key": "vision-test-key",
        "base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-flash",
        "timeout": 10.0,
        "max_tokens": 500,
        "max_images": 3,
        "max_image_bytes": 1024,
        "max_total_image_bytes": 2048,
        "max_image_pixels": 1024,
        "max_total_image_pixels": 2048,
    }
    values.update(overrides)
    return VisionRuntimeConfig(**values)


class ImageExtractionTests(unittest.TestCase):
    def test_extracts_qq_and_telegram_segments_without_exposing_locators(self):
        qq_url = "https://multimedia.nt.qq.com.cn/download?secret=one"
        qq = extract_image_attachments([
            {"type": "text", "data": {"text": "look"}},
            {
                "type": "image",
                "data": {
                    "file": "qq-file-id",
                    "url": qq_url,
                    "file_size": "123",
                    "summary": "screen",
                },
            },
        ], "qq")
        telegram = extract_image_attachments([
            {"type": "photo", "data": {"file": "telegram-file-id"}},
        ], "telegram", source="reply")

        self.assertEqual(len(qq), 1)
        self.assertEqual(qq[0].file_size, 123)
        self.assertEqual(telegram[0].source, "reply")
        self.assertNotIn(qq_url, repr(qq[0]))
        self.assertNotIn("qq-file-id", repr(qq[0]))
        self.assertNotIn("telegram-file-id", repr(telegram[0]))

    def test_deduplicates_without_reordering(self):
        first = ImageAttachment("qq", "one", url="https://a.qpic.cn/one")
        duplicate = ImageAttachment("qq", "one", url="https://a.qpic.cn/one")
        second = ImageAttachment("qq", "two", url="https://a.qpic.cn/two")

        self.assertEqual(
            deduplicate_image_attachments((first, duplicate, second)),
            (first, second),
        )

    def test_segment_scan_has_a_hard_streaming_limit(self):
        seen = 0

        def endless_segments():
            nonlocal seen
            while True:
                seen += 1
                yield {"type": "text", "data": {"text": "noise"}}

        self.assertEqual(extract_image_attachments(endless_segments(), "qq"), ())
        self.assertEqual(seen, image_input._MAX_SCANNED_SEGMENTS)

    def test_accepts_only_supported_magic_signatures(self):
        self.assertEqual(detect_image_media_type(_PNG_BYTES), "image/png")
        self.assertEqual(
            detect_image_media_type(b"RIFF\x00\x00\x00\x00WEBPpayload"),
            "image/webp",
        )
        with self.assertRaises(ImageInputError):
            detect_image_media_type(b"not-an-image")


class VisionConfigurationTests(unittest.TestCase):
    def test_requires_explicit_independent_provider(self):
        for config in (
            _config(enabled=False),
            _config(api_key=""),
            _config(model="deepseek-v4-flash"),
            _config(base_url="https://api.deepseek.com", model="some-model"),
        ):
            with self.subTest(config=config):
                with self.assertRaises(VisionConfigurationError):
                    config.validate()

    def test_valid_provider_configuration_passes(self):
        _config().validate()

    def test_xiaomi_endpoint_requires_current_image_model(self):
        _config(
            base_url="https://api.xiaomimimo.com/v1",
            model="mimo-v2.5",
        ).validate()
        for model in ("mimo-v2-omni", "mimo-v2-flash", "qwen3.7-flash"):
            with self.subTest(model=model):
                with self.assertRaises(VisionConfigurationError):
                    _config(
                        base_url="https://api.xiaomimimo.com/v1",
                        model=model,
                    ).validate()

    def test_mimo_model_requires_exact_official_endpoint(self):
        for base_url in (
            "https://vision.example/v1",
            "https://proxy.xiaomimimo.com/v1",
            "https://api.xiaomimimo.com/v1/chat/completions",
            "https://api.xiaomimimo.com/v1?target=other",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(VisionConfigurationError):
                    _config(
                        base_url=base_url,
                        model="mimo-v2.5",
                    ).validate()

    def test_requires_https_provider_endpoint(self):
        with self.assertRaises(VisionConfigurationError):
            _config(base_url="http://vision.example/v1").validate()

    def test_qq_url_requires_scheme_matched_standard_port(self):
        self.assertTrue(image_input._qq_image_url_allowed(
            "https://multimedia.nt.qq.com.cn/image.png"
        ))
        self.assertTrue(image_input._qq_image_url_allowed(
            "http://multimedia.nt.qq.com.cn:80/image.png"
        ))
        self.assertFalse(image_input._qq_image_url_allowed(
            "https://multimedia.nt.qq.com.cn:80/image.png"
        ))
        self.assertFalse(image_input._qq_image_url_allowed(
            "http://multimedia.nt.qq.com.cn:443/image.png"
        ))


class ImagePreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_multicast_dns_answers(self):
        address_info = [(
            image_input.socket.AF_INET,
            image_input.socket.SOCK_STREAM,
            6,
            "",
            ("224.0.0.1", 443),
        )]
        with patch.object(
            image_input.asyncio,
            "to_thread",
            new=AsyncMock(return_value=address_info),
        ):
            with self.assertRaises(ImageInputError):
                await image_input._resolve_public_url_destinations(
                    "https://multimedia.nt.qq.com.cn/image.png"
                )

    async def test_prepares_base64_image_and_applies_count_limit(self):
        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
        attachments = (
            ImageAttachment("qq", f"base64://{encoded}"),
            ImageAttachment("qq", f"base64://{encoded}", source="reply"),
        )

        result = await prepare_image_attachments(
            SimpleNamespace(),
            attachments,
            _config(max_images=1),
        )

        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].media_type, "image/png")
        self.assertEqual(result.requested_count, 2)
        self.assertEqual(len(result.warnings), 1)

    async def test_declared_oversized_image_is_rejected_without_fetch(self):
        bot = SimpleNamespace(call_api=AsyncMock())
        attachment = ImageAttachment(
            "qq",
            "qq-file-id",
            file_size=2048,
        )

        with self.assertRaises(ImageInputError):
            await prepare_image_attachments(
                bot,
                (attachment,),
                _config(max_image_bytes=1024),
            )

        bot.call_api.assert_not_awaited()

    async def test_qq_redirect_to_untrusted_host_is_rejected(self):
        real_async_client = httpx.AsyncClient

        async def handle_request(request):
            return httpx.Response(
                302,
                headers={"location": "https://attacker.example/image.png"},
            )

        def client_factory(**kwargs):
            return real_async_client(
                transport=httpx.MockTransport(handle_request),
                follow_redirects=False,
            )

        with (
            patch.object(image_input.httpx, "AsyncClient", side_effect=client_factory),
            patch.object(
                image_input,
                "_resolve_public_url_destinations",
                new=AsyncMock(return_value=("1.1.1.1",)),
            ),
        ):
            with self.assertRaises(ImageInputError):
                await image_input._download_bounded_url(
                    "https://multimedia.nt.qq.com.cn/download?id=one",
                    max_bytes=1024,
                    timeout=10.0,
                    allowed_url=image_input._qq_image_url_allowed,
                )

    async def test_rejects_compressed_http_response_before_decoding(self):
        async def handle_request(request):
            return httpx.Response(
                200,
                headers={"content-encoding": "gzip"},
                content=b"compressed-body-not-read",
            )

        def client_factory(**kwargs):
            return httpx.AsyncClient(
                transport=httpx.MockTransport(handle_request),
                follow_redirects=False,
            )

        with (
            patch.object(image_input.httpx, "AsyncClient", side_effect=client_factory),
            patch.object(
                image_input,
                "_resolve_public_url_destinations",
                new=AsyncMock(return_value=("1.1.1.1",)),
            ),
        ):
            with self.assertRaises(ImageInputError):
                await image_input._download_bounded_url(
                    "https://multimedia.nt.qq.com.cn/download?id=one",
                    max_bytes=1024,
                    timeout=10.0,
                    allowed_url=image_input._qq_image_url_allowed,
                )

    async def test_download_pins_verified_ip_and_preserves_host_and_sni(self):
        captured = {}

        async def handle_request(request):
            captured["url"] = str(request.url)
            captured["host"] = request.headers.get("host")
            captured["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, content=_PNG_BYTES)

        real_async_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return real_async_client(
                transport=httpx.MockTransport(handle_request),
                follow_redirects=False,
            )

        with (
            patch.object(image_input.httpx, "AsyncClient", side_effect=client_factory),
            patch.object(
                image_input,
                "_resolve_public_url_destinations",
                new=AsyncMock(return_value=("1.1.1.1",)),
            ),
        ):
            data = await image_input._download_bounded_url(
                "https://multimedia.nt.qq.com.cn/download?id=one",
                max_bytes=1024,
                timeout=10.0,
                allowed_url=image_input._qq_image_url_allowed,
            )

        self.assertEqual(data, _PNG_BYTES)
        self.assertEqual(captured["url"], "https://1.1.1.1/download?id=one")
        self.assertEqual(captured["host"], "multimedia.nt.qq.com.cn")
        self.assertEqual(captured["sni"], "multimedia.nt.qq.com.cn")

    async def test_reads_napcat_local_file_only_through_mapped_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            mapped_root = Path(temporary_directory) / "qq"
            image_path = mapped_root / "cache" / "image.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(_PNG_BYTES)
            config = _config(
                qq_napcat_source_root="/app/.config/QQ",
                qq_napcat_mapped_root=str(mapped_root),
            )
            bot = SimpleNamespace(call_api=AsyncMock(return_value={
                "file": "/app/.config/QQ/cache/image.png",
                "url": "/app/.config/QQ/cache/image.png",
                "file_size": len(_PNG_BYTES),
            }))

            result = await prepare_image_attachments(
                bot,
                (ImageAttachment("qq", "qq-file-id"),),
                config,
            )

            self.assertEqual(result.images[0].data, _PNG_BYTES)
            bot.call_api.assert_awaited_once_with("get_image", file="qq-file-id")

    async def test_falls_back_to_qq_cdn_when_napcat_cache_disappears(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(
                qq_napcat_source_root="/app/.config/QQ",
                qq_napcat_mapped_root=temporary_directory,
            )
            cdn_url = "https://multimedia.nt.qq.com.cn/download?id=image"
            bot = SimpleNamespace(call_api=AsyncMock(return_value={
                "file": "/app/.config/QQ/cache/missing.png",
                "url": cdn_url,
            }))

            with patch.object(
                image_input,
                "_download_bounded_url",
                new=AsyncMock(return_value=_PNG_BYTES),
            ) as download:
                result = await prepare_image_attachments(
                    bot,
                    (ImageAttachment("qq", "qq-file-id"),),
                    config,
                )

            self.assertEqual(result.images[0].data, _PNG_BYTES)
            download.assert_awaited_once_with(
                cdn_url,
                max_bytes=config.max_image_bytes,
                timeout=config.timeout,
                allowed_url=image_input._qq_image_url_allowed,
            )

    async def test_rejects_napcat_path_escape_symlink_and_oversized_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            mapped_root = temporary_root / "qq"
            mapped_root.mkdir()
            outside_image = temporary_root / "outside.png"
            outside_image.write_bytes(_PNG_BYTES)
            (mapped_root / "escape.png").symlink_to(outside_image)
            oversized_image = mapped_root / "oversized.png"
            oversized_image.write_bytes(_PNG_BYTES + b"x" * 2048)
            config = _config(
                max_image_bytes=1024,
                qq_napcat_source_root="/app/.config/QQ",
                qq_napcat_mapped_root=str(mapped_root),
            )

            self.assertIsNone(image_input._map_napcat_image_path(
                "/app/.config/QQ/../outside.png",
                config,
            ))
            self.assertIsNone(image_input._map_napcat_image_path(
                "/app/.config/QQ/escape.png",
                config,
            ))
            bot = SimpleNamespace(call_api=AsyncMock(return_value={
                "file": "/app/.config/QQ/oversized.png",
            }))
            with self.assertRaises(ImageInputError):
                await prepare_image_attachments(
                    bot,
                    (ImageAttachment("qq", "qq-file-id"),),
                    config,
                )

    def test_napcat_openat_rejects_intermediate_symlink_swap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            mapped_root = temporary_root / "qq"
            cache_path = mapped_root / "cache"
            cache_path.mkdir(parents=True)
            (cache_path / "image.png").write_bytes(_PNG_BYTES)
            outside_path = temporary_root / "outside"
            outside_path.mkdir()
            (outside_path / "image.png").write_bytes(b"outside-secret")
            config = _config(
                qq_napcat_source_root="/app/.config/QQ",
                qq_napcat_mapped_root=str(mapped_root),
            )

            mapped = image_input._map_napcat_image_path(
                "/app/.config/QQ/cache/image.png",
                config,
            )
            self.assertIsNotNone(mapped)
            cache_path.rename(mapped_root / "old-cache")
            cache_path.symlink_to(outside_path, target_is_directory=True)

            with self.assertRaises(ImageInputError):
                image_input._read_bounded_regular_file_at(
                    mapped[0],
                    mapped[1],
                    1024,
                )

    async def test_rejects_high_pixel_image_before_provider_request(self):
        encoded = base64.b64encode(_make_png(64, 64)).decode("ascii")
        with self.assertRaises(ImageInputError):
            await prepare_image_attachments(
                SimpleNamespace(),
                (ImageAttachment("qq", f"base64://{encoded}"),),
                _config(max_image_pixels=1024),
            )

    async def test_telegram_public_download_uses_pinned_dns_policy(self):
        bot = SimpleNamespace(
            get_file=AsyncMock(return_value=SimpleNamespace(
                file_size=len(_PNG_BYTES),
                file_path="photos/image.png",
            )),
            bot_config=SimpleNamespace(
                api_server="https://api.telegram.org/",
                token="test-token",
            ),
        )

        with patch.object(
            image_input,
            "_download_bounded_url",
            new=AsyncMock(return_value=_PNG_BYTES),
        ) as download:
            data = await image_input._resolve_telegram_image_data(
                bot,
                ImageAttachment("telegram", "telegram-file-id"),
                _config(),
                1024,
            )

        self.assertEqual(data, _PNG_BYTES)
        self.assertTrue(download.await_args.kwargs["require_public_destination"])
        allowed_url = download.await_args.kwargs["allowed_url"]
        self.assertTrue(allowed_url(
            "https://api.telegram.org/file/bottest-token/photos/image.png"
        ))
        self.assertFalse(allowed_url("https://api.telegram.org:8443/file"))
        self.assertFalse(allowed_url("https://localhost/file"))

    async def test_telegram_loopback_server_stays_local_and_port_bound(self):
        bot = SimpleNamespace(
            get_file=AsyncMock(return_value=SimpleNamespace(
                file_size=len(_PNG_BYTES),
                file_path="photos/image.png",
            )),
            bot_config=SimpleNamespace(
                api_server="http://127.0.0.1:8081/",
                token="test-token",
            ),
        )

        with patch.object(
            image_input,
            "_download_bounded_url",
            new=AsyncMock(return_value=_PNG_BYTES),
        ) as download:
            await image_input._resolve_telegram_image_data(
                bot,
                ImageAttachment("telegram", "telegram-file-id"),
                _config(),
                1024,
            )

        self.assertFalse(download.await_args.kwargs["require_public_destination"])
        allowed_url = download.await_args.kwargs["allowed_url"]
        self.assertTrue(allowed_url("http://127.0.0.1:8081/file"))
        self.assertFalse(allowed_url("http://127.0.0.1:8082/file"))
        self.assertFalse(allowed_url("https://api.telegram.org/file"))

    async def test_rejects_magic_only_corrupt_image(self):
        encoded = base64.b64encode(b"\x89PNG\r\n\x1a\ncorrupt").decode("ascii")
        with self.assertRaises(ImageInputError):
            await prepare_image_attachments(
                SimpleNamespace(),
                (ImageAttachment("qq", f"base64://{encoded}"),),
                _config(),
            )


class VisionProxyRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_sdk_serializes_text_first_and_base64_image(self):
        captured = {}

        async def handle_request(request):
            captured.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={
                "id": "vision-test",
                "object": "chat.completion",
                "created": 0,
                "model": "vision-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "图 1 是一个测试界面。",
                    },
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            })

        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
        attachment = ImageAttachment(
            "qq",
            f"base64://{encoded}",
            url="https://multimedia.nt.qq.com.cn/download?secret=not-forwarded",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
        ) as http_client:
            async with AsyncOpenAI(
                api_key="vision-test-key",
                base_url="https://vision.example/v1",
                http_client=http_client,
                max_retries=0,
            ) as client:
                result = await request_vision_description(
                    client,
                    SimpleNamespace(),
                    (attachment,),
                    "这张图是什么？",
                    _config(),
                )

        content = captured["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("图 1=当前消息", content[0]["text"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1]["max_pixels"], 1024)
        self.assertFalse(captured["enable_thinking"])
        self.assertNotIn("multimedia.nt.qq.com.cn", json.dumps(captured))
        self.assertNotIn("not-forwarded", json.dumps(captured))
        self.assertEqual(result.description, "图 1 是一个测试界面。")
        self.assertEqual(result.image_count, 1)

    async def test_mimo_sdk_serializes_official_image_request_fields(self):
        captured = {}

        async def handle_request(request):
            captured.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={
                "id": "mimo-vision-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mimo-v2.5",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "图 1 是一个测试界面。",
                    },
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            })

        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
        attachment = ImageAttachment("qq", f"base64://{encoded}")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
        ) as http_client:
            async with AsyncOpenAI(
                api_key="vision-test-key",
                base_url="https://api.xiaomimimo.com/v1",
                http_client=http_client,
                max_retries=0,
            ) as client:
                result = await request_vision_description(
                    client,
                    SimpleNamespace(),
                    (attachment,),
                    "这张图是什么？",
                    _config(
                        base_url="https://api.xiaomimimo.com/v1",
                        model="mimo-v2.5",
                    ),
                )

        content = captured["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1]["type"], "text")
        self.assertNotIn("max_pixels", content[0])
        self.assertEqual(captured["max_completion_tokens"], 500)
        self.assertNotIn("max_tokens", captured)
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(result.description, "图 1 是一个测试界面。")

    async def test_incomplete_vision_response_is_rejected(self):
        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="partial"),
        )])
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response)),
        ))

        with self.assertRaises(VisionServiceError):
            await request_vision_description(
                client,
                SimpleNamespace(),
                (ImageAttachment("qq", f"base64://{encoded}"),),
                "describe",
                _config(),
            )

    async def test_provider_call_has_a_wall_clock_timeout(self):
        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")

        async def slow_create(**kwargs):
            await asyncio.sleep(1)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=slow_create),
        ))

        with self.assertRaises(VisionServiceError):
            await request_vision_description(
                client,
                SimpleNamespace(),
                (ImageAttachment("qq", f"base64://{encoded}"),),
                "describe",
                _config(timeout=0.01),
            )

    async def test_metadata_lookup_uses_the_same_wall_clock_timeout(self):
        async def slow_get_file(file_id):
            await asyncio.sleep(1)

        bot = SimpleNamespace(get_file=slow_get_file)
        with self.assertRaises(VisionServiceError):
            await request_vision_description(
                SimpleNamespace(),
                bot,
                (ImageAttachment("telegram", "telegram-file-id"),),
                "describe",
                _config(timeout=0.01),
            )

    async def test_sync_image_decode_cannot_block_the_async_deadline(self):
        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")

        def slow_validate(data, max_pixels):
            time.sleep(0.2)
            return "image/png", 256

        with patch.object(image_input, "validate_image_data", slow_validate):
            started_at = time.monotonic()
            with self.assertRaises(VisionServiceError):
                await request_vision_description(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    (ImageAttachment("qq", f"base64://{encoded}"),),
                    "describe",
                    _config(timeout=0.01),
                )
            elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
