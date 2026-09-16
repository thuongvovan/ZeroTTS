from __future__ import annotations

import httpx
import numpy as np
import pytest

from zerotts_api.app import Settings, UnifiedRuntime, create_app
from zerotts_api.engines import AudioFrame, MisakiG2P


class FakeG2P:
    @staticmethod
    def backend_for(language):
        return "misaki" if language in {"en-us", "en-gb", "ja", "zh"} else "espeak-fallback"


class FakeEngine:
    def __init__(self, engine_id, voices, sample_rate, languages):
        self.id = engine_id
        self.voices = voices
        self.sample_rate = sample_rate
        self.languages = languages
        self.provider = "CUDAExecutionProvider"
        self.ready = True
        self.g2p = FakeG2P()

    def resolve_voice(self, requested, aliases):
        if requested.startswith(("vf_", "vm_", "vi_")):
            requested = requested[3:]
        requested = requested.removeprefix("kokoro:")
        requested = aliases.get(requested, requested)
        return requested if requested in self.voices else None

    def supports_language(self, language):
        return language in self.languages

    async def stream(self, text, voice, language, speed):
        assert text and voice in self.voices and language in self.languages
        samples = round(self.sample_rate * 0.1)
        yield AudioFrame(np.full(samples, 0.1, dtype=np.float32), self.sample_rate)


def make_app(
    api_key: str = "",
    espeak_fallback: bool = True,
    voice_aliases: str = "{}",
):
    settings = Settings(
        api_key=api_key,
        normalize_vietnamese=False,
        kokoro_espeak_fallback=espeak_fallback,
        voice_aliases=voice_aliases,
    )
    zero = FakeEngine("zerotts", ["maichi", "giahuy"], 48_000, ("vi",))
    kokoro = FakeEngine(
        "kokoro",
        ["af_heart", "bf_emma", "ef_dora", "jf_alpha"],
        24_000,
        ("en-us", "en-gb", "es", "fr", "hi", "it", "ja", "pt-br", "zh"),
    )
    runtime = UnifiedRuntime(settings, zero_engine=zero, kokoro_engine=kokoro)
    return create_app(settings, runtime, load_model=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def post(app, path, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, **kwargs)


@pytest.mark.anyio
async def test_gendered_vi_voice_routes_to_zerotts_and_resamples_pcm_without_model():
    response = await post(make_app(), "/v1/audio/speech", json={
            "input": "Xin chào.",
            "voice": "vf_maichi",
            "response_format": "pcm",
            "stream": False,
        })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-tts-engine"] == "zerotts"
    assert response.headers["x-tts-voice"] == "vf_maichi"
    assert response.headers["x-tts-language"] == "vi"
    assert len(response.content) == 2_400 * 2


@pytest.mark.anyio
async def test_kokoro_voice_routes_by_prefix_and_model_is_ignored():
    response = await post(make_app(), "/v1/audio/speech", json={
            "model": "anything-the-client-sends",
            "input": "Hello from Kokoro.",
            "voice": "af_heart",
            "response_format": "pcm",
            "stream": False,
        })

    assert response.status_code == 200
    assert response.headers["x-tts-engine"] == "kokoro"
    assert response.headers["x-tts-g2p"] == "misaki"
    assert response.headers["x-tts-voice"] == "af_heart"
    assert response.headers["x-tts-language"] == "en-us"
    assert len(response.content) == 2_400 * 2


@pytest.mark.anyio
async def test_voice_prefix_can_select_kokoro():
    response = await post(make_app(), "/v1/audio/speech", json={
            "model": "auto",
            "input": "Hello.",
            "voice": "bf_emma",
            "response_format": "pcm",
        })

    assert response.status_code == 200
    assert response.headers["x-tts-engine"] == "kokoro"
    assert response.headers["x-tts-language"] == "en-gb"


@pytest.mark.anyio
async def test_sse_stream_format_emits_audio_events():
    response = await post(make_app(), "/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Hello.",
            "voice": "af_heart",
            "response_format": "pcm",
            "stream_format": "sse",
        })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: speech.audio.delta" in response.text
    assert "event: speech.audio.done" in response.text


@pytest.mark.anyio
async def test_auth_uses_openai_error_shape():
    response = await post(make_app("secret"), "/v1/audio/speech", json={
            "model": "tts-1",
            "input": "Xin chào.",
            "voice": "vf_maichi",
        })

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.anyio
async def test_rejects_unknown_voice():
    response = await post(make_app(), "/v1/audio/speech", json={
            "model": "zerotts",
            "input": "Xin chào.",
            "voice": "does-not-exist",
            "response_format": "wav",
        })

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "voice"


@pytest.mark.anyio
async def test_language_extension_cannot_override_voice_routing():
    response = await post(make_app(), "/v1/audio/speech", json={
            "model": "zerotts",
            "input": "Hello.",
            "voice": "af_heart",
            "lang_code": "vi",
        })

    assert response.status_code == 200
    assert response.headers["x-tts-engine"] == "kokoro"
    assert response.headers["x-tts-language"] == "en-us"


@pytest.mark.anyio
async def test_alias_must_resolve_to_a_canonical_voice():
    app = make_app(voice_aliases='{"alloy":"vf_maichi"}')
    response = await post(app, "/v1/audio/speech", json={
            "model": "tts-1",
            "input": "Xin chào.",
            "voice": "alloy",
            "response_format": "pcm",
        })

    assert response.status_code == 200
    assert response.headers["x-tts-engine"] == "zerotts"
    assert response.headers["x-tts-voice"] == "vf_maichi"


@pytest.mark.anyio
async def test_voices_endpoint_advertises_canonical_voice_ids():
    transport = httpx.ASGITransport(app=make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/audio/voices")

    voices = {voice["id"]: voice for voice in response.json()["data"]}
    ids = set(voices)
    assert "vf_maichi" in ids
    assert "vm_giahuy" in ids
    assert "vi_maichi" not in ids
    assert "maichi" not in ids
    assert "af_heart" in ids
    assert voices["vf_maichi"]["gender"] == "female"
    assert voices["vm_giahuy"]["gender"] == "male"
    assert voices["af_heart"]["gender"] == "female"


@pytest.mark.anyio
async def test_legacy_vi_voice_is_normalized_to_gendered_voice():
    response = await post(make_app(), "/v1/audio/speech", json={
            "input": "Xin chào.",
            "voice": "vi_maichi",
            "response_format": "pcm",
        })

    assert response.status_code == 200
    assert response.headers["x-tts-voice"] == "vf_maichi"


@pytest.mark.anyio
async def test_rejects_wrong_zerotts_gender_prefix():
    response = await post(make_app(), "/v1/audio/speech", json={
            "input": "Xin chào.",
            "voice": "vm_maichi",
        })

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "voice"
    assert "vf_maichi" in response.json()["error"]["message"]


def test_espeak_is_only_used_for_languages_without_native_misaki():
    enabled = MisakiG2P(allow_espeak_fallback=True)
    disabled = MisakiG2P(allow_espeak_fallback=False)

    assert enabled.backend_for("en-us") == "misaki"
    assert enabled.backend_for("ja") == "misaki"
    assert enabled.backend_for("es") == "espeak-fallback"
    assert disabled.backend_for("es") == "unavailable"
