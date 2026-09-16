"""Inference engines and language routing for the unified TTS API."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from zerotts import ZeroTTS, normalize_vi_text
from zerotts.chunking import chunk_text, clean_segment_punctuation, normalize_punctuation

LOGGER = logging.getLogger("zerotts.api.engines")

KOKORO_LANGUAGE_CODES = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt-br": "p",
    "zh": "z",
}

LANGUAGE_ALIASES = {
    "vi-vn": "vi",
    "vn": "vi",
    "en": "en-us",
    "en_us": "en-us",
    "en-gb": "en-gb",
    "en_gb": "en-gb",
    "es-es": "es",
    "fr-fr": "fr",
    "hi-in": "hi",
    "it-it": "it",
    "jp": "ja",
    "ja-jp": "ja",
    "pt": "pt-br",
    "pt_br": "pt-br",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "cmn": "zh",
}


def normalize_language(language: str) -> str:
    value = language.strip().lower().replace("_", "-")
    return LANGUAGE_ALIASES.get(value, value)


def provider_config(provider: str, device_id: int) -> tuple[str, list[Any]]:
    """Resolve one ONNX Runtime provider and fail fast on bad CUDA setups."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    requested = provider.lower()
    if requested == "auto":
        selected = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available
            else "CPUExecutionProvider"
        )
    elif requested == "cuda":
        selected = "CUDAExecutionProvider"
    elif requested == "cpu":
        selected = "CPUExecutionProvider"
    else:
        raise RuntimeError("ZEROTTS_PROVIDER must be one of: auto, cuda, cpu")
    if selected not in available:
        raise RuntimeError(
            f"{selected} is unavailable; ONNX Runtime providers: {', '.join(available)}"
        )
    providers: list[Any] = [selected]
    if selected == "CUDAExecutionProvider":
        providers = [(selected, {"device_id": device_id})]
    return selected, providers


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SynthesisPlan:
    engine: Any
    engine_id: str
    voice: str
    public_voice: str
    language: str
    g2p_backend: str


def _respeed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0 or not audio.size:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    from scipy.signal import resample_poly

    ratio = Fraction(1.0 / speed).limit_denominator(1000)
    return resample_poly(
        np.asarray(audio).reshape(-1), ratio.numerator, ratio.denominator
    ).astype(np.float32)


def _next_or_none(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _iterate_sync(iterator) -> AsyncIterator[np.ndarray]:
    """Pull a blocking inference generator without blocking the event loop."""
    try:
        while True:
            has_item, item = await asyncio.to_thread(_next_or_none, iterator)
            if not has_item:
                break
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


class ZeroTTSEngine:
    id = "zerotts"
    sample_rate = 48_000
    languages = ("vi",)

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.tts: ZeroTTS | None = None
        self.provider = ""
        self.voices: list[str] = []

    @property
    def ready(self) -> bool:
        return self.tts is not None

    def load(self) -> None:
        selected, providers = provider_config(
            self.settings.provider, self.settings.device_id
        )
        LOGGER.info("Loading ZeroTTS %s with %s", self.settings.model, selected)
        self.tts = ZeroTTS.from_pretrained(
            self.settings.model,
            providers=providers,
            intra_op_num_threads=self.settings.intra_op_threads,
            local_files_only=self.settings.local_files_only,
        )
        if self.settings.voices_dir:
            self.tts.add_voices(self.settings.voices_dir)
        self.provider = selected
        self.voices = self.tts.list_voices()
        if not self.voices:
            raise RuntimeError("The loaded ZeroTTS model does not contain any voices")
        LOGGER.info("ZeroTTS ready with %d voices", len(self.voices))

    def resolve_voice(self, requested: str, aliases: dict[str, str]) -> str | None:
        value = requested
        if value.startswith(("vf_", "vm_", "vi_")):
            value = value[3:]
        if value in self.voices:
            return value
        aliased = aliases.get(requested)
        if aliased in self.voices:
            return aliased
        return None

    async def stream(
        self, text: str, voice: str, language: str, speed: float
    ) -> AsyncIterator[AudioFrame]:
        if self.tts is None:
            raise RuntimeError("ZeroTTS is not loaded")
        text = normalize_punctuation(text.strip())
        if self.settings.normalize_vietnamese:
            text = normalize_vi_text(text)
        segments = [
            cleaned
            for segment in chunk_text(text, self.settings.max_chunk_seconds)
            if (cleaned := clean_segment_punctuation(segment))
        ]
        if not segments:
            raise ValueError("Input must contain speakable text")

        for index, segment in enumerate(segments):
            iterator = self.tts.synthesize_stream(segment, voice=voice)
            async for chunk in _iterate_sync(iterator):
                yield AudioFrame(_respeed(np.asarray(chunk), speed), self.tts.sample_rate)
            if index < len(segments) - 1 and self.settings.segment_silence_seconds > 0:
                size = round(self.tts.sample_rate * self.settings.segment_silence_seconds)
                yield AudioFrame(np.zeros(size, dtype=np.float32), self.tts.sample_rate)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", url, destination)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=60) as source, open(temporary, "wb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MisakiG2P:
    """Misaki-first G2P registry; eSpeak is an explicit last resort."""

    NATIVE_LANGUAGES = frozenset({"en-us", "en-gb", "ja", "zh"})
    ESPEAK_LANGUAGES = {
        "es": "es",
        "fr": "fr-fr",
        "hi": "hi",
        "it": "it",
        "pt-br": "pt-br",
    }

    def __init__(self, allow_espeak_fallback: bool) -> None:
        self.allow_espeak_fallback = allow_espeak_fallback
        self._instances: dict[str, Any] = {}

    def backend_for(self, language: str) -> str:
        if language in self.NATIVE_LANGUAGES:
            return "misaki"
        if language in self.ESPEAK_LANGUAGES and self.allow_espeak_fallback:
            return "espeak-fallback"
        return "unavailable"

    def _build(self, language: str):
        if language in {"en-us", "en-gb"}:
            from misaki import en, espeak

            british = language == "en-gb"
            fallback = (
                espeak.EspeakFallback(british=british)
                if self.allow_espeak_fallback
                else None
            )
            return en.G2P(trf=False, british=british, fallback=fallback)
        if language == "ja":
            from misaki import ja

            return ja.JAG2P()
        if language == "zh":
            from misaki import zh

            return zh.ZHG2P(version="1.0")
        if language in self.ESPEAK_LANGUAGES and self.allow_espeak_fallback:
            from misaki.espeak import EspeakG2P

            return EspeakG2P(language=self.ESPEAK_LANGUAGES[language])
        raise ValueError(
            f"No Misaki frontend is available for language '{language}', "
            "and eSpeak fallback is disabled"
        )

    def phonemize(self, text: str, language: str) -> str:
        g2p = self._instances.get(language)
        if g2p is None:
            g2p = self._build(language)
            self._instances[language] = g2p
        result = g2p(text)
        phonemes = result[0] if isinstance(result, tuple) else result
        if not isinstance(phonemes, str) or not phonemes.strip():
            raise ValueError(f"Misaki produced no phonemes for language '{language}'")
        return phonemes


class KokoroEngine:
    id = "kokoro"
    sample_rate = 24_000
    languages = tuple(KOKORO_LANGUAGE_CODES)

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.kokoro: Any | None = None
        self.provider = ""
        self.voices: list[str] = []
        self.g2p = MisakiG2P(settings.kokoro_espeak_fallback)

    @property
    def ready(self) -> bool:
        return self.kokoro is not None

    def load(self) -> None:
        model_path = Path(self.settings.kokoro_model_path)
        voices_path = Path(self.settings.kokoro_voices_path)
        if self.settings.kokoro_download:
            if not model_path.is_file():
                _download(self.settings.kokoro_model_url, model_path)
            if not voices_path.is_file():
                _download(self.settings.kokoro_voices_url, voices_path)
        missing = [str(path) for path in (model_path, voices_path) if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Missing Kokoro asset(s): " + ", ".join(missing)
                + ". Enable KOKORO_DOWNLOAD or mount the files."
            )

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        selected, providers = provider_config(
            self.settings.provider, self.settings.device_id
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.settings.intra_op_threads
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        vocab_config = self.settings.kokoro_vocab_config or None
        self.kokoro = Kokoro.from_session(
            session, str(voices_path), vocab_config=vocab_config
        )
        self.provider = selected
        self.voices = self.kokoro.get_voices()
        if not self.voices:
            raise RuntimeError("The Kokoro voice bundle does not contain any voices")
        LOGGER.info("Kokoro ONNX ready with %d voices using %s", len(self.voices), selected)

    def supports_language(self, language: str) -> bool:
        return (
            language in KOKORO_LANGUAGE_CODES
            and self.g2p.backend_for(language) != "unavailable"
        )

    def resolve_voice(self, requested: str, aliases: dict[str, str]) -> str | None:
        value = requested.removeprefix("kokoro:")
        if value in self.voices:
            return value
        aliased = aliases.get(requested)
        if aliased in self.voices:
            return aliased
        return None

    async def stream(
        self, text: str, voice: str, language: str, speed: float
    ) -> AsyncIterator[AudioFrame]:
        if self.kokoro is None:
            raise RuntimeError("Kokoro is not loaded")
        # The graph supports 0.5–2.0. Preserve the wider OpenAI range with a
        # small post-resample only outside that native interval.
        model_speed = min(2.0, max(0.5, speed))
        post_speed = speed / model_speed
        phonemes = await asyncio.to_thread(self.g2p.phonemize, text.strip(), language)
        async for samples, sample_rate in self.kokoro.create_stream(
            phonemes,
            voice=voice,
            speed=model_speed,
            lang=KOKORO_LANGUAGE_CODES[language],
            is_phonemes=True,
        ):
            yield AudioFrame(_respeed(np.asarray(samples), post_speed), sample_rate)
