"""OpenAI-compatible speech API backed by ZeroTTS and Kokoro ONNX."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from zerotts.audio import resample_wav
from zerotts_api.engines import (
    KOKORO_LANGUAGE_CODES,
    AudioFrame,
    KokoroEngine,
    SynthesisPlan,
    ZeroTTSEngine,
)

LOGGER = logging.getLogger("zerotts.api")

ZEROTTS_VOICE_GENDERS = {
    "baotrang": "f",
    "giahuy": "m",
    "hamy": "f",
    "huuduc": "m",
    "kimoanh": "f",
    "maichi": "f",
    "quangminh": "m",
    "tiendat": "m",
}
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = "zeroweight-ai/ZeroTTS"
    provider: str = "auto"
    device_id: int = 0
    intra_op_threads: int = 4
    max_concurrency: int = 1
    api_key: str = ""
    voice_aliases: str = "{}"
    normalize_vietnamese: bool = True
    max_chunk_seconds: float = 15.0
    segment_silence_seconds: float = 0.15
    local_files_only: bool = False
    voices_dir: str = ""
    kokoro_enabled: bool = True
    kokoro_model_path: str = "/data/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "/data/kokoro/voices-v1.0.bin"
    kokoro_vocab_config: str = ""
    kokoro_model_url: str = KOKORO_MODEL_URL
    kokoro_voices_url: str = KOKORO_VOICES_URL
    kokoro_download: bool = True
    kokoro_espeak_fallback: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.getenv("ZEROTTS_MODEL", cls.model),
            provider=os.getenv("ZEROTTS_PROVIDER", cls.provider).lower(),
            device_id=int(os.getenv("ZEROTTS_DEVICE_ID", str(cls.device_id))),
            intra_op_threads=int(os.getenv(
                "ZEROTTS_INTRA_OP_THREADS", str(cls.intra_op_threads)
            )),
            max_concurrency=max(1, int(os.getenv(
                "ZEROTTS_MAX_CONCURRENCY", str(cls.max_concurrency)
            ))),
            api_key=os.getenv("ZEROTTS_API_KEY", ""),
            voice_aliases=os.getenv("TTS_VOICE_ALIASES", os.getenv(
                "ZEROTTS_VOICE_ALIASES", "{}"
            )),
            normalize_vietnamese=_env_bool("ZEROTTS_NORMALIZE_VI", True),
            max_chunk_seconds=float(os.getenv(
                "ZEROTTS_MAX_CHUNK_SECONDS", str(cls.max_chunk_seconds)
            )),
            segment_silence_seconds=float(os.getenv(
                "ZEROTTS_SEGMENT_SILENCE_SECONDS", str(cls.segment_silence_seconds)
            )),
            local_files_only=_env_bool("ZEROTTS_LOCAL_FILES_ONLY", False),
            voices_dir=os.getenv("ZEROTTS_VOICES_DIR", ""),
            kokoro_enabled=_env_bool("KOKORO_ENABLED", True),
            kokoro_model_path=os.getenv("KOKORO_MODEL_PATH", cls.kokoro_model_path),
            kokoro_voices_path=os.getenv("KOKORO_VOICES_PATH", cls.kokoro_voices_path),
            kokoro_vocab_config=os.getenv("KOKORO_VOCAB_CONFIG", ""),
            kokoro_model_url=os.getenv("KOKORO_MODEL_URL", KOKORO_MODEL_URL),
            kokoro_voices_url=os.getenv("KOKORO_VOICES_URL", KOKORO_VOICES_URL),
            kokoro_download=_env_bool("KOKORO_DOWNLOAD", True),
            kokoro_espeak_fallback=_env_bool("KOKORO_ESPEAK_FALLBACK", True),
        )


class VoiceReference(BaseModel):
    id: str = Field(..., min_length=1)


class SpeechRequest(BaseModel):
    # Accepted for OpenAI SDK compatibility. Voice is the only routing key.
    model: str = Field("tts-1", min_length=1)
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str | VoiceReference
    instructions: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    stream: bool = True
    stream_format: Literal["audio", "sse"] = "audio"


class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.param = param
        self.code = code
        self.error_type = error_type


def _error_response(exc: APIError) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


class UnifiedRuntime:
    def __init__(
        self,
        settings: Settings,
        zero_engine: ZeroTTSEngine | None = None,
        kokoro_engine: KokoroEngine | None = None,
    ) -> None:
        self.settings = settings
        self.zero = zero_engine or ZeroTTSEngine(settings)
        self.kokoro = (
            kokoro_engine or KokoroEngine(settings) if settings.kokoro_enabled else None
        )
        self.voice_aliases = self._parse_voice_aliases(settings.voice_aliases)
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)

    @staticmethod
    def _parse_voice_aliases(raw: str) -> dict[str, str]:
        if not raw.strip():
            return {}
        try:
            aliases = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("TTS_VOICE_ALIASES must be valid JSON") from exc
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in aliases.items()
        ):
            raise RuntimeError("TTS_VOICE_ALIASES must be a JSON object of string pairs")
        return aliases

    def load(self) -> None:
        self.zero.load()
        if self.kokoro is not None:
            self.kokoro.load()

    @property
    def ready(self) -> bool:
        return self.zero.ready and (self.kokoro is None or self.kokoro.ready)

    def status(self) -> dict:
        engines = {
            "zerotts": {
                "ready": self.zero.ready,
                "provider": self.zero.provider or None,
                "voices": len(self.zero.voices),
                "languages": list(self.zero.languages),
            }
        }
        if self.kokoro is not None:
            engines["kokoro"] = {
                "ready": self.kokoro.ready,
                "provider": self.kokoro.provider or None,
                "voices": len(self.kokoro.voices),
                "languages": list(self.kokoro.languages),
                "g2p": "misaki-first",
                "espeak_fallback": self.settings.kokoro_espeak_fallback,
            }
        return engines

    @staticmethod
    def _requested_voice(body: SpeechRequest) -> str:
        return body.voice if isinstance(body.voice, str) else body.voice.id

    def _canonical_voice(self, requested_voice: str) -> str:
        voice = requested_voice.strip()
        seen: set[str] = set()
        while voice in self.voice_aliases:
            if voice in seen:
                raise APIError(
                    f"Voice alias cycle detected at '{voice}'.",
                    param="voice",
                    code="invalid_value",
                )
            seen.add(voice)
            voice = self.voice_aliases[voice].strip()
        return voice

    @staticmethod
    def _kokoro_language(voice: str) -> str | None:
        native_voice = voice.removeprefix("kokoro:")
        if len(native_voice) < 3 or native_voice[1:3] not in {"f_", "m_"}:
            return None
        return next(
            (
                language
                for language, prefix in KOKORO_LANGUAGE_CODES.items()
                if native_voice.startswith(prefix)
            ),
            None,
        )

    @staticmethod
    def _zero_public_voice(native_voice: str) -> str:
        gender = ZEROTTS_VOICE_GENDERS.get(native_voice)
        return f"v{gender}_{native_voice}" if gender else f"vi_{native_voice}"

    def plan(self, body: SpeechRequest) -> SynthesisPlan:
        requested_voice = self._requested_voice(body)
        public_voice = self._canonical_voice(requested_voice)

        if public_voice.startswith(("vf_", "vm_", "vi_")):
            engine = self.zero
            language = "vi"
        else:
            language = self._kokoro_language(public_voice)
            engine = self.kokoro if language is not None else None

        if language is None:
            raise APIError(
                f"Voice '{requested_voice}' cannot select an engine. Use a vf_* or "
                "vm_* ZeroTTS voice, or a native Kokoro voice such as af_heart.",
                param="voice",
                code="invalid_value",
            )
        if engine is None:
            raise APIError(
                "Kokoro is disabled. Set KOKORO_ENABLED=true to use this voice.",
                503,
                param="voice",
                code="model_not_available",
                error_type="server_error",
            )
        if not engine.ready:
            raise APIError(
                f"Engine '{engine.id}' is not loaded.",
                503,
                code="model_not_available",
                error_type="server_error",
            )
        if engine is self.kokoro and not self.kokoro.supports_language(language):
            suffix = (
                " eSpeak fallback is disabled."
                if not self.settings.kokoro_espeak_fallback
                else ""
            )
            raise APIError(
                f"Kokoro language '{language}' is unavailable.{suffix}",
                param="voice",
                code="unsupported_language",
            )

        voice = engine.resolve_voice(public_voice, {})
        if voice is None:
            available = [
                self._zero_public_voice(item) if engine is self.zero else item
                for item in engine.voices[:20]
            ]
            raise APIError(
                f"Unknown {engine.id} voice '{requested_voice}'. Available voices: "
                f"{', '.join(available)}",
                param="voice",
                code="invalid_value",
            )
        if engine is self.zero:
            canonical_voice = self._zero_public_voice(voice)
            if public_voice.startswith(("vf_", "vm_")) and public_voice != canonical_voice:
                raise APIError(
                    f"Voice '{requested_voice}' has the wrong gender prefix; "
                    f"use '{canonical_voice}'.",
                    param="voice",
                    code="invalid_value",
                )
            public_voice = canonical_voice
        g2p_backend = "zerotts-native"
        if engine is self.kokoro:
            g2p_backend = self.kokoro.g2p.backend_for(language)
        return SynthesisPlan(
            engine, engine.id, voice, public_voice, language, g2p_backend
        )

    def voices_payload(self) -> list[dict]:
        data = [
            {
                "id": self._zero_public_voice(voice),
                "name": self._zero_public_voice(voice),
                "object": "audio.voice",
                "engine": "zerotts",
                "languages": ["vi"],
                "gender": {
                    "f": "female",
                    "m": "male",
                }.get(ZEROTTS_VOICE_GENDERS.get(voice), "unknown"),
            }
            for voice in self.zero.voices
        ]
        if self.kokoro is not None:
            for voice in self.kokoro.voices:
                languages = [
                    language
                    for language, prefix in KOKORO_LANGUAGE_CODES.items()
                    if voice.startswith(prefix)
                ]
                data.append({
                    "id": voice,
                    "name": voice,
                    "object": "audio.voice",
                    "engine": "kokoro",
                    "languages": languages,
                    "gender": {
                        "f": "female",
                        "m": "male",
                    }.get(voice[1:2], "unknown"),
                })
        return data


# Compatibility name retained for code importing the first API revision.
ZeroTTSRuntime = UnifiedRuntime


def _ffmpeg_args(output_format: str, sample_rate: int) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise APIError(
            "ffmpeg is required for streaming audio encoding.",
            500,
            error_type="server_error",
        )
    codecs = {
        "mp3": ["-codec:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "opus": ["-codec:a", "libopus", "-b:a", "64k", "-f", "ogg"],
        "aac": ["-codec:a", "aac", "-b:a", "128k", "-f", "adts"],
        "flac": ["-codec:a", "flac", "-f", "flac"],
        "wav": ["-codec:a", "pcm_s16le", "-f", "wav"],
        "pcm": ["-ar", "24000", "-codec:a", "pcm_s16le", "-f", "s16le"],
    }
    return [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *codecs[output_format], "-flush_packets", "1", "pipe:1",
    ]


async def encode_stream(
    frames: AsyncIterator[AudioFrame], output_format: str, sample_rate: int
) -> AsyncIterator[bytes]:
    """Feed inference frames and read encoded bytes concurrently."""
    if output_format == "pcm":
        async for frame in frames:
            samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
            if frame.sample_rate != 24_000:
                samples = resample_wav(samples, frame.sample_rate, 24_000)
            samples = np.clip(samples, -1.0, 1.0)
            yield (samples * 32767.0).astype("<i2").tobytes()
        return

    process = await asyncio.create_subprocess_exec(
        *_ffmpeg_args(output_format, sample_rate),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def feed() -> None:
        assert process.stdin is not None
        try:
            async for frame in frames:
                if frame.sample_rate != sample_rate:
                    raise RuntimeError("The engine changed sample rate during one request")
                samples = np.asarray(frame.samples, dtype="<f4").reshape(-1)
                process.stdin.write(samples.tobytes())
                await process.stdin.drain()
        finally:
            process.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()

    feeder = asyncio.create_task(feed())
    try:
        assert process.stdout is not None
        while chunk := await process.stdout.read(64 * 1024):
            yield chunk
        await feeder
        stderr = await process.stderr.read() if process.stderr is not None else b""
        return_code = await process.wait()
        if return_code:
            detail = stderr.decode("utf-8", errors="replace").strip()
            LOGGER.error("ffmpeg failed: %s", detail)
            raise RuntimeError("Audio encoding failed")
    finally:
        if not feeder.done():
            feeder.cancel()
            with suppress(asyncio.CancelledError):
                await feeder
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _speech_bytes(plan: SynthesisPlan, body: SpeechRequest):
    frames = plan.engine.stream(body.input, plan.voice, plan.language, body.speed)
    async for chunk in encode_stream(frames, body.response_format, plan.engine.sample_rate):
        yield chunk


async def _sse_stream(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        payload = json.dumps({
            "type": "speech.audio.delta",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }, separators=(",", ":"))
        yield f"event: speech.audio.delta\ndata: {payload}\n\n".encode()
    yield b'event: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n'


def create_app(
    settings: Settings | None = None,
    runtime: UnifiedRuntime | None = None,
    load_model: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime = runtime or UnifiedRuntime(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_model:
            await asyncio.to_thread(runtime.load)
        yield

    app = FastAPI(
        title="ZeroTTS + Kokoro ONNX OpenAI-compatible API",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError):
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = first.get("loc", ())
        param = str(loc[-1]) if loc else None
        return _error_response(APIError(
            first.get("msg", "Invalid request."),
            status_code=422,
            param=param,
            code="invalid_value",
        ))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception):
        LOGGER.exception("Unhandled API error", exc_info=exc)
        return _error_response(APIError(
            "An internal error occurred while generating audio.",
            status_code=500,
            error_type="server_error",
        ))

    async def authorize(request: Request) -> None:
        if not settings.api_key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(token, settings.api_key)
        if not valid:
            raise APIError(
                "Invalid API key.",
                status_code=401,
                code="invalid_api_key",
            )

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok" if runtime.ready else "starting",
            "ready": runtime.ready,
            "engines": runtime.status(),
        }

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict:
        now = int(time.time())
        names = ["auto", "zerotts", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
        if runtime.kokoro is not None:
            names.insert(2, "kokoro")
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": now, "owned_by": "local"}
                for name in names
            ],
        }

    @app.get("/v1/audio/voices", dependencies=[Depends(authorize)])
    async def voices() -> dict:
        return {"object": "list", "data": runtime.voices_payload()}

    @app.post("/v1/audio/speech", dependencies=[Depends(authorize)])
    async def create_speech(body: SpeechRequest) -> Response:
        plan = runtime.plan(body)
        headers = {
            "Content-Disposition": f'inline; filename="speech.{body.response_format}"',
            "X-TTS-Engine": plan.engine_id,
            "X-TTS-Voice": plan.public_voice,
            "X-TTS-Language": plan.language,
            "X-TTS-G2P": plan.g2p_backend,
        }

        async def guarded_chunks() -> AsyncIterator[bytes]:
            async with runtime.semaphore:
                async for chunk in _speech_bytes(plan, body):
                    yield chunk

        chunks: AsyncIterator[bytes] = guarded_chunks()
        if body.stream:
            if body.stream_format == "sse":
                return StreamingResponse(
                    _sse_stream(chunks), media_type="text/event-stream", headers=headers
                )
            return StreamingResponse(
                chunks, media_type=MEDIA_TYPES[body.response_format], headers=headers
            )

        async with runtime.semaphore:
            content = b"".join([
                chunk async for chunk in _speech_bytes(plan, body)
            ])
        return Response(
            content=content,
            media_type=MEDIA_TYPES[body.response_format],
            headers=headers,
        )

    return app


app = create_app()
