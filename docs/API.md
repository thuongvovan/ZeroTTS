# ZeroTTS + Kokoro ONNX OpenAI-compatible API

One FastAPI process exposes `POST /v1/audio/speech` and keeps both ONNX models
loaded. The `voice` value alone selects the engine and language.

## Start with NVIDIA CUDA

Requirements:

- Docker Engine with the Compose plugin
- an NVIDIA driver visible through `nvidia-smi`
- NVIDIA Container Toolkit configured for Docker

```bash
cp .env.example .env
docker compose up --build
```

The image contains CUDA 12, cuDNN and a single `onnxruntime-gpu` installation.
ZeroTTS weights are cached in `zerotts-model-cache`; Kokoro's ONNX graph and
voice bundle are cached in `kokoro-model-cache`. Only one API worker is started
so GPU memory is not duplicated.

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/audio/voices \
  -H "Authorization: Bearer change-me"
docker compose logs -f zerotts-api
```

## Routing

| `voice` pattern | Engine | Inferred language |
| --- | --- | --- |
| `vf_*` | ZeroTTS | Vietnamese female |
| `vm_*` | ZeroTTS | Vietnamese male |
| `af_*`, `am_*` | Kokoro ONNX | American English |
| `bf_*`, `bm_*` | Kokoro ONNX | British English |
| other native Kokoro prefixes below | Kokoro ONNX | From the prefix |

There is no `language` routing parameter. The `model` field remains accepted
for OpenAI client compatibility, but its value does not select an engine and
may be omitted in direct HTTP requests. Thus `model="kokoro"` with
`voice="vf_maichi"` still uses ZeroTTS, while `model="zerotts"` with
`voice="af_heart"` still uses Kokoro.

Bare ZeroTTS names such as `maichi` and ambiguous OpenAI names such as `alloy`
are deliberately not guessed. Use `vf_maichi`, or configure an explicit alias
such as `{"alloy":"vf_maichi"}` in `TTS_VOICE_ALIASES`. Alias targets must
also be canonical `vf_*`, `vm_*`, or native Kokoro voice names. Legacy `vi_*`
requests remain accepted and are normalized to their gendered canonical ID.

Built-in ZeroTTS voices:

| Female (`vf_`) | Male (`vm_`) |
| --- | --- |
| `vf_baotrang` | `vm_giahuy` |
| `vf_hamy` | `vm_huuduc` |
| `vf_kimoanh` | `vm_quangminh` |
| `vf_maichi` | `vm_tiendat` |

## Languages and G2P

| Inferred language | Voice prefix | G2P |
| --- | --- | --- |
| `vi` | `vf_`, `vm_` (ZeroTTS) | ZeroTTS native tokenizer |
| `en-US` | `af_`, `am_` | Misaki English |
| `en-GB` | `bf_`, `bm_` | Misaki English |
| `ja-JP` | `jf_`, `jm_` | Misaki Japanese |
| `zh-CN` | `zf_`, `zm_` | Misaki Chinese |
| `es` | `ef_`, `em_` | eSpeak fallback |
| `fr` | `ff_`, `fm_` | eSpeak fallback |
| `hi` | `hf_`, `hm_` | eSpeak fallback |
| `it` | `if_`, `im_` | eSpeak fallback |
| `pt`, `pt-BR` | `pf_`, `pm_` | eSpeak fallback |

The Kokoro adapter always supplies phonemes with `is_phonemes=True`; it does
not use Kokoro's built-in eSpeak tokenizer as its primary path. Native Misaki
frontends are preferred. For English, eSpeak is only passed to Misaki for
out-of-dictionary words. Other listed languages fall back to `EspeakG2P` only
where Misaki currently has no native frontend. Set
`KOKORO_ESPEAK_FALLBACK=false` to disable every eSpeak path; those languages
then return `unsupported_language`.

The response headers make routing observable:

- `X-TTS-Engine`: `zerotts` or `kokoro`
- `X-TTS-Voice`: canonical public voice (for example `vf_maichi`)
- `X-TTS-Language`: language inferred from the voice
- `X-TTS-G2P`: `misaki`, `espeak-fallback`, or `zerotts-native`

## Requests

Vietnamese with ZeroTTS:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Xin chào, đây là ZeroTTS.",
    "voice": "vf_maichi",
    "response_format": "mp3"
  }' --output vi.mp3
```

English with Kokoro ONNX and Misaki:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Kokoro uses Misaki as its primary G2P frontend.",
    "voice": "af_heart",
    "response_format": "opus",
    "speed": 1.0
  }' --output en.ogg
```

Raw streamed audio is the default. To receive base64-encoded Server-Sent
Events, send `"stream_format":"sse"`. To buffer generation before the HTTP
response, send `"stream":false`.

## OpenAI Python SDK

The standard OpenAI request shape works; put the canonical routing code in
`voice`:

```python
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key="change-me", base_url="http://localhost:8000/v1")

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="vf_maichi",
    input="Xin chào, đây là ZeroTTS.",
    response_format="mp3",
) as response:
    response.stream_to_file(Path("speech.mp3"))
```

For English, only the voice changes:

```python
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="af_heart",
    input="Hello from Kokoro ONNX.",
    response_format="mp3",
) as response:
    response.stream_to_file(Path("english.mp3"))
```

The OpenAI Python SDK may type-check `voice` against OpenAI's built-in names,
but custom strings are transmitted normally at runtime. Configure an alias if
your client strictly limits that field. Use `GET /v1/audio/voices` to discover
all accepted canonical voices.

## Request fields

| Field | Behavior |
| --- | --- |
| `model` | Optional on raw HTTP, defaults to `tts-1`; accepted and ignored for OpenAI compatibility. |
| `input` | Required, 1–4096 characters. Both engines split long text. |
| `voice` | Required string or `{ "id": "voice-name" }`; the canonical prefix selects engine and language. |
| `response_format` | `mp3`, `opus`, `aac`, `flac`, `wav`, or OpenAI-compatible 24 kHz `pcm`. |
| `speed` | `0.25`–`4.0`. Kokoro uses its native `0.5`–`2.0` range and resamples only outside it. |
| `stream` | `true` by default; `false` buffers the encoded response. |
| `stream_format` | `audio` by default, or `sse` for `speech.audio.delta` events. |
| `instructions` | Accepted for SDK compatibility; neither local model is instruction-conditioned. |

Errors use OpenAI-style JSON objects. GPU inference is concurrency-limited for
the full lifetime of a streamed response.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ZEROTTS_MODEL` | `zeroweight-ai/ZeroTTS` | Hugging Face model ID or local directory. |
| `ZEROTTS_PROVIDER` | `auto` (`cuda` in Compose) | `auto`, `cuda`, or `cpu`; CUDA fails fast if unavailable. |
| `ZEROTTS_DEVICE_ID` | `0` | CUDA device index shared by both engines. |
| `ZEROTTS_API_KEY` | empty | Optional bearer token. |
| `TTS_VOICE_ALIASES` | `{}` | JSON aliases whose values are canonical `vf_*`, `vm_*`, or Kokoro voices. |
| `ZEROTTS_MAX_CONCURRENCY` | `1` | Maximum simultaneous inference streams. |
| `ZEROTTS_INTRA_OP_THREADS` | `4` | ONNX Runtime CPU threads. |
| `ZEROTTS_NORMALIZE_VI` | `true` | Vietnamese number/date/abbreviation normalization. |
| `ZEROTTS_LOCAL_FILES_ONLY` | `false` | Disable ZeroTTS downloads. |
| `ZEROTTS_VOICES_DIR` | empty | Optional ZeroTTS voice-pack directory. |
| `KOKORO_ENABLED` | `true` | Load and advertise Kokoro. |
| `KOKORO_DOWNLOAD` | `true` | Download missing Kokoro assets at startup. |
| `KOKORO_MODEL_PATH` | `/data/kokoro/kokoro-v1.0.onnx` | Kokoro ONNX graph. |
| `KOKORO_VOICES_PATH` | `/data/kokoro/voices-v1.0.bin` | Kokoro voice bundle. |
| `KOKORO_VOCAB_CONFIG` | empty | Optional vocabulary config for a custom graph. |
| `KOKORO_ESPEAK_FALLBACK` | `true` | Allow eSpeak only where described above. |

## Local CPU development

Official Misaki supports this service's Python range. Install ffmpeg and eSpeak
NG through the operating system, then:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[api,kokoro,dev]"
python -m spacy download en_core_web_sm
ZEROTTS_PROVIDER=cpu zerotts-api
```

For a direct CUDA installation outside Docker, replace `onnxruntime` with
`onnxruntime-gpu`; do not keep both runtime packages installed together.
