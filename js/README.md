# ZeroTTS browser demo

## Cài như thư viện từ Git

Repository có thể được cài trực tiếp vào một ứng dụng JavaScript/TypeScript:

```bash
npm install github:thuongvovan/ZeroTTS
```

```ts
import { TtsWorker, normalizeViText, textSegments } from 'zerotts-web';

const tts = new TtsWorker();
const loaded = await tts.load({
  engine: 'ggml-cpu',
  gguf: 'gguf/zerotts-q4_0.gguf',
});
const voiceName = loaded.voices.voices[0]?.name;
if (!voiceName) throw new Error('Không có giọng đọc');

const run = tts.generate({
  segments: textSegments(normalizeViText('Xin chào'), 15),
  voiceName,
  options: { cfgScale: 1 },
  seed: 1234,
});

for await (const pcm of run.chunks) {
  // PCM mono Float32Array, sample rate nằm trong loaded.sampleRate.
}
```

Package chỉ chạy trong trình duyệt. Worker, runtime GGML và WASM được bundler
sao chép tự động; Vite và webpack 5 hỗ trợ mẫu URL mà package sử dụng. Để chạy
đa luồng, server của ứng dụng cần gửi hai header:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Thiếu hai header này không làm thư viện lỗi: nó tự dùng runtime đơn luồng đi kèm.
Model không nằm trong package npm mà được tải từ Hugging Face hoặc `repo` URL do
ứng dụng truyền vào, sau đó lưu trong Cache API của trình duyệt.

ZeroTTS chạy hoàn toàn trong trình duyệt: GGML/GGUF tạo mã âm thanh, codec ONNX
giải mã từng khối PCM và Web Audio phát ngay khi khối đầu tiên sẵn sàng. Không có
server suy luận và không tải văn bản hay giọng của người dùng lên mạng.

Thư mục này chứa demo và mã nguồn của package `zerotts-web`.

## Cài mới từ đầu

Yêu cầu: Git, Node.js 20+, CMake 3.16+, trình biên dịch C/C++ và Emscripten.
Các lệnh dưới đây phù hợp với Linux, macOS và WSL. Emscripten 6.0.2 là phiên bản
đã được kiểm tra với bản web hiện tại.

```bash
git clone --recurse-submodules https://github.com/thuongvovan/ZeroTTS.git
cd ZeroTTS

# Cài Emscripten cạnh repository
git clone https://github.com/emscripten-core/emsdk.git ../emsdk
../emsdk/emsdk install 6.0.2
../emsdk/emsdk activate 6.0.2
source ../emsdk/emsdk_env.sh

# Build hai artifact CPU: đa luồng và đơn luồng
./cpp/build-wasm.sh

# Cài và chạy web
cd js
npm install
npm run dev
```

Mở các địa chỉ:

- Demo đầy đủ: `http://localhost:5173/`
- Ví dụ streaming tối giản: `http://localhost:5173/examples/streaming.html`
- Benchmark: `http://localhost:5173/bench-ggml.html`

Muốn mở từ máy khác trong LAN:

```bash
npm run dev -- --host 0.0.0.0
```

Khi truy cập bằng `http://IP:5173`, origin không đủ tin cậy để dùng
`SharedArrayBuffer`. Demo tự chọn artifact CPU đơn luồng thay vì treo sau khi
tải model. Chế độ này chạy chậm hơn; dùng `localhost` hoặc HTTPS có COOP/COEP để
được đa luồng.

Model không nằm trong bundle. Lần chạy đầu trình duyệt tải model từ Hugging Face
và lưu trong Cache API; những lần sau dùng lại bản đã tải. Nếu chế độ riêng tư
hoặc quota của trình duyệt không cho lưu file lớn, lần chạy hiện tại vẫn hoạt
động nhưng lần mở sau sẽ phải tải lại.

### Build và chạy bản production

```bash
cd js
npm run typecheck
npm run build
npm run preview
```

Thư mục cần triển khai là `js/dist/`. Máy chủ production phải trả hai header sau
để WASM đa luồng hoạt động:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Nếu trang không `crossOriginIsolated`, GGML CPU tự dùng artifact đơn luồng và
`loaded.wasmThreads` trả về `false`.

> Bản f32 GGML CPU đã được kiểm tra bit-exact với ONNX/Python.

## The download

**~820 MB** on the default (ggml/GGUF) backend — a 772 MB fp32 GGUF plus the
45 MB ONNX codec decoder — fetched once and persisted via the Cache API. The
onnxruntime backend fetches ~900 MB of fp32 graphs instead. Smaller quantized
GGUFs are selectable (q8_0 at 206 MB, q4_0 at 124 MB) but are *slower*, not
faster — see below. The UI states the size before downloading either way. See
[../docs/BROWSER.md](../docs/BROWSER.md).

## Backends

**GGML/GGUF trên CPU là mặc định**. ONNX Runtime/WASM được giữ lại
cho CFG và codec:

| | download | realtime | notes |
|---|---:|---:|---|
| **ggml/GGUF fp32** (default) | 772 MB | 6.1x | bit-exact against ONNX; no CFG |
| ggml/GGUF q8_0 | 206 MB | 3.6x | 4.9% of codes drawn differently |
| ggml/GGUF q4_0 | 124 MB | 4.1x | 36.5% differently |
| onnxruntime-web | ~858 MB | 4.2x | the only one with CFG |

Realtime figures at 4 threads on one machine; the ggml backend takes
`min(hardwareConcurrency, 8)` and gains from the extra threads, onnxruntime
caps at 4 and does not.

## API engine ổn định

Ứng dụng chỉ chọn một `EngineId`; không cần biết chi tiết runtime:

| EngineId | Runtime | Điểm khác biệt |
|---|---|---|
| `ggml-cpu` | GGML trên WASM CPU | mặc định, GGUF/q4/q8/f32 |
| `onnx-wasm` | ONNX Runtime trên WASM CPU | hỗ trợ `cfgScale > 1` |

`TtsWorker.generate()` và luồng PCM hoàn toàn giống nhau với cả hai engine. Chỉ
cấu hình lúc tải thay đổi:

```ts
import { TtsWorker, type EngineId } from './src';

const tts = new TtsWorker();
const engine: EngineId = 'ggml-cpu'; // chỉ đổi dòng này theo máy

const loaded = await tts.load({
  engine,
  gguf: 'gguf/zerotts-q4_0.gguf',
}, onProgress);

console.log(loaded.engine, loaded.threads, loaded.wasmThreads);
```

Đổi engine mà giữ nguyên API streaming:

```ts
await tts.switchEngine('onnx-wasm', onProgress);
// Mã gọi tts.generate(...) bên dưới không đổi.
```

Khi đổi, Worker cũ bị kết thúc để giải phóng hẳn heap WASM và pthread;
GGUF, số thread và repo tùy chỉnh được giữ lại khi phù hợp. Model đã
có trong Cache API nên không phải tải lại. Có thể đọc `ENGINES[id].capabilities`
để bật/tắt tính năng UI và gọi `await probeEngine(id)` để kiểm tra tương thích;
probe chỉ kiểm tra khả năng chạy, còn engine nhanh nhất vẫn phải xác định bằng
benchmark trên máy đích.

Sau khi đã benchmark và xác định thứ tự mong muốn, ứng dụng có thể chọn engine
đầu tiên mà nền tảng hỗ trợ:

```ts
import { selectSupportedEngine } from './src';

const engine = await selectSupportedEngine([
  'ggml-cpu',
  'onnx-wasm',
]);
await tts.load({ engine, gguf: 'gguf/zerotts-q4_0.gguf' }, onProgress);
```

API truyền tham số theo vị trí cũ vẫn hoạt động để không làm hỏng ứng dụng hiện
có, nhưng được đánh dấu deprecated.

`src/ggmlBackend.ts` is the browser-side wrapper for [`../cpp/`](../cpp/) — same
`generateFrames` signature as `synthesizer.ts`, verified to produce identical
frame codes at fp32. Everything downstream of frame generation (chunking, the
streaming codec session, inter-segment silence) is shared, so the two backends
differ only in `FrameSource`. The codec itself stays on onnxruntime either way.

`bench-ggml.html` đo GGML CPU và ONNX/WASM bằng cùng câu, giọng,
seed và số frame. Quantized weights có thể *chậm hơn* fp32 trong WebAssembly —
vì vậy cần đo trên máy đích thay vì suy ra từ kích thước model. Chi tiết thuật
toán và các phép đo gốc nằm trong [`../cpp/README.md`](../cpp/README.md).

## Ví dụ streaming tối giản

Chạy:

```bash
npm run example
```

Mã nguồn ở [`examples/streaming.ts`](examples/streaming.ts). Ví dụ dùng
`TtsWorker` để model không khóa giao diện, nhận từng `Float32Array` PCM qua async
iterator và xếp từng khối vào Web Audio ngay khi nhận được:

```ts
const loaded = await worker.load(
  { engine: 'ggml-cpu', gguf: 'gguf/zerotts-q4_0.gguf' },
  onProgress,
);

const run = worker.generate({
  segments: ['Xin chào. Đây là âm thanh streaming.'],
  voiceName: 'maichi',
  options: { cfgScale: 1 },
  seed: 1234,
});

for await (const pcm of run.chunks) {
  playChunk(pcm); // PCM mono float32, sample rate = loaded.sampleRate
}
```

`run.cancel()` dừng lượt đang chạy ở ranh giới frame kế tiếp.

## Benchmark trên nhiều máy và trình duyệt

```bash
npm run benchmark
```

Trang benchmark tự động:

1. Ghi user agent, số luồng CPU, bộ nhớ trình duyệt công bố, độ phân giải
   và trạng thái cross-origin isolation.
2. Warm-up bằng toàn bộ chuỗi có cùng số frame với lượt đo.
3. Chạy 1–10 lượt, báo trung vị, khoảng min–max và tốc độ realtime.
4. Chạy lại cùng seed, ghi hash và số mã khác nhau để phát hiện backend không
   lặp lại.
5. Xuất một file JSON có cả cấu hình, thông tin máy và số liệu từng lượt.

Để so sánh công bằng, trên mỗi máy hãy giữ nguyên GGUF, văn bản, giọng, seed,
số frame và số lượt. Chọn **Chạy cả GGML + ONNX**, sau đó **Tải JSON**.

Mỗi cấu hình chạy trong một Worker mới nên model và heap WASM của lượt trước
không còn giữ lại để làm sai phép đo kế tiếp. Nút **Dừng** kết thúc Worker hiện
tại nếu một backend chạy quá lâu trên máy yếu.

Có thể điều khiển từ DevTools hoặc công cụ tự động hóa trình duyệt:

```js
await window.zbench.all({
  gguf: 'zerotts-q4_0.gguf',
  frames: 80,
  runs: 3,
  seed: 1234,
});

const report = await window.zbench.report();
console.log(JSON.stringify(report, null, 2));
```

Muốn dùng model tự host, mở:

```text
/bench-ggml.html?model=https://host/model&ggufBase=https://host/gguf
```

## Layout

| File | Role |
|---|---|
| `src/synthesizer.ts` | the two-calls-per-frame loop — see [../docs/RUNTIME.md](../docs/RUNTIME.md) |
| `src/index.ts` | stable public exports for embedding the web runtime |
| `src/engine.ts` | engine ids, capabilities, compatibility probe and load options |
| `src/chunking.ts` | long-form segmentation, port of `zerotts.chunking` |
| `src/textNorm.ts` | Vietnamese text normalization, port of `zerotts.text_norm` |
| `src/codec.ts` | MOSS decoder: batch + KV-cached streaming |
| `src/tokenizer.ts` | BPE over `tokenizer.json` |
| `src/loader.ts` | create sessions from the downloaded graphs |
| `src/repo.ts` | resolve repo URLs, size the download, load voices |
| `src/worker.ts` | the Web Worker the model runs in |
| `src/workerClient.ts` | main-thread handle on that worker |
| `src/cache.ts` | download progress + Cache API persistence |
| `src/player.ts` | AudioWorklet ring buffer, WAV export |
| `src/rng.ts` | seedable PRNG — the sampler's draws are graph *inputs* |
| `src/samples.ts` | the sample texts, shared with the Python UI |
| `src/voicePack.ts` | read voice packs out of a dropped `.zip` |
| `src/voiceStore.ts` | keep imported voices in IndexedDB across visits |
| `src/main.ts` | demo UI wiring (imports no runtime code) |
| `src/ggmlBackend.ts` | the ggml/GGUF backend (see [../cpp/](../cpp/)) |
| `src/benchGgml.ts` | the ONNX-vs-ggml A/B page |
| `src/benchWorker.ts` | isolated worker used by each benchmark case |
| `examples/streaming.ts` | minimal Web Worker + Web Audio streaming example |

The model runs in a Web Worker: ORT-web's WASM backend computes on the calling
thread, and two graph calls per 80 ms frame on the UI thread freeze the tab for
the whole take. `main.ts` therefore imports nothing that pulls in
`onnxruntime-web` — it sends text to the worker and gets audio chunks back. See
[../docs/BROWSER.md](../docs/BROWSER.md) for the two subtleties (cancellation
needs a macrotask; chunks are transferred, not copied).

## Live playback

**Phát ngay trong lúc đang tạo** is on by default: chunks go to an AudioWorklet
ring buffer as they are decoded, so the take is audible while it is still being
made. Turn it off on a machine that generates slower than realtime — the buffer
runs dry between chunks and the worklet plays silence into the gaps, which
sounds like a broken model rather than a slow one. Off, the chunks are still
collected and the WAV still assembled; they are simply not pushed at the
speakers. The choice is remembered in `localStorage`.

(`webui/app.py` has the same checkbox, and skips opening its stream entirely
when it is off.)

## Your own voices

The voice card's second tab, **Nhập giọng của bạn**, takes the `.zip` a
[platform.zeroweight.ai](https://platform.zeroweight.ai/audio) voice downloads
as. Drop it on the page — no unzipping — and the voice joins the picker under
its own group.

Nothing is uploaded. `src/voicePack.ts` reads the archive in the page: the pack
folders inside it, then `voice.bin` when it is there (the raw float32 array) or
`voice.npz` when it is not, which means unzipping a second time and parsing
`voice_emb.npy`. Both zip layers can be stored or deflated, the latter via
`DecompressionStream`. `__MACOSX/` and other file-manager droppings are skipped.
The latents then go to the worker in the `generate` message, as `voiceEmb`
instead of a `voiceName` it would fetch.

`src/voiceStore.ts` keeps the imported voices in IndexedDB — the browser's
answer to the Python side copying a pack into `~/.zerotts/voices` — so they are
in the picker on the next visit, and each one has an × to remove it. Storage
being unavailable (a private window, a full quota) costs persistence and nothing
else.

A pack whose latents are the wrong length for the loaded weights is dropped with
a message rather than offered — wrong latents are the right dtype and rank, so
they would generate cleanly in a voice that is nobody's. See
[../docs/VOICES.md](../docs/VOICES.md).

## Parity checks

Two, both of which have caught real bugs:

**Frame codes vs. Python** — the end-to-end one. Sampling happens inside
`local_frame_decode.onnx` and takes its random draws as graph *inputs*, so the
caller owns the randomness and an exact cross-language comparison is possible:

```bash
npm install --no-save onnxruntime-node
node --experimental-strip-types test/frames.mjs /path/to/model
python test/py_frames.py /path/to/model
```

Frame codes must match element for element. (Audio *samples* may differ in the
last bits at chunk seams when comparing streaming to batch decode — that is the
KV-cached decoder, not an error — but the codes must not.)

**Text normalization vs. Python** — see below.

**ggml vs. ONNX** — the third, and the one that keeps the two backends honest
about each other. An fp32 GGUF must reproduce the ONNX frame codes exactly:

```bash
npm install --no-save onnxruntime-node
npm run parity:ggml
```

## Normalizer parity

`src/textNorm.ts` and `src/zerotts/text_norm/vi_normalizer.py` are
hand-maintained copies of the same rules. Nothing but a test keeps them in
step, and a divergence is invisible in normal use — it shows up only as the
browser and the package speaking the same sentence differently.

```bash
npm run parity        # 1262 cases, must be 100%
```

After changing either side, regenerate the corpus from Python and re-run:

```bash
python test/gen_cases.py > test/cases.json
npm run parity
```

CI runs it on every push.

## Cross-origin isolation

`vite.config.ts` sets COOP/COEP headers so `SharedArrayBuffer` is available and
the CPU engines can use multi-threaded WASM. Without a trustworthy origin or
these headers, ZeroTTS selects the separate single-thread GGML artifact and ORT
also uses one thread; generation is several times slower but remains functional.
**Whatever hosts the built bundle should send the same two headers** for full
CPU performance. GitHub Pages does not, so a Pages deployment will be slow
unless you add a service-worker shim.

## Things that are easy to get wrong here

Three of these were real bugs in an earlier revision; they are called out
because each one fails *quietly*.

- **One codec session across all segments.** The backbone re-primes its
  `[voice | soa]` prefix per segment, but the codec's streaming decoder is
  causal and KV-cached — opening a fresh decoder per segment restarts that cache
  cold and clicks at every boundary. Each segment sounds fine in isolation.
- **The playback ring buffer must not lap itself.** Generation runs ~2x
  realtime, so the writer gains about a second of audio per second played. An
  unconditional modulo write silently overwrites unplayed samples once the lead
  exceeds the buffer. It is sized for the model's 120 s ceiling and reports an
  overflow rather than wrapping. A read/write index pair also makes "full" look
  identical to "empty"; monotonic counters are used instead.
- **int64 must literally be a `BigInt64Array`.** ORT rejects anything else
  ("A int64 tensor's data must be type of function BigInt64Array()"). A graph's
  int64 *output* is not guaranteed to come back as one across ORT-web versions
  and execution providers, and `outputs.x.data as BigInt64Array` is a cast that
  asserts rather than checks — so a wrong type survives until that value is fed
  back in as an input, which surfaces during `warmup()` at load time. Every int64
  input goes through `toBigInt64` / `i64` for that reason.
- **One RNG across segments.** Sampling draws are graph *inputs*, so a fresh
  `Rng(seed)` per segment replays the identical draw sequence for every segment.

## Known gaps

- ONNX remains necessary for the waveform codec, and ONNX is still the only
  generation backend that implements CFG above 1.
