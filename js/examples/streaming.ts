/** Minimal source example using only ZeroTTS's public web entry point. */
import {
  TtsWorker, normalizeViText, textSegments,
} from '../src';
import type { EngineId } from '../src';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const loadButton = $<HTMLButtonElement>('load');
const speakButton = $<HTMLButtonElement>('speak');
const stopButton = $<HTMLButtonElement>('stop');
const status = $<HTMLParagraphElement>('status');
const engine = $<HTMLSelectElement>('engine');
const text = $<HTMLTextAreaElement>('text');

const worker = new TtsWorker();
let sampleRate = 48_000;
let voiceName = 'maichi';
let cancel: (() => void) | null = null;
let audioContext: AudioContext | null = null;
let nextStart = 0;
let stopped = false;
const playingSources = new Set<AudioBufferSourceNode>();

async function startPlayer(): Promise<void> {
  audioContext ??= new AudioContext();
  await audioContext.resume();
  nextStart = audioContext.currentTime + 0.05;
}

function playChunk(pcm: Float32Array): void {
  if (!audioContext) return;
  const buffer = audioContext.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  playingSources.add(source);
  source.onended = () => playingSources.delete(source);
  const startAt = Math.max(nextStart, audioContext.currentTime + 0.02);
  source.start(startAt);
  nextStart = startAt + buffer.duration;
}

function stopPlayer(): void {
  for (const source of playingSources) {
    try { source.stop(); } catch { /* source already ended */ }
  }
  playingSources.clear();
  nextStart = 0;
}

loadButton.addEventListener('click', async () => {
  loadButton.disabled = true;
  speakButton.disabled = true;
  engine.disabled = true;
  try {
    const loaded = await worker.load({
      engine: engine.value as EngineId,
      gguf: 'gguf/zerotts-q4_0.gguf',
    }, (progress) => {
        const done = (progress.overallLoaded / 1e6).toFixed(0);
        const total = progress.overallTotal > 0
          ? `/${(progress.overallTotal / 1e6).toFixed(0)}` : '';
        status.textContent = progress.overallTotal > 0
          && progress.overallLoaded >= progress.overallTotal
          ? 'Đã tải xong — đang khởi tạo engine…'
          : `Đang tải ${done}${total} MB…`;
      });
    sampleRate = loaded.sampleRate;
    voiceName = loaded.voices.voices.some((voice) => voice.name === 'maichi')
      ? 'maichi'
      : (loaded.voices.voices[0]?.name ?? '');
    engine.value = loaded.engine;
    status.textContent = `Sẵn sàng với ${loaded.engine}`
      + (loaded.engine === 'ggml-cpu' && loaded.wasmThreads === false
        ? ' đơn luồng (origin HTTP).' : '.');
    speakButton.disabled = false;
    loadButton.disabled = false;
    loadButton.textContent = 'Đổi / tải lại engine';
    engine.disabled = false;
  } catch (error) {
    status.textContent = `Tải thất bại: ${(error as Error).message}`;
    loadButton.disabled = false;
    engine.disabled = false;
  }
});

speakButton.addEventListener('click', async () => {
  const input = text.value.trim();
  if (!input) return;
  speakButton.disabled = true;
  loadButton.disabled = true;
  engine.disabled = true;
  stopButton.disabled = false;
  stopped = false;
  let samples = 0;
  try {
    await startPlayer();
    const run = worker.generate({
      segments: textSegments(normalizeViText(input), 15),
      voiceName,
      options: { cfgScale: 1 },
      seed: 1234,
    });
    cancel = run.cancel;
    for await (const chunk of run.chunks) {
      playChunk(chunk);
      samples += chunk.length;
      status.textContent = `Đang phát… đã nhận ${(samples / sampleRate).toFixed(1)} giây audio.`;
    }
    if (!stopped) status.textContent = `Hoàn tất ${(samples / sampleRate).toFixed(1)} giây audio.`;
  } catch (error) {
    status.textContent = `Tạo giọng thất bại: ${(error as Error).message}`;
  } finally {
    cancel = null;
    speakButton.disabled = false;
    loadButton.disabled = false;
    engine.disabled = false;
    stopButton.disabled = true;
  }
});

stopButton.addEventListener('click', () => {
  stopped = true;
  cancel?.();
  stopPlayer();
  status.textContent = 'Đã yêu cầu dừng.';
});
