/** Demo UI wiring. The interesting code is in synthesizer.ts / codec.ts.
 *
 *  The page mirrors webui/app.py: a basic view that is model → voice → text →
 *  player and nothing else, with the run log, the segments and every sampling
 *  knob folded into "Tuỳ chọn nâng cao". Templates and the take history are
 *  lists of real rows with a preview, not a strip of tags.
 *
 *  Nothing here imports onnxruntime-web, and that is load-bearing: the model
 *  runs in a worker (worker.ts) so that a generation — minutes of blocking WASM
 *  compute — never touches this thread. This file only ever handles text going
 *  out and finished audio chunks coming back.
 */

import bannerUrl from '../../docs/assets/banner.png';

import { fetchWithCache } from './cache';
import { textSegments } from './chunking';
import {
  DEFAULT_ENGINE, EngineId, engineDefinition,
} from './engine';
import { normalizeViText } from './textNorm';
import {
  Backend, DEFAULT_GGUF, GGUF_BUILDS, defaultRepo, voicePreviewUrl,
} from './repo';
import { loadSampleTexts } from './samples';
import { StreamPlayer, toWavBlob } from './player';
import { LocalVoice, readVoiceZip } from './voicePack';
import { deleteVoice, loadVoices, saveVoice } from './voiceStore';
import { TtsWorker } from './workerClient';
import { VoiceIndex } from './types';

/** Same first-load text as the Gradio UI. */
const DEFAULT_TEXT = 'Xin chào tất cả mọi người. Giọng nói này được tạo ra bởi ZeroTTS.';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  banner: $<HTMLImageElement>('banner'),
  load: $<HTMLButtonElement>('load'),
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  download: $<HTMLAnchorElement>('download'),
  clear: $<HTMLButtonElement>('clear-cache'),
  text: $<HTMLTextAreaElement>('text'),
  voice: $<HTMLSelectElement>('voice'),
  repo: $<HTMLInputElement>('repo'),
  engine: $<HTMLSelectElement>('engine'),
  quant: $<HTMLSelectElement>('quant'),
  quantField: $<HTMLElement>('quant-field'),
  seed: $<HTMLInputElement>('seed'),
  cfg: $<HTMLInputElement>('cfg'),
  temperature: $<HTMLInputElement>('temperature'),
  chunkSec: $<HTMLInputElement>('chunk-sec'),
  textNorm: $<HTMLInputElement>('text-norm'),
  status: $<HTMLDivElement>('status'),
  bar: $<HTMLDivElement>('bar'),
  barWrap: $<HTMLDivElement>('bar-wrap'),
  sizeNote: $<HTMLParagraphElement>('size-note'),
  preview: $<HTMLAudioElement>('preview'),
  voiceMeta: $<HTMLParagraphElement>('voice-meta'),
  tabPick: $<HTMLButtonElement>('tab-pick'),
  tabImport: $<HTMLButtonElement>('tab-import'),
  panelPick: $<HTMLDivElement>('panel-pick'),
  panelImport: $<HTMLDivElement>('panel-import'),
  voiceDrop: $<HTMLLabelElement>('voice-drop'),
  voiceZip: $<HTMLInputElement>('voice-zip'),
  voiceLoadStatus: $<HTMLParagraphElement>('voice-load-status'),
  voiceList: $<HTMLDivElement>('voice-list'),
  segments: $<HTMLPreElement>('segments'),
  result: $<HTMLAudioElement>('result'),
  livePill: $<HTMLDivElement>('live-pill'),
  speakLive: $<HTMLInputElement>('speak-live'),
  history: $<HTMLDivElement>('history'),
  templates: $<HTMLDivElement>('templates'),
};

interface Take { url: string; title: string; text: string; }

let samples: Record<string, string> = {};

const tts = new TtsWorker();
let sampleRate = 0;
let voices: VoiceIndex = { voices: [] };
/** Installed voices, by the `local:` value they get in the picker. Kept on this
 *  thread: the page previews and labels them, and the latents ride along on
 *  each generate message. Restored from IndexedDB at startup. */
const localVoices = new Map<string, LocalVoice>();
/** Preview object URLs this page made for them, and therefore has to revoke. */
const localPreviewUrls = new Map<string, string>();
/** The latents shape these weights want, once the model has said. Null before
 *  the first load — a pack can be picked before then, and is checked when the
 *  model arrives instead of being refused for a reason nobody can act on. */
let voiceShape: { nVoiceQueries: number; dModel: number } | null = null;
let base = '';
let player: StreamPlayer | null = null;
let cancelRun: (() => void) | null = null;
/** Set by Stop, so the run that unwinds afterwards knows not to report itself. */
let stopped = false;
const takes: Take[] = [];

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;

function status(text: string): void {
  els.status.textContent = text;
}

function progress(fraction: number | null): void {
  els.barWrap.style.display = fraction === null ? 'none' : 'block';
  if (fraction !== null) els.bar.style.width = `${Math.round(fraction * 100)}%`;
}

function live(on: boolean): void {
  els.livePill.classList.toggle('on', on);
}

/** The stable engine id is the only runtime choice the rest of the UI needs. */
function engine(): EngineId {
  return (els.engine.value as EngineId) || DEFAULT_ENGINE;
}

function backend(): Backend {
  return engineDefinition(engine()).backend;
}

function repo(): string {
  return els.repo.value || engineDefinition(engine()).defaultRepo;
}

/** Only one GGUF is ever fetched — whichever this names. The others in the
 *  repository are alternatives, not extra downloads. */
function gguf(): string | undefined {
  return backend() === 'ggml' ? (els.quant.value || DEFAULT_GGUF) : undefined;
}

for (const build of GGUF_BUILDS) {
  const option = document.createElement('option');
  option.value = build.file;
  option.textContent = build.label;
  option.selected = build.file === DEFAULT_GGUF;
  els.quant.append(option);
}

function updateEngineUi(): void {
  const definition = engineDefinition(engine());
  const isGgml = definition.backend === 'ggml';
  els.quantField.hidden = !isGgml;
  els.cfg.disabled = !definition.capabilities.cfg;
  if (!definition.capabilities.cfg) els.cfg.value = '1';
}
updateEngineUi();

function disableEngineControls(disabled: boolean): void {
  els.engine.disabled = disabled;
  els.quant.disabled = disabled;
  els.repo.disabled = disabled;
}

/** Sequence number for the size lookup. Changing backend, quantization or repo
 *  all fire one, each involves a HEAD per file, and they do not come back in
 *  the order they were sent — without this a slow lookup for the previous
 *  selection lands after the current one and reports the wrong size. */
let sizeRequest = 0;

async function refreshSizeNote(): Promise<void> {
  const mine = ++sizeRequest;
  const kind = backend() === 'ggml'
    ? `GGUF ${(gguf() ?? '').replace(/^gguf\/zerotts-|\.gguf$/g, '')}`
    : 'ONNX fp32';
  try {
    const info = await tts.downloadInfo({ engine: engine(), repo: repo(), gguf: gguf() });
    if (mine !== sizeRequest) return;
    els.sizeNote.textContent = info.cached
      ? `Mô hình đã có sẵn trên máy (${mb(info.bytes)}) — tải sẽ rất nhanh.`
      : `Lần đầu sẽ tải khoảng ${mb(info.bytes)} (${kind}) và ` +
        `lưu lại cho những lần sau. Nên dùng máy tính với mạng nhanh.`;
  } catch {
    if (mine !== sizeRequest) return;
    els.sizeNote.textContent = backend() === 'ggml'
      ? 'Lần đầu sẽ tải khoảng 820 MB (GGUF f32) và lưu lại cho những lần sau.'
      : 'Lần đầu sẽ tải khoảng 900 MB (ONNX fp32) và lưu lại cho những lần sau.';
  }
}

els.load.addEventListener('click', async () => {
  els.load.disabled = true;
  els.generate.disabled = true;
  disableEngineControls(true);
  try {
    status('Đang tải mô hình…');
    progress(0);
    const loaded = await tts.load({
      engine: engine(), repo: repo(), gguf: gguf(),
    }, (p) => {
      if (p.overallTotal > 0) progress(p.overallLoaded / p.overallTotal);
      els.sizeNote.textContent = p.overallTotal > 0 && p.overallLoaded >= p.overallTotal
        ? `Đã tải xong ${mb(p.overallTotal)} — đang khởi tạo engine, `
          + 'bước này có thể lâu với GGUF f32 hoặc CPU đơn luồng…'
        : `Đang tải ${p.file.split('/').pop()} — `
          + `${mb(p.overallLoaded)} / ${mb(p.overallTotal)}`;
    });
    voices = loaded.voices;
    base = loaded.base;
    sampleRate = loaded.sampleRate;
    voiceShape = { nVoiceQueries: loaded.nVoiceQueries, dModel: loaded.dModel };
    // A pack picked before the model was loaded went in unchecked; now there is
    // something to check it against.
    const mismatch = dropMismatchedLocalVoices();
    if (mismatch) voiceLoadStatus(mismatch);

    renderVoiceOptions();
    // Prefer "maichi" (Mai Chi) as the default, same as the README/webui — its
    // dropdown position is not guaranteed to be first once a repo ships more
    // presets than the shipped index happens to list it first.
    if (voices.voices.some((v) => v.name === 'maichi')) els.voice.value = 'maichi';
    else if (voices.voices.length) els.voice.value = voices.voices[0].name;
    updateVoiceUi();

    player = new StreamPlayer(sampleRate);
    progress(null);
    els.engine.value = loaded.engine;
    updateEngineUi();
    const runtime = loaded.engine
      + (loaded.engine === 'ggml-cpu' && loaded.wasmThreads === false
        ? ' (đơn luồng do origin HTTP)' : '');
    els.sizeNote.textContent =
      `Đã sẵn sàng — ${voices.voices.length} giọng, ${sampleRate / 1000} kHz, `
      + `bộ máy ${runtime}.`;
    status(`Ready — ${voices.voices.length} voice(s), ${sampleRate / 1000} kHz, ${runtime}.`);
    els.voice.disabled = false;
    els.generate.disabled = false;
    els.load.disabled = false;
    els.load.textContent = 'Đổi / tải lại engine';
    disableEngineControls(false);
  } catch (error) {
    progress(null);
    els.sizeNote.textContent = `Tải mô hình thất bại: ${(error as Error).message}`;
    status(`Load failed: ${(error as Error).message}`);
    els.load.disabled = false;
    disableEngineControls(false);
  }
});

els.generate.addEventListener('click', async () => {
  if (!player) return;
  const text = els.text.value.trim();
  if (!text) { status('Hãy nhập văn bản trước.'); return; }

  els.generate.disabled = true;
  els.load.disabled = true;
  disableEngineControls(true);
  els.stop.disabled = false;
  els.download.style.display = 'none';
  stopped = false;

  // Read once, here: toggling mid-run would start or stop the player halfway
  // through a take, which is worse than either setting.
  const speakLive = els.speakLive.checked;

  const seedValue = Number(els.seed.value);
  const seed = Number.isFinite(seedValue) && seedValue >= 0 ? seedValue : undefined;

  const selected = els.voice.value;
  const local = localVoices.get(selected);
  const voiceName = local ? '' : selected;
  const chunks: Float32Array[] = [];
  const started = performance.now();
  let firstChunkAt: number | null = null;

  // Normalize BEFORE segmenting — an expansion is several times longer than
  // what it replaces, and the chunk budget has to size the text the model
  // actually receives. Same order as the Python engine.
  const normalized = els.textNorm.checked ? normalizeViText(text) : text;
  // The model is bounded by maxFrames (120 s); a whole article as one utterance
  // is silently truncated mid-sentence.
  const segments = textSegments(normalized, Number(els.chunkSec.value));
  els.segments.textContent = segments.map((s, i) => `[${i + 1}] ${s}`).join('\n');

  try {
    // Off, the chunks are still collected and the WAV is still assembled — they
    // are simply not pushed at the speakers. On a machine slower than realtime
    // the ring buffer would run dry between chunks and play silence into the
    // gaps, which sounds like a broken model rather than a slow one.
    if (speakLive) await player.start();
    status('Generating…');
    live(speakLive);

    // The worker loads the voice and runs the model; this thread stays free to
    // paint, so the buttons and the log update while generation is under way.
    const run = tts.generate({
      segments, voiceName, voiceEmb: local?.emb,
      options: {
        cfgScale: Number(els.cfg.value), audioTemperature: Number(els.temperature.value),
      },
      seed,
    });
    cancelRun = run.cancel;

    for await (const chunk of run.chunks) {
      if (firstChunkAt === null) {
        firstChunkAt = performance.now() - started;
        status(`${speakLive ? 'Playing' : 'Generating'} — first audio in `
          + `${firstChunkAt.toFixed(0)} ms`);
      }
      chunks.push(chunk);
      if (speakLive) player.push(chunk);
    }
    if (speakLive) player.finish();
    // Stop already reset the player and said so; a partial take is not worth
    // overwriting that with statistics.
    if (stopped) return;

    const total = chunks.reduce((n, c) => n + c.length, 0);
    const audio = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) { audio.set(c, offset); offset += c.length; }

    const elapsed = (performance.now() - started) / 1000;
    const duration = total / sampleRate;
    const overflow = speakLive && player.overflowed
      ? ' — WARNING: playback buffer overflowed, live audio is incomplete (the ' +
        'downloaded WAV is not)'
      : '';
    status(`${duration.toFixed(2)}s audio in ${elapsed.toFixed(2)}s — ` +
           `${(duration / elapsed).toFixed(1)}x realtime, ` +
           `first audio ${firstChunkAt?.toFixed(0) ?? '?'} ms, ` +
           `${segments.length} segment(s)${overflow}`);

    // One blob per take, kept for the session: it is both the main player's
    // source and the history entry's, so it must outlive this handler.
    const url = URL.createObjectURL(toWavBlob(audio, sampleRate));
    els.download.href = url;
    els.download.download = 'zerotts.wav';
    els.download.style.display = 'inline-block';
    showTake(url, false);
    addTake({ url, text, title: takeTitle(selected, duration) });
  } catch (error) {
    if (!stopped) status(`Generation failed: ${(error as Error).message}`);
  } finally {
    live(false);
    els.generate.disabled = false;
    els.load.disabled = false;
    disableEngineControls(false);
    els.stop.disabled = true;
    cancelRun = null;
  }
});

els.stop.addEventListener('click', async () => {
  stopped = true;
  cancelRun?.();
  await player?.stop();
  live(false);
  status('Stopped.');
});

els.clear.addEventListener('click', async () => {
  await tts.clearCache();
  await refreshSizeNote();
  status('Cache cleared. The next load will re-download the model.');
});

/** Populate the voice <select> from `voices.voices`. No tag filter: with ~9
 *  shipped packs the labels already carry the tags, and a filter above the
 *  picker is one more thing to understand before hearing anything. */
function renderVoiceOptions(): void {
  const keep = els.voice.value;
  els.voice.innerHTML = '<option value="">(không dùng giọng nào)</option>';

  const option = (value: string, label: string, tags: string[]) => {
    const el = document.createElement('option');
    el.value = value;
    el.textContent = tags.length ? `${label} — ${tags.join(', ')}` : label;
    return el;
  };

  // The user's own voices lead, in their own group: someone who has just
  // loaded a folder is looking for what they loaded, not for the presets.
  if (localVoices.size) {
    const group = document.createElement('optgroup');
    group.label = 'Giọng của bạn';
    for (const [value, v] of localVoices) {
      group.append(option(value, v.displayName, v.tags));
    }
    els.voice.append(group);
  }

  const shipped = document.createElement('optgroup');
  shipped.label = 'Giọng có sẵn';
  for (const v of voices.voices) {
    shipped.append(option(v.name, v.display_name || v.name, v.tags ?? []));
  }
  if (voices.voices.length) els.voice.append(shipped);

  // A re-render happens when a voice is added or dropped, not when one is
  // chosen — losing the selection to it would be a silent switch of speaker.
  if (keep && [...els.voice.options].some((o) => o.value === keep)) els.voice.value = keep;
}

/** Preview clips already fetched, as object URLs. */
const previewUrls = new Map<string, string>();
/** Bumped on every voice change so a slow fetch cannot land on a newer voice. */
let previewToken = 0;

async function updateVoiceUi(): Promise<void> {
  const name = els.voice.value;
  const token = ++previewToken;

  // An installed voice carries its own preview and metadata — neither needs the
  // weights repo, so this runs before the model is loaded as well as after.
  const local = localVoices.get(name);
  if (local) {
    els.voiceMeta.textContent = [local.displayName, local.language, ...local.tags]
      .filter(Boolean).join(' · ');
    const url = localPreviewUrl(name);
    els.preview.style.display = url ? 'block' : 'none';
    if (url) els.preview.src = url;
    else els.preview.removeAttribute('src');
    return;
  }

  if (!name || !base) {
    els.preview.removeAttribute('src');
    els.preview.style.display = 'none';
    els.voiceMeta.textContent = name ? '' : 'Không chọn giọng — mô hình tự chọn một '
      + 'giọng, và giọng đó không giống nhau giữa các đoạn.';
    return;
  }
  const info = voices.voices.find((v) => v.name === name);
  els.voiceMeta.textContent = info
    ? [info.display_name || info.name, info.language, ...(info.tags ?? [])]
        .filter(Boolean).join(' · ')
    : name;

  // Fetch the clip whole and hand the element a blob, rather than pointing it
  // at the CDN and letting it stream: a progressive fetch that stalls or gets
  // cancelled — which is easy while the model download is saturating the
  // connection — fires `error` mid-playback, and the audio would cut out and
  // vanish. A blob is either there or it never appears.
  els.preview.style.display = 'block';
  try {
    let url = previewUrls.get(name);
    if (!url) {
      const buf = await fetchWithCache(voicePreviewUrl(base, name));
      url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
      previewUrls.set(name, url);
    }
    if (token !== previewToken) return;  // the user moved on while we fetched
    els.preview.src = url;
  } catch {
    // No preview shipped for this voice — an empty player, not a broken one.
    if (token === previewToken) els.preview.style.display = 'none';
  }
}

// ── the user's own voices ───────────────────────────────────────────────────

/**
 * The voice card's two tabs.
 *
 * Picking is the default and the common act; importing is the once-per-voice
 * one, so it sits behind a tab rather than taking a card of its own next to the
 * picker it feeds.
 */
function showVoiceTab(tab: 'pick' | 'import'): void {
  const importing = tab === 'import';
  els.panelPick.hidden = importing;
  els.panelImport.hidden = !importing;
  els.tabPick.classList.toggle('on', !importing);
  els.tabImport.classList.toggle('on', importing);
  els.tabPick.setAttribute('aria-selected', String(!importing));
  els.tabImport.setAttribute('aria-selected', String(importing));
}

els.tabPick.addEventListener('click', () => showVoiceTab('pick'));
els.tabImport.addEventListener('click', () => showVoiceTab('import'));

/** The picker value an installed voice gets. Namespaced so a voice called
 *  `maichi` cannot be confused with the shipped pack of that name. */
const localValue = (name: string) => `local:${name}`;

function voiceLoadStatus(text: string): void {
  els.voiceLoadStatus.textContent = text;
}

/** The preview clip's URL, made on demand and cached — a voice restored from
 *  storage has a Blob, not a URL, and most of them are never played. */
function localPreviewUrl(value: string): string | null {
  const voice = localVoices.get(value);
  if (!voice?.preview) return null;
  let url = localPreviewUrls.get(value);
  if (!url) localPreviewUrls.set(value, (url = URL.createObjectURL(voice.preview)));
  return url;
}

function forgetLocalVoice(value: string): void {
  localVoices.delete(value);
  const url = localPreviewUrls.get(value);
  if (url) { URL.revokeObjectURL(url); localPreviewUrls.delete(value); }
}

/**
 * Drop any installed voice whose latents do not fit these weights.
 *
 * Shape is the only check worth making and the only one that can be made: the
 * wrong latents have the right dtype and rank, so they feed the model cleanly
 * and come out as a confident voice that is nobody's. Runs once the model has
 * reported its shape, which may be after the voice was installed. Returns the
 * sentence to show, or '' when nothing was dropped.
 */
function dropMismatchedLocalVoices(): string {
  if (!voiceShape) return '';
  const expected = voiceShape.nVoiceQueries * voiceShape.dModel;
  const dropped: string[] = [];
  for (const [value, voice] of localVoices) {
    if (voice.emb.length === expected) continue;
    forgetLocalVoice(value);
    void deleteVoice(voice.name);  // and do not offer it again next visit
    dropped.push(voice.displayName);
  }
  if (!dropped.length) return '';
  return `Đã bỏ ${dropped.join(', ')}: latents không khớp mô hình này `
    + `(cần ${expected} số thực) — giọng này thuộc về một bộ trọng số khác.`;
}

/** The installed voices, each with a way to remove it. Without this a voice
 *  that turns out to be wrong is stuck in the browser for good. */
function renderVoiceList(): void {
  if (!localVoices.size) {
    els.voiceList.innerHTML = '';
    return;
  }
  els.voiceList.replaceChildren(...[...localVoices].map(([value, voice]) => {
    const row = document.createElement('div');
    row.className = 'voice-row';

    const label = document.createElement('button');
    label.type = 'button';
    label.className = 'voice-row-name';
    label.textContent = voice.tags.length
      ? `${voice.displayName} — ${voice.tags.join(', ')}` : voice.displayName;
    label.addEventListener('click', () => {
      els.voice.value = value;
      els.voice.disabled = false;
      void updateVoiceUi();
      showVoiceTab('pick');
    });

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'voice-row-x';
    remove.title = `Xoá ${voice.displayName}`;
    remove.setAttribute('aria-label', `Xoá ${voice.displayName}`);
    remove.textContent = '×';
    remove.addEventListener('click', () => {
      const wasSelected = els.voice.value === value;
      forgetLocalVoice(value);
      void deleteVoice(voice.name);
      renderVoiceOptions();
      renderVoiceList();
      if (wasSelected) void updateVoiceUi();
      voiceLoadStatus(`Đã xoá ${voice.displayName}.`);
    });

    row.append(label, remove);
    return row;
  }));
}

/** Install a dropped zip: read it, check it, keep it. */
async function installVoiceZip(file: File): Promise<void> {
  voiceLoadStatus(`Đang đọc ${file.name}…`);
  const { voices: loaded, errors } = await readVoiceZip(file);

  for (const voice of loaded) {
    // Re-dropping the same voice is an update, not a second copy.
    forgetLocalVoice(localValue(voice.name));
    localVoices.set(localValue(voice.name), voice);
  }

  const mismatch = dropMismatchedLocalVoices();
  const kept = loaded.filter((v) => localVoices.has(localValue(v.name)));
  // Persisted only once it has survived the shape check, so a voice for other
  // weights is not waiting in the picker on the next visit.
  await Promise.all(kept.map(saveVoice));

  renderVoiceOptions();
  renderVoiceList();
  if (kept.length) {
    // Select what was just installed: dropping a voice in has one obvious
    // follow-up, and hunting for it in the dropdown is not it.
    els.voice.value = localValue(kept[0].name);
    els.voice.disabled = false;
    await updateVoiceUi();
    // The import is done and the voice is selected; leaving the user on the
    // drop zone hides the thing they just made happen.
    showVoiceTab('pick');
  }

  const lines: string[] = [];
  if (kept.length) {
    lines.push(`✅ Đã nạp ${kept.length} giọng: `
      + kept.map((v) => v.displayName).join(', ')
      + '. Giọng được lưu trong trình duyệt — lần sau mở lại là có sẵn.');
    if (!voiceShape) lines.push('Hãy tải mô hình để nghe thử và tạo giọng nói.');
  }
  if (mismatch) lines.push(mismatch);
  lines.push(...errors);
  voiceLoadStatus(lines.join(' '));
}

/** One zip at a time, in the order they were dropped — installing two at once
 *  would race on the picker selection and the status line. */
async function installVoiceZips(files: File[]): Promise<void> {
  for (const file of files) await installVoiceZip(file);
}

const isZip = (file: File) =>
  file.name.toLowerCase().endsWith('.zip') || file.type === 'application/zip';

function acceptDropped(files: File[]): void {
  const zips = files.filter(isZip);
  if (!zips.length) {
    voiceLoadStatus('Hãy chọn file .zip đã tải về từ thư viện giọng.');
    return;
  }
  void installVoiceZips(zips);
}

els.voiceZip.addEventListener('change', () => {
  const files = [...(els.voiceZip.files ?? [])];
  // Reset first, so picking the same file again still fires `change`.
  els.voiceZip.value = '';
  acceptDropped(files);
});

// The whole card is the drop target, not just the button inside it: a file
// aimed at a 200px label and released 10px off would otherwise navigate the
// tab to the zip.
for (const type of ['dragenter', 'dragover'] as const) {
  els.voiceDrop.addEventListener(type, (event) => {
    event.preventDefault();
    els.voiceDrop.classList.add('over');
  });
}
for (const type of ['dragleave', 'dragend'] as const) {
  els.voiceDrop.addEventListener(type, () => els.voiceDrop.classList.remove('over'));
}
els.voiceDrop.addEventListener('drop', (event) => {
  event.preventDefault();
  els.voiceDrop.classList.remove('over');
  acceptDropped([...(event.dataTransfer?.files ?? [])]);
});
// A file dropped anywhere else on the page still opens in the tab by default,
// which loses whatever was typed. Swallow it.
for (const type of ['dragover', 'drop'] as const) {
  window.addEventListener(type, (event) => {
    if (!els.voiceDrop.contains(event.target as Node)) event.preventDefault();
  });
}

// ── the two list panels ──────────────────────────────────────────────────────

/** One clickable row: a bold title over a single-line preview. */
function listItem(title: string, preview: string, onClick: () => void): HTMLButtonElement {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'item';
  const t = document.createElement('div');
  t.className = 't';
  t.textContent = title;
  const p = document.createElement('div');
  p.className = 'p';
  p.textContent = preview;
  item.append(t, p);
  item.addEventListener('click', onClick);
  return item;
}

/** Collapse a sample down to the one line a row shows (the CSS ellipsises it). */
const oneLine = (text: string) => text.replace(/\s+/g, ' ').trim();

function renderTemplates(): void {
  els.templates.replaceChildren(
    ...Object.entries(samples).map(([name, text]) =>
      listItem(name, oneLine(text), () => {
        els.text.value = text;
        els.text.scrollIntoView({ behavior: 'smooth', block: 'center' });
      })),
  );
}

function takeTitle(value: string, seconds: number): string {
  const voice = voices.voices.find((v) => v.name === value);
  const label = localVoices.get(value)?.displayName
    ?? (voice ? (voice.display_name || voice.name) : 'không giọng');
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  return `🔊  ${label}  ·  ${now}  ·  ${seconds.toFixed(0)}s`;
}

/**
 * Load a take into the main player — the one place audio comes from on this
 * page, so a history row never spawns a second player of its own.
 *
 * `autoplay` is false for a take that was just generated. Generation runs
 * faster than real time, so when the last chunk is produced the live ring
 * buffer still has seconds of audio left to play; starting the finished WAV
 * from zero at that moment plays the take on top of itself.
 */
function showTake(url: string, autoplay: boolean): void {
  els.result.src = url;
  if (autoplay) {
    els.result.play().catch(() => { /* autoplay may be blocked; controls still work */ });
  }
}

function addTake(take: Take): void {
  takes.unshift(take);
  renderHistory();
}

function renderHistory(): void {
  if (!takes.length) {
    els.history.innerHTML = '<div class="empty">Chưa có bản nào trong phiên này.</div>';
    return;
  }
  els.history.replaceChildren(
    ...takes.map((t) => listItem(t.title, oneLine(t.text), () => showTake(t.url, true))),
  );
}

/**
 * Live playback is remembered across visits.
 *
 * The reason to turn it off is the machine, not the take — someone who has
 * switched it off once because their laptop generates at 0.4x wants it off
 * every time, and re-ticking it on each reload is the same annoyance repeated.
 * Wrapped because storage throws in a private window, where the default simply
 * stands.
 */
const SPEAK_LIVE_KEY = 'zerotts:speak-live';

try {
  if (localStorage.getItem(SPEAK_LIVE_KEY) === 'off') els.speakLive.checked = false;
} catch { /* no storage: the default is on, which is the right guess */ }

els.speakLive.addEventListener('change', () => {
  try {
    localStorage.setItem(SPEAK_LIVE_KEY, els.speakLive.checked ? 'on' : 'off');
  } catch { /* the setting still holds for this session */ }
});

els.voice.addEventListener('change', updateVoiceUi);

function markEngineConfigChanged(): void {
  if (!tts.currentEngine) return;
  els.generate.disabled = true;
  els.load.disabled = false;
  els.load.textContent = 'Tải engine đã chọn';
  status('Cấu hình engine đã thay đổi — hãy tải engine đã chọn để tiếp tục.');
}

els.repo.addEventListener('change', () => {
  markEngineConfigChanged();
  void refreshSizeNote();
});

// Engines may read different repositories, so switching one moves the other
// unless the user has typed their own.
els.quant.addEventListener('change', () => {
  markEngineConfigChanged();
  void refreshSizeNote();
});

els.engine.addEventListener('change', () => {
  updateEngineUi();
  const current = els.repo.value.trim();
  if (!current || current === defaultRepo('ggml') || current === defaultRepo('onnx')) {
    els.repo.value = engineDefinition(engine()).defaultRepo;
  }
  markEngineConfigChanged();
  void refreshSizeNote();
});

els.banner.src = bannerUrl;
els.text.value = DEFAULT_TEXT;
refreshSizeNote();

// Voices installed on an earlier visit. Restored before the model loads, so
// they are in the picker and playable from the first moment the page is up.
loadVoices().then((stored) => {
  if (!stored.length) return;
  for (const voice of stored) localVoices.set(localValue(voice.name), voice);
  renderVoiceOptions();
  renderVoiceList();
  els.voice.disabled = false;
});
loadSampleTexts().then((loaded) => {
  samples = loaded;
  renderTemplates();
});
