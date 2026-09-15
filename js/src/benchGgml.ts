/** UI and report export for the isolated browser benchmark workers. */
import {
  BenchConfig, BenchResult, BenchWorkerResponse, backendLabel,
} from './benchTypes';

interface NavigatorWithHardware extends Navigator {
  deviceMemory?: number;
  userAgentData?: { brands?: Array<{ brand: string; version: string }>; mobile?: boolean };
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const log = (message: string) => {
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  $('log').textContent += `[${now}] ${message}\n`;
};
const trimBase = (value: string) => value.trim().replace(/\/$/, '');
const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, Number.isFinite(value) ? Math.floor(value) : min));

const params = new URLSearchParams(location.search);
$<HTMLInputElement>('model-base').value = params.get('model')
  ?? 'https://huggingface.co/zeroweight-ai/ZeroTTS/resolve/main';
$<HTMLInputElement>('gguf-base').value = params.get('ggufBase')
  ?? 'https://huggingface.co/zeroweight-ai/ZeroTTS-GGUF/resolve/main/gguf';
$('log').textContent = '';

async function inspectSystem() {
  const nav = navigator as NavigatorWithHardware;
  const report: Record<string, unknown> = {
    capturedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    languages: navigator.languages,
    userAgentData: nav.userAgentData ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemoryGiB: nav.deviceMemory ?? null,
    isSecureContext: self.isSecureContext,
    crossOriginIsolated: self.crossOriginIsolated,
    screen: `${screen.width}x${screen.height} @ ${devicePixelRatio}x`,
  };
  return report;
}

const systemPromise = inspectSystem();
systemPromise.then((system) => { $('system').textContent = JSON.stringify(system, null, 2); });

function readConfig(): BenchConfig {
  return {
    backend: $<HTMLSelectElement>('backend').value as BenchConfig['backend'],
    gguf: $<HTMLSelectElement>('gguf').value,
    threads: clamp(Number($<HTMLInputElement>('threads').value), 1, 16),
    frames: clamp(Number($<HTMLInputElement>('frames').value), 10, 500),
    runs: clamp(Number($<HTMLInputElement>('runs').value), 1, 10),
    seed: clamp(Number($<HTMLInputElement>('seed').value), 0, 0x7fffffff),
    voice: $<HTMLInputElement>('voice').value.trim(),
    text: $<HTMLTextAreaElement>('text').value.trim(),
    modelBase: trimBase($<HTMLInputElement>('model-base').value),
    ggufBase: trimBase($<HTMLInputElement>('gguf-base').value),
  };
}

function validate(config: BenchConfig): void {
  if (!config.text) throw new Error('Văn bản không được để trống');
  if (!config.voice) throw new Error('Tên giọng không được để trống');
  if (!config.modelBase) throw new Error('Model base không được để trống');
  if (!config.ggufBase && config.backend !== 'onnx-wasm') {
    throw new Error('GGUF base không được để trống');
  }
}

let cancelActive: (() => void) | null = null;
let stopRequested = false;
let running = false;

function runIsolated(config: BenchConfig): Promise<BenchResult> {
  try {
    validate(config);
  } catch (error) {
    return Promise.reject(error);
  }
  return new Promise((resolve) => {
    const worker = new Worker(new URL('./benchWorker.ts', import.meta.url), {
      type: 'module', name: 'zerotts-benchmark',
    });
    let settled = false;
    let cancelThis: () => void;
    const finish = (result: BenchResult) => {
      if (settled) return;
      settled = true;
      if (cancelActive === cancelThis) cancelActive = null;
      worker.terminate();
      resolve(result);
    };
    cancelThis = () => finish({
      backend: config.backend,
      label: backendLabel(config),
      config,
      status: 'error',
      error: 'Đã dừng theo yêu cầu',
    });
    cancelActive = cancelThis;
    worker.onmessage = (event: MessageEvent<BenchWorkerResponse>) => {
      if (event.data.type === 'log') log(event.data.message);
      else finish(event.data.result);
    };
    worker.onerror = (event) => finish({
      backend: config.backend,
      label: backendLabel(config),
      config,
      status: 'error',
      error: event.message || 'Benchmark worker bị dừng',
    });
    worker.postMessage({ type: 'run', config });
  });
}

const results: BenchResult[] = [];

function renderResults(): void {
  const body = $<HTMLTableSectionElement>('result-body');
  body.replaceChildren();
  if (!results.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 8;
    cell.textContent = 'Chưa có kết quả.';
    return;
  }
  for (const result of results) {
    const row = body.insertRow();
    const add = (value: string) => { row.insertCell().textContent = value; };
    add(result.label);
    if (result.status === 'error') {
      for (let i = 0; i < 6; i++) add('—');
      const statusCell = row.insertCell();
      statusCell.textContent = result.error ?? 'Lỗi';
      statusCell.className = 'error';
      continue;
    }
    const times = result.measurements?.map((run) => run.wallMs) ?? [];
    add(`${(result.loadMs! / 1000).toFixed(2)} s`);
    add(`${(result.warmupMs! / 1000).toFixed(2)} s`);
    add(`${(result.medianMs! / 1000).toFixed(2)} s`);
    add(`${result.medianRealtime!.toFixed(2)}x`);
    add(`${(Math.min(...times) / 1000).toFixed(2)}–${(Math.max(...times) / 1000).toFixed(2)} s`);
    add(result.reproducible ? 'Có' : `Không · lệch ${result.driftFromFirst?.join('/')}`);
    const statusCell = row.insertCell();
    statusCell.textContent = 'Đạt';
    statusCell.className = 'ok';
  }
}

async function makeReport() {
  return {
    schemaVersion: 1,
    exportedAt: new Date().toISOString(),
    system: await systemPromise,
    results,
  };
}

async function runCases(configs: BenchConfig[]): Promise<BenchResult[]> {
  if (running) throw new Error('Một benchmark khác đang chạy');
  running = true;
  stopRequested = false;
  const buttons = Array.from(document.querySelectorAll('button')) as HTMLButtonElement[];
  buttons.forEach((button) => { button.disabled = true; });
  $<HTMLButtonElement>('stop').disabled = false;
  const startedAt = results.length;
  try {
    for (const config of configs) {
      $('status').textContent = `Đang chạy ${backendLabel(config)}…`;
      const result = await runIsolated(config).catch((error) => ({
        backend: config.backend,
        label: backendLabel(config),
        config,
        status: 'error' as const,
        error: (error as Error)?.message ?? String(error),
      }));
      if (result.status === 'error') log(`${result.label}: LỖI · ${result.error}`);
      results.push(result);
      renderResults();
      if (stopRequested) break;
    }
    const completed = results.length - startedAt;
    $('status').textContent = stopRequested
      ? `Đã dừng sau ${completed} cấu hình.`
      : `Hoàn tất ${completed} cấu hình.`;
    return results.slice(startedAt);
  } finally {
    running = false;
    cancelActive = null;
    $<HTMLButtonElement>('run-selected').disabled = false;
    $<HTMLButtonElement>('run-all').disabled = false;
    $<HTMLButtonElement>('stop').disabled = true;
    $<HTMLButtonElement>('clear').disabled = false;
    const hasResults = results.length > 0;
    $<HTMLButtonElement>('copy').disabled = !hasResults;
    $<HTMLButtonElement>('download').disabled = !hasResults;
  }
}

$('run-selected').addEventListener('click', () => { void runCases([readConfig()]); });
$('run-all').addEventListener('click', () => {
  const base = readConfig();
  void runCases([
    { ...base, backend: 'ggml-cpu' },
    { ...base, backend: 'onnx-wasm' },
  ]);
});
$('stop').addEventListener('click', () => {
  stopRequested = true;
  cancelActive?.();
  $<HTMLButtonElement>('stop').disabled = true;
  $('status').textContent = 'Đang dừng benchmark…';
});
$('clear').addEventListener('click', () => {
  results.splice(0);
  $('log').textContent = '';
  $('status').textContent = 'Đã xoá kết quả.';
  $<HTMLButtonElement>('copy').disabled = true;
  $<HTMLButtonElement>('download').disabled = true;
  renderResults();
});
$('copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(await makeReport(), null, 2));
    $('status').textContent = 'Đã sao chép JSON.';
  } catch (error) {
    $('status').textContent = `Không thể sao chép: ${(error as Error).message}`;
  }
});
$('download').addEventListener('click', async () => {
  const blob = new Blob([JSON.stringify(await makeReport(), null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `zerotts-browser-benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
});

declare global {
  interface Window {
    zbench: {
      run(overrides?: Partial<BenchConfig>): Promise<BenchResult[]>;
      all(overrides?: Partial<BenchConfig>): Promise<BenchResult[]>;
      report(): ReturnType<typeof makeReport>;
      results: BenchResult[];
    };
  }
}

window.zbench = {
  run: (overrides = {}) => runCases([{ ...readConfig(), ...overrides }]),
  all: (overrides = {}) => {
    const config = { ...readConfig(), ...overrides };
    return runCases([
      { ...config, backend: 'ggml-cpu' },
      { ...config, backend: 'onnx-wasm' },
    ]);
  },
  report: makeReport,
  results,
};
