/**
 * Backend detection — queries the Rust side via `get_gpu_info` to determine
 * which execution backend is available, and exposes a BackendStatus for the
 * UI indicator.
 */

import type { GpuInfo } from './poker';
import { isTauri } from './tauriEnv';

export type BackendMode = 'gpu' | 'cpu' | 'detecting' | 'demo';

export interface BackendStatus {
  mode: BackendMode;
  /** Human-readable description, e.g. "NVIDIA RTX 5090 (32 GB, CC 12.0)" */
  description: string;
  /** True when the backend is ready to run real solves. */
  functional: boolean;
  /** Raw GPU info from Tauri (only set when in Tauri mode). */
  gpuInfo?: GpuInfo;
}

/**
 * Detect the active solver backend.
 *
 * - Browser: "demo" — heuristic fallback, no real solver.
 * - Tauri + GPU functional: "gpu" with description of the detected GPU.
 * - Tauri + GPU hardware present but backend WIP: "cpu" (autofallback),
 *   with description noting GPU was detected.
 * - Tauri + no GPU: "cpu".
 */
export async function detectBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return {
      mode: 'demo',
      description: 'Browser preview (heuristic only)',
      functional: false,
    };
  }

  try {
    const { invoke } = await import('@tauri-apps/api/core');
    const info = await invoke<GpuInfo>('get_gpu_info');

    if (info.has_cuda_gpu && info.gpu_backend_functional) {
      return {
        mode: 'gpu',
        description: info.gpu_description || 'CUDA GPU',
        functional: true,
        gpuInfo: info,
      };
    }

    if (info.has_cuda_gpu) {
      // GPU hardware present but backend not yet complete (Phase 4 WIP)
      return {
        mode: 'cpu',
        description: `CPU · ${info.gpu_description} detected (GPU backend WIP)`,
        functional: true,
        gpuInfo: info,
      };
    }

    return {
      mode: 'cpu',
      description: 'CPU (no CUDA GPU detected)',
      functional: true,
      gpuInfo: info,
    };
  } catch (err) {
    // Fallback: CPU works even if detection command fails
    return {
      mode: 'cpu',
      description: `CPU (detection failed: ${err instanceof Error ? err.message : String(err)})`,
      functional: true,
    };
  }
}
