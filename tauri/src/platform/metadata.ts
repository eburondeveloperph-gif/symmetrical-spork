import { getVersion } from '@tauri-apps/api/app';
import type { PlatformMetadata } from '@/platform/types';
import { hasTauriRuntime } from './runtime';

export const tauriMetadata: PlatformMetadata = {
  async getVersion(): Promise<string> {
    if (!hasTauriRuntime()) {
      return import.meta.env.VITE_APP_VERSION || '0.1.0';
    }

    try {
      return await getVersion();
    } catch (error) {
      console.error('Failed to get version:', error);
      return '0.1.0';
    }
  },
  get isTauri(): boolean {
    return hasTauriRuntime();
  },
};
