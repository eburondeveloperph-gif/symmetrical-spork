import type { PlatformFilesystem, FileFilter } from '@/platform/types';
import { hasTauriRuntime } from './runtime';

export const tauriFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, filters?: FileFilter[]) {
    if (!hasTauriRuntime()) {
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(anchor);
      return;
    }

    const { save } = await import('@tauri-apps/plugin-dialog');
    const { writeFile } = await import('@tauri-apps/plugin-fs');

    const filePath = await save({
      defaultPath: filename,
      filters: filters || [],
    });

    if (!filePath) return; // User cancelled the dialog

    const resolvedPath = typeof filePath === 'string'
      ? filePath
      : (filePath as { path: string }).path;

    if (!resolvedPath) {
      throw new Error('Failed to resolve save path from dialog');
    }

    const arrayBuffer = await blob.arrayBuffer();
    await writeFile(resolvedPath, new Uint8Array(arrayBuffer));
  },
};
