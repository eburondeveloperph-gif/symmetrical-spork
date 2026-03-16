import { invoke } from '@tauri-apps/api/core';
import { listen, emit } from '@tauri-apps/api/event';
import type { PlatformLifecycle } from '@/platform/types';
import { hasTauriRuntime, requireTauriRuntime } from './runtime';

class TauriLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;
  private closeHandlerSetup: Promise<void> | null = null;

  async startServer(remote = false): Promise<string> {
    requireTauriRuntime('Starting the bundled server');

    try {
      const result = await invoke<string>('start_server', { remote });
      console.log('Server started:', result);
      this.onServerReady?.();
      return result;
    } catch (error) {
      console.error('Failed to start server:', error);
      throw error;
    }
  }

  async stopServer(): Promise<void> {
    if (!hasTauriRuntime()) {
      return;
    }

    try {
      await invoke('stop_server');
      console.log('Server stopped');
    } catch (error) {
      console.error('Failed to stop server:', error);
      throw error;
    }
  }

  async setKeepServerRunning(keepRunning: boolean): Promise<void> {
    if (!hasTauriRuntime()) {
      return;
    }

    try {
      await invoke('set_keep_server_running', { keepRunning });
    } catch (error) {
      console.error('Failed to set keep server running setting:', error);
    }
  }

  async setupWindowCloseHandler(): Promise<void> {
    if (!hasTauriRuntime()) {
      return;
    }

    if (this.closeHandlerSetup) {
      await this.closeHandlerSetup;
      return;
    }

    this.closeHandlerSetup = listen<null>('window-close-requested', async () => {
      // Import store here to avoid circular dependency
      const { useServerStore } = await import('@/stores/serverStore');
      const keepRunning = useServerStore.getState().keepServerRunningOnClose;

      // Check if server was started by this app instance
      // @ts-expect-error - accessing module-level variable from another module
      const serverStartedByApp = window.__voiceboxServerStartedByApp ?? false;

      if (!keepRunning && serverStartedByApp) {
        // Stop server before closing (only if we started it)
        try {
          await this.stopServer();
        } catch (error) {
          console.error('Failed to stop server on close:', error);
        }
      }

      // Emit event back to Rust to allow close
      await emit('window-close-allowed');
    })
      .then(() => undefined)
      .catch((error) => {
        this.closeHandlerSetup = null;
        console.error('Failed to setup window close handler:', error);
        throw error;
      });

    await this.closeHandlerSetup;
  }
}

export const tauriLifecycle = new TauriLifecycle();
