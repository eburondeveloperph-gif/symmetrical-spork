import { invoke } from '@tauri-apps/api/core';
import type { PlatformAudio, AudioDevice } from '@/platform/types';
import { hasTauriRuntime, requireTauriRuntime } from './runtime';

export const tauriAudio: PlatformAudio = {
  isSystemAudioSupported(): boolean {
    return hasTauriRuntime();
  },

  async startSystemAudioCapture(maxDurationSecs: number): Promise<void> {
    requireTauriRuntime('System audio capture');

    await invoke('start_system_audio_capture', {
      maxDurationSecs,
    });
  },

  async stopSystemAudioCapture(): Promise<Blob> {
    requireTauriRuntime('System audio capture');

    const base64Data = await invoke<string>('stop_system_audio_capture');

    // Convert base64 to Blob
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    return new Blob([bytes], { type: 'audio/wav' });
  },

  async listOutputDevices(): Promise<AudioDevice[]> {
    requireTauriRuntime('Listing audio devices');
    return await invoke<AudioDevice[]>('list_audio_output_devices');
  },

  async playToDevices(audioData: Uint8Array, deviceIds: string[]): Promise<void> {
    requireTauriRuntime('Native audio playback');

    await invoke('play_audio_to_devices', {
      audioData: Array.from(audioData),
      deviceIds,
    });
  },

  stopPlayback(): void {
    if (!hasTauriRuntime()) {
      return;
    }

    invoke('stop_audio_playback').catch((error) => {
      console.error('Failed to stop audio playback:', error);
    });
  },
};
