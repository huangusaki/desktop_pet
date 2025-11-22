/**
 * Presets API Client
 */
import axios from 'axios';

const BASE_URL = 'http://localhost:8765/api';

export interface Preset {
    id: string;
    name: string;
    avatar_filename: string;
    bot_name: string;
    bot_persona: string;
    speech_pattern: string;
    constraints: string;
    format_example: string;
    created_at: string;
    updated_at: string;
    is_active: boolean;
}

export interface PresetCreateData {
    name: string;
    bot_name: string;
    bot_persona: string;
    speech_pattern?: string;
    constraints?: string;
    format_example?: string;
    avatar_filename?: string;
}

export interface PresetUpdateData {
    name?: string;
    bot_name?: string;
    bot_persona?: string;
    speech_pattern?: string;
    constraints?: string;
    format_example?: string;
    avatar_filename?: string;
}

export const presetsApi = {
    /**
     * Fetch all presets
     */
    async fetchAllPresets(): Promise<Preset[]> {
        const response = await axios.get(`${BASE_URL}/presets`);
        return response.data.presets;
    },

    /**
     * Fetch a single preset by ID
     */
    async fetchPreset(presetId: string): Promise<Preset> {
        const response = await axios.get(`${BASE_URL}/presets/${presetId}`);
        return response.data;
    },

    /**
     * Create a new preset
     */
    async createPreset(data: PresetCreateData): Promise<Preset> {
        const response = await axios.post(`${BASE_URL}/presets`, data);
        return response.data.preset;
    },

    /**
     * Update a preset
     */
    async updatePreset(presetId: string, data: PresetUpdateData): Promise<Preset> {
        const response = await axios.put(`${BASE_URL}/presets/${presetId}`, data);
        return response.data.preset;
    },

    /**
     * Delete a preset
     */
    async deletePreset(presetId: string): Promise<void> {
        await axios.delete(`${BASE_URL}/presets/${presetId}`);
    },

    /**
     * Activate a preset
     */
    async activatePreset(presetId: string): Promise<Preset> {
        const response = await axios.post(`${BASE_URL}/presets/${presetId}/activate`);
        return response.data.preset;
    },

    /**
     * Upload a preset avatar
     */
    async uploadAvatar(file: File): Promise<{ filename: string; url: string }> {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axios.post(`${BASE_URL}/presets/avatars/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    /**
     * Create preset from current config
     */
    async createFromCurrentConfig(): Promise<Preset> {
        const response = await axios.post(`${BASE_URL}/presets/from-current-config`);
        return response.data.preset;
    },

    /**
     * Get avatar URL
     */
    getAvatarUrl(filename: string): string {
        return `${BASE_URL}/presets/avatars/${filename}`;
    },

    /**
     * Reorder presets
     */
    async reorderPresets(presetIds: string[]): Promise<void> {
        await axios.post(`${BASE_URL}/presets/reorder`, { preset_ids: presetIds });
    },
};
