/**
 * Presets Store - Zustand state management for presets
 */
import { create } from 'zustand';
import { presetsApi, type Preset, type PresetCreateData, type PresetUpdateData } from '../api/presetsApi';

// LocalStorage key for last selected preset
const LAST_SELECTED_PRESET_KEY = 'arisu_last_selected_preset_id';

interface PresetsState {
    presets: Preset[];
    activePreset: Preset | null;
    lastSelectedPresetId: string | null;
    isLoading: boolean;
    error: string | null;

    // Actions
    fetchPresets: () => Promise<void>;
    createPreset: (data: PresetCreateData) => Promise<Preset | null>;
    updatePreset: (presetId: string, data: PresetUpdateData) => Promise<Preset | null>;
    deletePreset: (presetId: string) => Promise<boolean>;
    activatePreset: (presetId: string) => Promise<boolean>;
    uploadAvatar: (file: File) => Promise<string | null>;
    createFromCurrentConfig: () => Promise<Preset | null>;
    setLastSelectedPresetId: (presetId: string | null) => void;
    getLastSelectedPresetId: () => string | null;
    reorderPresets: (newPresets: Preset[]) => Promise<void>;
}

export const usePresetsStore = create<PresetsState>((set, get) => ({
    presets: [],
    activePreset: null,
    lastSelectedPresetId: null,
    isLoading: false,
    error: null,

    fetchPresets: async () => {
        set({ isLoading: true, error: null });
        try {
            const presets = await presetsApi.fetchAllPresets();
            const activePreset = presets.find(p => p.is_active) || null;
            set({ presets, activePreset, isLoading: false });
        } catch (error: any) {
            set({ error: error.message || '获取预设列表失败', isLoading: false });
        }
    },

    createPreset: async (data: PresetCreateData) => {
        set({ isLoading: true, error: null });
        try {
            const preset = await presetsApi.createPreset(data);
            const { presets } = get();
            set({ presets: [...presets, preset], isLoading: false });
            return preset;
        } catch (error: any) {
            set({ error: error.message || '创建预设失败', isLoading: false });
            return null;
        }
    },

    updatePreset: async (presetId: string, data: PresetUpdateData) => {
        set({ isLoading: true, error: null });
        try {
            const updatedPreset = await presetsApi.updatePreset(presetId, data);
            const { presets } = get();
            const newPresets = presets.map(p => p.id === presetId ? updatedPreset : p);
            set({ presets: newPresets, isLoading: false });
            return updatedPreset;
        } catch (error: any) {
            set({ error: error.message || '更新预设失败', isLoading: false });
            return null;
        }
    },

    deletePreset: async (presetId: string) => {
        set({ isLoading: true, error: null });
        try {
            await presetsApi.deletePreset(presetId);
            const { presets } = get();
            const newPresets = presets.filter(p => p.id !== presetId);

            // Clear last selected if it was the deleted preset
            const lastSelectedId = localStorage.getItem(LAST_SELECTED_PRESET_KEY);
            if (lastSelectedId === presetId) {
                localStorage.removeItem(LAST_SELECTED_PRESET_KEY);
                set({ lastSelectedPresetId: null });
            }

            set({ presets: newPresets, isLoading: false });
            return true;
        } catch (error: any) {
            set({ error: error.message || '删除预设失败', isLoading: false });
            return false;
        }
    },

    activatePreset: async (presetId: string) => {
        set({ isLoading: true, error: null });
        try {
            const activatedPreset = await presetsApi.activatePreset(presetId);
            const { presets } = get();
            const newPresets = presets.map(p => ({
                ...p,
                is_active: p.id === presetId
            }));
            set({
                presets: newPresets,
                activePreset: activatedPreset,
                isLoading: false
            });
            return true;
        } catch (error: any) {
            set({ error: error.message || '激活预设失败', isLoading: false });
            return false;
        }
    },

    uploadAvatar: async (file: File) => {
        set({ isLoading: true, error: null });
        try {
            const result = await presetsApi.uploadAvatar(file);
            set({ isLoading: false });
            return result.filename;
        } catch (error: any) {
            set({ error: error.message || '上传头像失败', isLoading: false });
            return null;
        }
    },

    createFromCurrentConfig: async () => {
        set({ isLoading: true, error: null });
        try {
            const preset = await presetsApi.createFromCurrentConfig();
            const { presets } = get();
            set({ presets: [...presets, preset], isLoading: false });
            return preset;
        } catch (error: any) {
            set({ error: error.message || '保存当前配置失败', isLoading: false });
            return null;
        }
    },

    setLastSelectedPresetId: (presetId: string | null) => {
        if (presetId) {
            localStorage.setItem(LAST_SELECTED_PRESET_KEY, presetId);
        } else {
            localStorage.removeItem(LAST_SELECTED_PRESET_KEY);
        }
        set({ lastSelectedPresetId: presetId });
    },

    getLastSelectedPresetId: () => {
        return localStorage.getItem(LAST_SELECTED_PRESET_KEY);
    },

    reorderPresets: async (newPresets: Preset[]) => {
        // Optimistic update
        set({ presets: newPresets });

        try {
            const presetIds = newPresets.map(p => p.id);
            await presetsApi.reorderPresets(presetIds);
        } catch (error: any) {
            // Revert on error (reload from server)
            const { fetchPresets } = get();
            await fetchPresets();
            set({ error: error.message || '重新排序失败' });
        }
    },
}));
