/**
 * Configuration State Management
 */
import { create } from 'zustand';
import { configApi, type ConfigCategory, type ConfigItem } from '../api/configApi';

interface ConfigState {
    // State
    configs: ConfigCategory | null;
    currentCategory: string;
    isLoading: boolean;
    isSaving: boolean;
    error: string | null;
    hasUnsavedChanges: boolean;

    // Actions
    fetchConfigs: () => Promise<void>;
    setCurrentCategory: (category: string) => void;
    updateConfigValue: (section: string, key: string, value: any) => void;
    saveConfigs: () => Promise<void>;
    resetChanges: () => Promise<void>;
    restartApp: () => Promise<void>;
}

export const useConfigStore = create<ConfigState>((set, get) => ({
    // Initial state
    configs: null,
    currentCategory: 'basic',
    isLoading: false,
    isSaving: false,
    error: null,
    hasUnsavedChanges: false,

    // Fetch all configurations
    fetchConfigs: async () => {
        set({ isLoading: true, error: null });
        try {
            const configs = await configApi.fetchAllConfigs();
            set({ configs, isLoading: false });
        } catch (error: any) {
            set({ error: error.message, isLoading: false });
        }
    },

    // Set current category
    setCurrentCategory: (category: string) => {
        set({ currentCategory: category });
    },

    // Update a config value in memory (marks as unsaved)
    updateConfigValue: (section: string, key: string, value: any) => {
        const { configs } = get();
        if (!configs) return;

        // Create deep copy and update value
        const newConfigs = { ...configs };
        Object.keys(newConfigs).forEach(category => {
            newConfigs[category] = newConfigs[category].map(item => {
                if (item.section === section && item.key === key) {
                    return { ...item, value };
                }
                return item;
            });
        });

        set({ configs: newConfigs, hasUnsavedChanges: true });
    },

    // Save configurations to backend
    saveConfigs: async () => {
        const { configs } = get();
        if (!configs) return;

        set({ isSaving: true, error: null });
        try {
            // Collect all changed values
            const updates: Record<string, any> = {};
            Object.values(configs).flat().forEach((item: ConfigItem) => {
                const configKey = `${item.section}.${item.key}`;
                updates[configKey] = item.value;
            });

            // Send update request
            const updateResult = await configApi.updateConfigs(updates);

            if (!updateResult.success) {
                throw new Error('Some configurations failed to update');
            }

            // Save to file
            const saveResult = await configApi.saveConfig();

            if (saveResult.success) {
                set({ isSaving: false, hasUnsavedChanges: false });
                alert(saveResult.message);
            } else {
                throw new Error(saveResult.message);
            }
        } catch (error: any) {
            set({ error: error.message, isSaving: false });
            alert(`保存失败: ${error.message}`);
        }
    },

    // Reset changes (reload from server)
    resetChanges: async () => {
        await get().fetchConfigs();
        set({ hasUnsavedChanges: false });
    },

    // Restart application
    restartApp: async () => {
        try {
            const confirmRestart = window.confirm(
                '确定要重启应用程序吗?\n\n重启后,您需要重新打开聊天窗口。'
            );

            if (!confirmRestart) return;

            await configApi.restartApp();
            alert('应用程序正在重启...');
        } catch (error: any) {
            alert(`重启失败: ${error.message}`);
        }
    },
}));
