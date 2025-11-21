/**
 * Configuration Management API Client
 */
import axios from 'axios';

const BASE_URL = 'http://localhost:8765/api';

export interface ConfigItem {
    section: string;
    key: string;
    value: any;
    value_type: string;
    default_value: any;
    label: string;
    description: string;
    category: string;
    required: boolean;
    sensitive: boolean;
    options?: string[];  // Available options for enum types
}

export interface ConfigCategory {
    [key: string]: ConfigItem[];
}

export const configApi = {
    /**
     * Fetch all configuration items organized by category
     */
    async fetchAllConfigs(): Promise<ConfigCategory> {
        const response = await axios.get(`${BASE_URL}/config/all`);
        return response.data;
    },

    /**
     * Fetch configuration items for a specific category
     */
    async fetchCategoryConfigs(category: string): Promise<ConfigItem[]> {
        const response = await axios.get(`${BASE_URL}/config/category/${category}`);
        return response.data.items;
    },

    /**
     * Update configuration values
     */
    async updateConfigs(updates: Record<string, any>): Promise<{
        success: boolean;
        updated: string[];
        errors: Record<string, string>;
    }> {
        const response = await axios.put(`${BASE_URL}/config`, { updates });
        return response.data;
    },

    /**
     * Save configuration to file
     */
    async saveConfig(): Promise<{
        success: boolean;
        message: string;
        errors?: Record<string, string>;
    }> {
        const response = await axios.post(`${BASE_URL}/config/save`);
        return response.data;
    },

    /**
     * Restart the application
     */
    async restartApp(): Promise<{
        success: boolean;
        message: string;
    }> {
        const response = await axios.post(`${BASE_URL}/system/restart`);
        return response.data;
    },
};
