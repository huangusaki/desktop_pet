import axios from 'axios';
import type { ChatMessage, Config } from '../types/chat';

const API_BASE = 'http://localhost:8765/api';

export const api = {
    // 获取配置
    async getConfig(): Promise<Config> {
        const { data } = await axios.get<Config>(`${API_BASE}/config`);
        return data;
    },

    // 获取聊天历史
    async getChatHistory(limit: number = 20): Promise<ChatMessage[]> {
        const { data } = await axios.get<{ messages: ChatMessage[] }>(
            `${API_BASE}/chat/history`,
            { params: { limit } }
        );
        return data.messages;
    },

    // 发送消息
    async sendMessage(text: string, files: string[] = []): Promise<{ status: string; message_id: string }> {
        const { data } = await axios.post<{ status: string; message_id: string }>(
            `${API_BASE}/chat/send`,
            { text, files }
        );
        return data;
    },

    // 上传文件
    async uploadFile(file: File): Promise<{ file_id: string; filename: string; mime_type: string; url: string }> {
        const formData = new FormData();
        formData.append('file', file);
        const { data } = await axios.post(`${API_BASE}/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return data;
    },

    // 切换Agent模式
    async toggleAgentMode(enabled: boolean): Promise<{ agent_mode: boolean }> {
        const { data } = await axios.post<{ agent_mode: boolean }>(
            `${API_BASE}/agent/toggle`,
            { enabled }
        );
        return data;
    },

    // 获取头像URL
    getAvatarUrl(type: 'bot' | 'user'): string {
        return `${API_BASE}/avatar/${type}`;
    },
};
