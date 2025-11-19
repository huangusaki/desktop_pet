import { create } from 'zustand';
import type { ChatMessage, Config } from '../types/chat';

interface ChatStore {
    messages: ChatMessage[];
    config: Config | null;
    isTyping: boolean;
    isConnected: boolean;

    addMessage: (message: ChatMessage) => void;
    setMessages: (messages: ChatMessage[]) => void;
    setConfig: (config: Config) => void;
    setTyping: (isTyping: boolean) => void;
    setConnected: (isConnected: boolean) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
    messages: [],
    config: null,
    isTyping: false,
    isConnected: false,

    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
    })),

    setMessages: (messages) => set({ messages }),
    setConfig: (config) => set({ config }),
    setTyping: (isTyping) => set({ isTyping }),
    setConnected: (isConnected) => set({ isConnected }),
}));
