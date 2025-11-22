import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageList } from './MessageList';
import { InputArea } from './InputArea';
import { useChatStore } from '../stores/chatStore';
import { api } from '../api/client';
import { SettingsIcon, ContactsIcon } from './Icons';

export const ChatWindow: React.FC = () => {
    const { setConfig, setMessages } = useChatStore();
    const navigate = useNavigate();

    // Note: WebSocket is now initialized at App level, not here

    // 加载初始数据
    useEffect(() => {
        const loadData = async () => {
            try {
                // 加载配置
                const configData = await api.getConfig();
                setConfig(configData);

                // 加载历史消息
                const messages = await api.getChatHistory(20);
                setMessages(messages);
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        };

        loadData();
    }, [setConfig, setMessages]);

    return (
        <div className="flex flex-col h-screen bg-transparent relative">
            {/* Header with Buttons */}
            <div className="absolute top-4 right-4 z-30 flex flex-col gap-2">
                {/* Settings Button */}
                <button
                    onClick={() => navigate('/config')}
                    className="
                        p-3 
                        rounded-full 
                        glass
                        text-slate-500 
                        hover:text-slate-800 
                        hover:bg-white/50 
                        dark:text-white/60 
                        dark:hover:text-white 
                        dark:hover:bg-white/10 
                        transition-all 
                        duration-200
                        shadow-lg
                    "
                    title="设置"
                >
                    <SettingsIcon className="w-5 h-5" />
                </button>

                {/* Friends Button */}
                <button
                    onClick={() => navigate('/friends')}
                    className="
                        p-3 
                        rounded-full 
                        glass
                        text-slate-500 
                        hover:text-slate-800 
                        hover:bg-white/50 
                        dark:text-white/60 
                        dark:hover:text-white 
                        dark:hover:bg-white/10 
                        transition-all 
                        duration-200
                        shadow-lg
                    "
                    title="好友"
                >
                    <ContactsIcon className="w-5 h-5" />
                </button>
            </div>

            {/* 消息列表 */}
            <MessageList />

            {/* 输入区域 */}
            <InputArea />
        </div>
    );
};
