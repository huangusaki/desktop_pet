import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageList } from './MessageList';
import { InputArea } from './InputArea';
import { useWebSocket } from '../hooks/useWebSocket';
import { useChatStore } from '../stores/chatStore';
import { api } from '../api/client';
import { SettingsIcon } from './Icons';

export const ChatWindow: React.FC = () => {
    const { setConfig, setMessages } = useChatStore();
    const navigate = useNavigate();

    // 初始化WebSocket
    useWebSocket();

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
            {/* Header with Settings Button */}
            <div className="absolute top-4 right-4 z-30">
                <button
                    onClick={() => navigate('/config')}
                    className="
                        p-3 
                        rounded-full 
                        glass
                        text-white/60 
                        hover:text-white 
                        hover:bg-white/10 
                        transition-all 
                        duration-200
                        shadow-lg
                    "
                    title="设置"
                >
                    <SettingsIcon className="w-5 h-5" />
                </button>
            </div>

            {/* 消息列表 */}
            <MessageList />

            {/* 输入区域 */}
            <InputArea />
        </div>
    );
};
