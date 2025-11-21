import React, { useEffect, useRef } from 'react';
import { Message } from './Message';
import { useChatStore } from '../stores/chatStore';

export const MessageList: React.FC = () => {
    const { messages, isTyping, config } = useChatStore();
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // 自动滚动到底部
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    return (
        <div className="flex-1 overflow-y-auto p-4 pb-32 bg-transparent scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/20 scrollbar-track-transparent">
            <div className="max-w-3xl mx-auto">
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-400 dark:text-white/40 text-center py-20">
                        <p className="text-lg mb-2 font-medium">还没有聊天记录</p>
                        <p className="text-sm">开始和 {config?.bot_name || 'Bot'} 聊天吧！</p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {messages.map((message) => (
                            <Message
                                key={message.id}
                                message={message}
                            />
                        ))}
                    </div>
                )}

                {/* 打字指示器 */}
                {isTyping && (
                    <div className="flex items-start gap-3 mb-4">
                        <img
                            src={config ? config.bot_avatar : ''}
                            className="w-10 h-10 rounded-full flex-shrink-0 border-2 border-slate-200 dark:border-white/20 shadow-md"
                            alt="Bot Avatar"
                        />
                        <div className="bg-white dark:bg-white/10 backdrop-blur-md text-slate-800 dark:text-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg border border-slate-200 dark:border-white/5">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 bg-slate-400 dark:bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                <span className="w-2 h-2 bg-slate-400 dark:bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                <span className="w-2 h-2 bg-slate-400 dark:bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>
        </div>
    );
};
