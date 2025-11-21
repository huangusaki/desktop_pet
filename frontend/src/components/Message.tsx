import React from 'react';
import type { ChatMessage as ChatMessageType } from '../types/chat';
import { api } from '../api/client';
import { motion } from 'framer-motion';

interface MessageProps {
    message: ChatMessageType;
    botAvatar?: string;
    userAvatar?: string;
}

export const Message: React.FC<MessageProps> = ({ message, botAvatar, userAvatar }) => {
    const isUser = message.is_user;
    const avatar = isUser ? userAvatar : botAvatar;

    const bubbleClass = isUser
        ? 'bg-gradient-to-br from-green-100 to-emerald-100 dark:from-blue-600/90 dark:to-purple-600/90 rounded-2xl rounded-tr-sm text-slate-800 dark:text-white shadow-sm'
        : 'bg-white dark:bg-white/10 rounded-2xl rounded-tl-sm text-slate-800 dark:text-gray-100 shadow-sm border border-slate-100 dark:border-white/5';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
        >
            {/* 头像 */}
            <motion.div
                whileHover={{ scale: 1.1, rotate: isUser ? -5 : 5 }}
                className="flex-shrink-0 mt-0"
            >
                <img
                    src={avatar || (isUser ? api.getAvatarUrl('user') : api.getAvatarUrl('bot'))}
                    className={`w-10 h-10 rounded-full object-cover border-2 border-slate-200 dark:border-white/20 shadow-lg`}
                    alt={isUser ? 'User Avatar' : 'Bot Avatar'}
                />
            </motion.div>

            {/* 消息内容列 */}
            <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[75%]`}>
                {/* 发送者名字 - 独立一行，位于气泡上方 */}
                <span className={`text-[12px] text-slate-500 dark:text-white/60 font-medium uppercase tracking-wider mb-1 px-1 ${isUser ? 'text-right' : 'text-left'}`}>
                    {message.sender}
                </span>

                {/* 气泡和时间行 */}
                <div className={`flex items-end gap-2 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                    {/* 消息气泡 */}
                    <motion.div
                        whileHover={{ scale: 1.01 }}
                        className={`
                            relative
                            px-4 py-2
                            text-sm md:text-base
                            shadow-lg backdrop-blur-md
                            break-words
                            border border-slate-200 dark:border-white/10
                            ${bubbleClass}
                        `}
                    >
                        {message.text}
                    </motion.div>

                    {/* 时间戳 - 位于气泡底部外侧 */}
                    <span className="text-[10px] text-slate-400 dark:text-white/40 mb-1 whitespace-nowrap flex-shrink-0">
                        {new Date(message.timestamp).toLocaleTimeString('zh-CN', {
                            hour: '2-digit',
                            minute: '2-digit'
                        })}
                    </span>
                </div>
            </div>
        </motion.div>
    );
};
