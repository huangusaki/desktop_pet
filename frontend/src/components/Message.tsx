import React from 'react';
import type { ChatMessage as ChatMessageType } from '../types/chat';
import { api } from '../api/client';
import { motion } from 'framer-motion';

interface MessageProps {
    message: ChatMessageType;
    petAvatar?: string;
    userAvatar?: string;
}

export const Message: React.FC<MessageProps> = ({ message, petAvatar, userAvatar }) => {
    const isUser = message.is_user;
    const avatar = isUser ? userAvatar : petAvatar;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className={`flex items-end gap-3 mb-6 ${isUser ? 'flex-row-reverse' : ''}`}
        >
            {/* 头像 */}
            <motion.div
                whileHover={{ scale: 1.1, rotate: isUser ? -5 : 5 }}
            >
                <img
                    src={avatar || (isUser ? api.getAvatarUrl('user') : api.getAvatarUrl('pet'))}
                    className="w-10 h-10 rounded-full object-cover border-2 border-white/10 shadow-lg"
                    alt={isUser ? 'User Avatar' : 'Pet Avatar'}
                />
            </motion.div>

            {/* 消息内容 */}
            <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[75%]`}>
                {/* 发送者名字 */}
                <span className={`text-[10px] text-white/40 mb-1 px-1 font-medium uppercase tracking-wider ${isUser ? 'text-right' : 'text-left'}`}>
                    {message.sender}
                </span>

                {/* 消息气泡 */}
                <motion.div
                    whileHover={{ scale: 1.01 }}
                    className={`
                        relative
                        px-5 py-3
                        text-sm md:text-base
                        shadow-lg backdrop-blur-md
                        break-words
                        border border-white/5
                        ${isUser
                            ? 'bg-gradient-to-br from-blue-600/90 to-purple-600/90 rounded-2xl rounded-tr-sm text-white'
                            : 'bg-white/10 rounded-2xl rounded-tl-sm text-gray-100'
                        }
                    `}
                >
                    {message.text}
                </motion.div>

                {/* 时间戳 */}
                <span className="text-[10px] text-white/30 mt-1 px-1">
                    {new Date(message.timestamp).toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit'
                    })}
                </span>
            </div>
        </motion.div>
    );
};
