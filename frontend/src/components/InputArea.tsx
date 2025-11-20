import React, { useState, useRef } from 'react';
import { api } from '../api/client';
import { useChatStore } from '../stores/chatStore';
import { SendIcon, AttachmentIcon } from './Icons';

export const InputArea: React.FC = () => {
    const [message, setMessage] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { config, addMessage } = useChatStore();

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        try {
            const uploadPromises = Array.from(files).map(file => api.uploadFile(file));
            const results = await Promise.all(uploadPromises);
            const fileIds = results.map(r => r.file_id);
            setUploadedFiles(prev => [...prev, ...fileIds]);

            // 可选：显示上传成功的提示
            console.log('Files uploaded:', results);
        } catch (error) {
            console.error('Error uploading files:', error);
            alert('文件上传失败，请重试');
        }
    };

    const handleAttachmentClick = () => {
        fileInputRef.current?.click();
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        const trimmedMessage = message.trim();
        if (!trimmedMessage || isSending) return;

        try {
            setIsSending(true);

            // 立即添加用户消息到UI
            const userMessage = {
                id: `temp_${Date.now()}`,
                sender: config?.user_name || 'User',
                text: trimmedMessage,
                timestamp: new Date().toISOString(),
                is_user: true
            };
            addMessage(userMessage);

            // 清空输入框和文件
            setMessage('');
            const filesToSend = uploadedFiles;
            setUploadedFiles([]);

            // 发送到后端
            await api.sendMessage(trimmedMessage, filesToSend);

        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setIsSending(false);
        }
    };

    return (
        <div className="absolute bottom-6 left-0 right-0 px-4 z-20 flex justify-center">
            <form
                onSubmit={handleSubmit}
                className="
                    w-full max-w-2xl 
                    glass-strong 
                    rounded-full 
                    p-2 
                    flex items-center gap-2 
                    shadow-2xl 
                    border border-white/10
                    transition-all duration-300
                    hover:border-white/20
                "
            >
                {/* 隐藏的文件输入 */}
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept="image/*,audio/*,video/*"
                    onChange={handleFileSelect}
                    className="hidden"
                />

                {/* 附件按钮 */}
                <button
                    type="button"
                    onClick={handleAttachmentClick}
                    className="
                        p-3 
                        rounded-full 
                        text-white/60 
                        hover:text-white 
                        hover:bg-white/10 
                        transition-all 
                        duration-200
                        relative
                    "
                    title="添加附件"
                >
                    <AttachmentIcon className="w-5 h-5" />
                    {uploadedFiles.length > 0 && (
                        <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                            {uploadedFiles.length}
                        </span>
                    )}
                </button>

                {/* 输入框 */}
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    className="
                        flex-1
                        bg-transparent
                        text-white
                        placeholder-white/40
                        focus:outline-none
                        px-2
                        py-2
                        text-sm
                    "
                    placeholder={`对${config?.bot_name || '爱丽丝'}说点什么...`}
                    disabled={isSending}
                />

                {/* 发送按钮 */}
                <button
                    type="submit"
                    disabled={!message.trim() || isSending}
                    className={`
                        p-3
                        rounded-full
                        transition-all
                        duration-300
                        shadow-lg
                        flex items-center justify-center
                        ${!message.trim() || isSending
                            ? 'bg-white/5 text-white/20 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:scale-105 hover:shadow-blue-500/25'
                        }
                    `}
                >
                    <SendIcon className="w-5 h-5" />
                </button>
            </form>
        </div>
    );
};
