import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConfigStore } from '../stores/useConfigStore';
import type { ConfigItem } from '../api/configApi';
import { PasswordInput } from './PasswordInput';

// Category names and labels
const CATEGORIES = [
    { id: 'basic', label: '基础设置' },
    { id: 'api', label: 'API配置' },
    { id: 'database', label: '数据库' },
    { id: 'memory', label: '记忆系统' },
    { id: 'screen', label: '屏幕分析' },
    { id: 'tts', label: '语音合成' },
    { id: 'agent', label: '智能体模式' },
    { id: 'avatars', label: '头像设置' },
];

export const ConfigPage: React.FC = () => {
    const navigate = useNavigate();
    const {
        configs,
        currentCategory,
        isLoading,
        isSaving,
        hasUnsavedChanges,
        fetchConfigs,
        setCurrentCategory,
        updateConfigValue,
        saveConfigs,
        resetChanges,
        restartApp,
    } = useConfigStore();

    // Load configs on mount
    useEffect(() => {
        fetchConfigs();
    }, [fetchConfigs]);

    const renderConfigInput = (item: ConfigItem) => {
        const handleChange = (value: any) => {
            updateConfigValue(item.section, item.key, value);
        };

        const commonInputClass = `
      w-full px-4 py-2 rounded-lg 
      bg-white/5 border border-white/10
      text-white placeholder-white/30
      focus:outline-none focus:border-blue-500/50
      transition-all duration-200
    `;

        // Boolean type - Toggle switch
        if (item.value_type === 'bool') {
            return (
                <div className="flex items-center space-x-3">
                    <button
                        onClick={() => handleChange(!item.value)}
                        className={`
              relative inline-flex h-6 w-11 items-center rounded-full
              transition-colors duration-200 ease-in-out
              ${item.value ? 'bg-blue-500' : 'bg-white/10'}
            `}
                    >
                        <span
                            className={`
                inline-block h-4 w-4 transform rounded-full bg-white
                transition duration-200 ease-in-out
                ${item.value ? 'translate-x-6' : 'translate-x-1'}
              `}
                        />
                    </button>
                    <span className="text-sm text-white/60">
                        {item.value ? '启用' : '禁用'}
                    </span>
                </div>
            );
        }

        // Number types
        if (item.value_type === 'int' || item.value_type === 'float') {
            return (
                <input
                    type="number"
                    value={item.value || ''}
                    onChange={(e) => {
                        const val = e.target.value;
                        handleChange(item.value_type === 'int' ? parseInt(val) || 0 : parseFloat(val) || 0);
                    }}
                    step={item.value_type === 'float' ? '0.01' : '1'}
                    className={commonInputClass}
                />
            );
        }

        // Text area for long text
        if (item.value_type === 'text') {
            return (
                <textarea
                    value={item.value || ''}
                    onChange={(e) => handleChange(e.target.value)}
                    rows={4}
                    className={commonInputClass}
                />
            );
        }

        // Password field
        if (item.value_type === 'password') {
            return (
                <PasswordInput
                    value={item.value || ''}
                    onChange={handleChange}
                    className={commonInputClass}
                />
            );
        }

        // Default: string type
        return (
            <input
                type="text"
                value={item.value || ''}
                onChange={(e) => handleChange(e.target.value)}
                className={commonInputClass}
            />
        );
    };

    if (isLoading && !configs) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-white text-xl">加载配置中...</div>
            </div>
        );
    }

    const currentConfigs = configs?.[currentCategory] || [];

    return (
        <div className="flex h-screen bg-transparent">
            {/* Left Sidebar - Category Navigation */}
            <div className="w-64 glass-strong border-r border-white/10 p-6 flex flex-col">
                <div className="mb-8">
                    <button
                        onClick={() => navigate('/')}
                        className="
              text-white/60 hover:text-white
              transition-colors duration-200
              flex items-center gap-2
            "
                    >
                        <span>←</span>
                        <span>返回聊天</span>
                    </button>
                </div>

                <h2 className="text-xl font-bold text-white mb-6">配置管理</h2>

                {/* Category List */}
                <nav className="flex-1 space-y-2">
                    {CATEGORIES.map((cat) => (
                        <button
                            key={cat.id}
                            onClick={() => setCurrentCategory(cat.id)}
                            className={`
                w-full text-left px-4 py-3 rounded-lg
                transition-all duration-200
                ${currentCategory === cat.id
                                    ? 'bg-blue-500/20 text-white border border-blue-500/30'
                                    : 'text-white/60 hover:text-white hover:bg-white/5'
                                }
              `}
                        >
                            {cat.label}
                        </button>
                    ))}
                </nav>

                {/* Actions */}
                <div className="mt-auto space-y-3">
                    <button
                        onClick={restartApp}
                        className="
              w-full px-4 py-2 rounded-lg
              bg-purple-500/20 text-purple-300
              hover:bg-purple-500/30
              border border-purple-500/30
              transition-all duration-200
            "
                    >
                        重启应用
                    </button>
                </div>
            </div>

            {/* Right Content Area */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="glass-strong border-b border-white/10 p-6 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-white">
                            {CATEGORIES.find(c => c.id === currentCategory)?.label}
                        </h1>
                        <p className="text-white/50 text-sm mt-1">
                            修改后需要保存并重启应用
                        </p>
                    </div>

                    <div className="flex gap-3">
                        {hasUnsavedChanges && (
                            <button
                                onClick={resetChanges}
                                disabled={isSaving}
                                className="
                  px-6 py-2 rounded-lg
                  bg-white/5 text-white/60
                  hover:bg-white/10 hover:text-white
                  border border-white/10
                  transition-all duration-200
                  disabled:opacity-50 disabled:cursor-not-allowed
                "
                            >
                                重置
                            </button>
                        )}

                        <button
                            onClick={saveConfigs}
                            disabled={!hasUnsavedChanges || isSaving}
                            className={`
                px-6 py-2 rounded-lg
                transition-all duration-200
                ${hasUnsavedChanges && !isSaving
                                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:scale-105 shadow-lg'
                                    : 'bg-white/10 text-white/30 cursor-not-allowed'
                                }
              `}
                        >
                            {isSaving ? '保存中...' : '保存配置'}
                        </button>
                    </div>
                </div>

                {/* Config Items */}
                <div className="flex-1 overflow-y-auto p-8">
                    <div className="max-w-4xl mx-auto space-y-6">
                        {currentConfigs.map((item) => (
                            <div
                                key={`${item.section}.${item.key}`}
                                className="glass rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all duration-200"
                            >
                                <div className="mb-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <label className="text-white font-medium">
                                            {item.label}
                                        </label>
                                        {item.required && (
                                            <span className="text-red-400 text-sm">*</span>
                                        )}
                                    </div>
                                    <p className="text-white/50 text-sm">
                                        {item.description}
                                    </p>
                                </div>

                                {renderConfigInput(item)}

                                <div className="mt-2 text-xs text-white/30">
                                    {item.section}.{item.key}
                                    {item.default_value !== undefined && (
                                        <span className="ml-2">
                                            (默认: {String(item.default_value)})
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
