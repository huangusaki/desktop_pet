import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConfigStore } from '../stores/useConfigStore';
import { useThemeStore } from '../stores/useThemeStore';
import type { ConfigItem } from '../api/configApi';
import { PasswordInput } from './PasswordInput';
import { CustomSelect } from './CustomSelect';
import {
    SettingsIcon,
    BasicIcon,
    ApiIcon,
    DatabaseIcon,
    MemoryIcon,
    ScreenIcon,
    TtsIcon,
    AgentIcon,
    AvatarIcon,
    MoonIcon,
    SunIcon,
    ArrowLeftIcon
} from './Icons';
// Category names and labels with icons
const CATEGORIES = [
    { id: 'basic', label: '基础设置', icon: BasicIcon },
    { id: 'api', label: 'API配置', icon: ApiIcon },
    { id: 'database', label: '数据库', icon: DatabaseIcon },
    { id: 'memory', label: '记忆系统', icon: MemoryIcon },
    { id: 'screen', label: '屏幕分析', icon: ScreenIcon },
    { id: 'tts', label: '语音合成', icon: TtsIcon },
    { id: 'agent', label: '智能体模式', icon: AgentIcon },
    { id: 'avatars', label: '头像设置', icon: AvatarIcon },
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
    const { theme, toggleTheme } = useThemeStore();

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
      bg-white/80 backdrop-blur-sm border border-slate-200/80
      text-slate-800 placeholder-slate-400
      focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent
      dark:bg-white/5 dark:border-white/5
      dark:text-white dark:placeholder-white/30
      dark:focus:ring-blue-400/50
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
              transition-all duration-300 ease-in-out
              ${item.value ? 'bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]' : 'bg-slate-200 dark:bg-white/10'}
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
                    <span className="text-sm text-slate-600 dark:text-white/60">
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

        // Enum type - Dropdown select
        if (item.value_type === 'enum' && item.options) {
            return (
                <CustomSelect
                    value={item.value || item.default_value}
                    onChange={(val) => handleChange(val)}
                    options={item.options}
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
                <div className="text-slate-600 dark:text-white text-xl">加载配置中...</div>
            </div>
        );
    }

    const currentConfigs = configs?.[currentCategory] || [];

    return (
        <div className="flex h-screen bg-transparent">
            {/* Left Sidebar - Category Navigation */}
            <div className="w-20 md:w-64 glass-strong border-r border-slate-200 dark:border-white/10 p-4 md:p-6 flex flex-col transition-all duration-300">
                <div className="mb-8 flex justify-center md:justify-start">
                    <button
                        onClick={() => navigate('/')}
                        className="
                            group
                            text-slate-500 hover:text-slate-800
                            dark:text-white/60 dark:hover:text-white
                            transition-colors duration-200
                            flex items-center gap-2
                            text-sm font-medium
                            hover:-translate-x-1
                        "
                        title="返回聊天"
                    >
                        <div className="p-2 rounded-full bg-white/50 dark:bg-white/5 group-hover:bg-white dark:group-hover:bg-white/10 transition-colors shadow-sm">
                            <ArrowLeftIcon className="w-5 h-5" />
                        </div>
                        <span className="hidden md:inline">返回聊天</span>
                    </button>
                </div>

                <div className="flex items-center justify-center md:justify-start gap-3 mb-6 text-slate-800 dark:text-white">
                    <SettingsIcon className="w-6 h-6 flex-shrink-0" />
                    <h2 className="text-xl font-bold hidden md:block">配置管理</h2>
                </div>

                {/* Category List */}
                <nav className="flex-1 space-y-2">
                    {CATEGORIES.map((cat) => (
                        <button
                            key={cat.id}
                            onClick={() => setCurrentCategory(cat.id)}
                            className={`
                w-full text-center md:text-left px-2 md:px-4 py-3 rounded-lg
                transition-all duration-200 flex items-center justify-center md:justify-start gap-3
                ${currentCategory === cat.id
                                    ? 'bg-gradient-to-r from-blue-500/10 to-purple-500/10 text-blue-600 border-l-4 border-blue-500 dark:text-blue-400'
                                    : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100 dark:text-white/60 dark:hover:text-white dark:hover:bg-white/5 border-l-4 border-transparent'
                                }
              `}
                            title={cat.label}
                        >
                            <cat.icon className={`w-5 h-5 flex-shrink-0 transition-transform duration-200 ${currentCategory === cat.id ? 'scale-110' : 'group-hover:scale-110'}`} />
                            <span className="hidden md:inline">{cat.label}</span>
                        </button>
                    ))}
                </nav>

                {/* Actions */}
                <div className="mt-auto space-y-3">
                    <button
                        onClick={toggleTheme}
                        className="
              w-full px-2 md:px-4 py-2 rounded-lg
              bg-slate-100 text-slate-600
              hover:bg-slate-200
              border border-slate-200
              dark:bg-white/5 dark:text-white/60
              dark:hover:bg-white/10 dark:border-white/10
              transition-all duration-200
              flex items-center justify-center gap-2
            "
                        title={theme === 'light' ? '切换到暗色模式' : '切换到亮色模式'}
                    >
                        {theme === 'light' ? <MoonIcon className="w-5 h-5" /> : <SunIcon className="w-5 h-5" />}
                        <span className="hidden md:inline">{theme === 'light' ? '暗色模式' : '亮色模式'}</span>
                    </button>
                    <button
                        onClick={restartApp}
                        className="
              w-full px-2 md:px-4 py-2 rounded-lg
              bg-purple-50 text-purple-600
              hover:bg-purple-100
              border border-purple-200
              dark:bg-purple-500/20 dark:text-purple-300
              dark:hover:bg-purple-500/30 dark:border-purple-500/30
              transition-all duration-200
              flex items-center justify-center gap-2
            "
                        title="重启应用"
                    >
                        <span className="md:hidden">↻</span>
                        <span className="hidden md:inline">重启应用</span>
                    </button>
                </div>
            </div>

            {/* Right Content Area */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Header */}
                <div className="glass-strong border-b border-slate-200 dark:border-white/10 p-4 md:p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-xl md:text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600 dark:from-white dark:to-white/80">
                            {CATEGORIES.find(c => c.id === currentCategory)?.label}
                        </h1>
                        <p className="text-slate-500 dark:text-white/50 text-xs md:text-sm mt-1">
                            修改后需要保存并重启应用
                        </p>
                    </div>

                    <div className="flex gap-2 md:gap-3 self-end md:self-auto">
                        {hasUnsavedChanges && (
                            <button
                                onClick={resetChanges}
                                disabled={isSaving}
                                className="
                  px-3 md:px-6 py-1.5 md:py-2 rounded-lg text-sm md:text-base
                  bg-white text-slate-500
                  hover:bg-slate-50 hover:text-slate-700
                  border border-slate-200
                  dark:bg-white/5 dark:text-white/60
                  dark:hover:bg-white/10 dark:hover:text-white
                  dark:border-white/10
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
                px-4 md:px-6 py-1.5 md:py-2 rounded-lg text-sm md:text-base
                transition-all duration-200
                ${hasUnsavedChanges && !isSaving
                                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:shadow-lg hover:shadow-blue-500/25 hover:scale-105'
                                    : 'bg-slate-100 text-slate-400 dark:bg-white/10 dark:text-white/30 cursor-not-allowed'
                                }
              `}
                        >
                            {isSaving ? '...' : '保存配置'}
                        </button>
                    </div>
                </div>

                {/* Config Items */}
                <div className="flex-1 overflow-y-auto p-4 md:p-8 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/20 scrollbar-track-transparent">
                    <div className="max-w-full md:max-w-4xl mx-auto space-y-4 md:space-y-6 animate-fade-in-up">
                        {currentConfigs.map((item) => (
                            <div
                                key={`${item.section}.${item.key}`}
                                className="glass rounded-xl p-4 md:p-6 border border-white/20 shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
                            >
                                <div className="mb-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <label className="text-slate-800 dark:text-white font-medium text-sm md:text-base">
                                            {item.label}
                                        </label>
                                        {item.required && (
                                            <span className="text-red-500 dark:text-red-400 text-sm">*</span>
                                        )}
                                    </div>
                                    <p className="text-slate-500 dark:text-white/50 text-xs md:text-sm">
                                        {item.description}
                                    </p>
                                </div>

                                {renderConfigInput(item)}

                                <div className="mt-2 text-[10px] md:text-xs text-slate-400 dark:text-white/30">
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
