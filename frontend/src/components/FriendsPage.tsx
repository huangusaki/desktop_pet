import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    DndContext,
    closestCenter,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    type DragEndEvent,
    DragOverlay,
    type DragStartEvent,
} from '@dnd-kit/core';
import {
    arrayMove,
    SortableContext,
    sortableKeyboardCoordinates,
    verticalListSortingStrategy,
    useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { usePresetsStore } from '../stores/usePresetsStore';
import { presetsApi, type Preset, type PresetCreateData } from '../api/presetsApi';
import {
    ContactsIcon,
    CopyIcon,
    AvatarIcon,
    AgentIcon,
    SettingsIcon,
    ArrowLeftIcon,
    HomeIcon,
    PlusIcon
} from './Icons';

interface PresetItemProps {
    preset: Preset;
    isSelected?: boolean;
    onClick?: () => void;
    style?: React.CSSProperties;
    isDragging?: boolean;
}

const PresetItem = React.forwardRef<HTMLDivElement, PresetItemProps>(
    ({ preset, isSelected, onClick, style, isDragging, ...props }, ref) => {
        return (
            <div ref={ref} style={style} {...props}>
                <button
                    onClick={onClick}
                    className={`
                        w-full p-2 md:p-3 rounded-xl text-left group
                        flex items-center justify-center md:justify-start gap-3 border
                        transition-colors duration-200
                        ${isSelected
                            ? 'bg-white dark:bg-white/10 border-blue-500/50 shadow-md'
                            : 'bg-transparent border-transparent hover:bg-white/50 dark:hover:bg-white/5'
                        }
                        ${isDragging ? 'opacity-50' : ''}
                    `}
                    title={preset.name}
                >
                    <div className="relative flex-shrink-0">
                        <img
                            src={presetsApi.getAvatarUrl(preset.avatar_filename)}
                            alt={preset.name}
                            className={`w-10 h-10 md:w-12 md:h-12 rounded-full object-cover transition-transform duration-200 ${isSelected ? 'scale-105' : 'group-hover:scale-105'}`}
                            onError={(e) => {
                                (e.target as HTMLImageElement).src = '/api/avatar/bot';
                            }}
                        />
                        {preset.is_active && (
                            <div className="absolute -bottom-1 -right-1 w-3 h-3 md:w-4 md:h-4 bg-green-500 rounded-full border-2 border-white dark:border-[#1a1b26]" title="当前激活" />
                        )}
                    </div>
                    <div className="flex-1 min-w-0 hidden md:block">
                        <div className={`font-medium truncate ${isSelected ? 'text-blue-600 dark:text-blue-400' : 'text-slate-700 dark:text-slate-200'}`}>
                            {preset.name}
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 truncate">
                            {preset.bot_name}
                        </div>
                    </div>
                </button>
            </div>
        );
    }
);

interface SortablePresetItemProps {
    preset: Preset;
    isSelected: boolean;
    onSelect: (preset: Preset) => void;
}

const SortablePresetItem = ({ preset, isSelected, onSelect }: SortablePresetItemProps) => {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging,
    } = useSortable({ id: preset.id });

    const style = {
        transform: CSS.Translate.toString(transform),
        transition,
    };

    return (
        <PresetItem
            ref={setNodeRef}
            style={style}
            preset={preset}
            isSelected={isSelected}
            onClick={() => onSelect(preset)}
            isDragging={isDragging}
            {...attributes}
            {...listeners}
        />
    );
};

export const FriendsPage: React.FC = () => {
    const navigate = useNavigate();
    const {
        presets,
        fetchPresets,
        createPreset,
        updatePreset,
        deletePreset,
        activatePreset,
        uploadAvatar,
        createFromCurrentConfig,
        setLastSelectedPresetId,
        getLastSelectedPresetId,
        reorderPresets,
    } = usePresetsStore();

    const [selectedPreset, setSelectedPreset] = useState<Preset | null>(null);
    const [isEditing, setIsEditing] = useState(false);
    const [isCreating, setIsCreating] = useState(false);
    const [avatarFile, setAvatarFile] = useState<File | null>(null);
    const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'persona' | 'advanced'>('overview');
    const [activeId, setActiveId] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                distance: 8,
            },
        }),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    // Form state
    const [formData, setFormData] = useState<PresetCreateData>({
        name: '',
        bot_name: '',
        bot_persona: '',
        speech_pattern: '',
        constraints: '',
        format_example: '',
        avatar_filename: 'default_preset.png',
    });

    useEffect(() => {
        fetchPresets();
    }, [fetchPresets]);

    // Auto-select preset: last selected > active > first
    useEffect(() => {
        if (presets.length > 0 && !selectedPreset) {
            // Try to select last selected preset from localStorage
            const lastSelectedId = getLastSelectedPresetId();
            if (lastSelectedId) {
                const lastSelected = presets.find(p => p.id === lastSelectedId);
                if (lastSelected) {
                    setSelectedPreset(lastSelected);
                    return;
                }
            }

            // Otherwise, select active preset or first preset
            const activePreset = presets.find(p => p.is_active);
            if (activePreset) {
                setSelectedPreset(activePreset);
            } else if (presets.length > 0) {
                setSelectedPreset(presets[0]);
            }
        }
    }, [presets, selectedPreset, getLastSelectedPresetId]);

    useEffect(() => {
        if (selectedPreset) {
            setFormData({
                name: selectedPreset.name,
                bot_name: selectedPreset.bot_name,
                bot_persona: selectedPreset.bot_persona,
                speech_pattern: selectedPreset.speech_pattern,
                constraints: selectedPreset.constraints,
                format_example: selectedPreset.format_example,
                avatar_filename: selectedPreset.avatar_filename,
            });
            setAvatarPreview(presetsApi.getAvatarUrl(selectedPreset.avatar_filename));
        }
    }, [selectedPreset]);

    const handlePresetSelect = (preset: Preset) => {
        setSelectedPreset(preset);
        setLastSelectedPresetId(preset.id);  // Save to localStorage
        setIsEditing(false);
        setIsCreating(false);
        setAvatarFile(null);
        setActiveTab('overview');
    };

    const handleCreateNew = () => {
        setSelectedPreset(null);
        setIsCreating(true);
        setIsEditing(false);
        setFormData({
            name: '',
            bot_name: '',
            bot_persona: '',
            speech_pattern: '',
            constraints: '',
            format_example: '',
            avatar_filename: 'default_preset.png',
        });
        setAvatarPreview(null);
        setAvatarFile(null);
        setActiveTab('overview');
    };

    const handleDragStart = (event: DragStartEvent) => {
        setActiveId(event.active.id as string);
    };

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;

        if (over && active.id !== over.id) {
            const oldIndex = presets.findIndex((p) => p.id === active.id);
            const newIndex = presets.findIndex((p) => p.id === over.id);

            const newPresets = arrayMove(presets, oldIndex, newIndex);
            reorderPresets(newPresets);
        }
        setActiveId(null);
    };

    const handleSaveCurrentConfig = async () => {
        // Prompt for name
        const name = window.prompt("请输入好友名称:", "新好友");
        if (!name) return;

        // Check duplicate
        if (presets.some(p => p.name === name)) {
            alert("好友名称已存在，请使用其他名称。");
            return;
        }

        const preset = await createFromCurrentConfig();
        if (preset) {
            // Update name if different from default (which is bot name)
            if (preset.name !== name) {
                await updatePreset(preset.id, { name });
                preset.name = name;
            }

            setSelectedPreset(preset);
            setLastSelectedPresetId(preset.id);
            alert(`已保存当前配置为好友: ${preset.name}`);
        }
    };

    const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setAvatarFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setAvatarPreview(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSave = async () => {
        try {
            let avatarFilename = formData.avatar_filename;
            if (avatarFile) {
                const filename = await uploadAvatar(avatarFile);
                if (filename) {
                    avatarFilename = filename;
                }
            }

            // Check duplicate name
            const isDuplicate = presets.some(p =>
                p.name === formData.name &&
                (!selectedPreset || p.id !== selectedPreset.id)
            );

            if (isDuplicate) {
                alert("好友名称已存在，请使用其他名称。");
                return;
            }

            const dataToSave = { ...formData, avatar_filename: avatarFilename };

            if (isCreating) {
                const newPreset = await createPreset(dataToSave);
                if (newPreset) {
                    setSelectedPreset(newPreset);
                    setIsCreating(false);
                }
            } else if (selectedPreset) {
                const updated = await updatePreset(selectedPreset.id, dataToSave);
                if (updated) {
                    setSelectedPreset(updated);
                    setIsEditing(false);
                }
            }
            setAvatarFile(null);
        } catch (error) {
            console.error('保存预设失败:', error);
        }
    };

    const handleDelete = async () => {
        if (selectedPreset && window.confirm(`确定要删除预设"${selectedPreset.name}"吗?`)) {
            const success = await deletePreset(selectedPreset.id);
            if (success) {
                setSelectedPreset(null);
                setIsEditing(false);
            }
        }
    };

    const handleActivate = async () => {
        if (selectedPreset) {
            const success = await activatePreset(selectedPreset.id);
            if (success) {
                alert(`预设"${selectedPreset.name}"已激活!`);
            }
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
    };

    const inputClass = `
        w-full px-4 py-3 rounded-xl
        bg-white/50 backdrop-blur-sm border border-slate-200/60
        text-slate-800 placeholder-slate-400
        focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent
        dark:bg-white/5 dark:border-white/10
        dark:text-white dark:placeholder-white/30
        dark:focus:ring-blue-400/50
        transition-all duration-200
    `;

    const labelClass = "block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2";

    const renderTabContent = () => {
        const isReadOnly = !isEditing && !isCreating;

        switch (activeTab) {
            case 'overview':
                return (
                    <div className="space-y-6 animate-fade-in">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label className={labelClass}>好友名称</label>
                                <input
                                    type="text"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    disabled={isReadOnly}
                                    className={inputClass}
                                    placeholder="例如: 爱丽丝"
                                />
                            </div>
                            <div>
                                <label className={labelClass}>Bot名称</label>
                                <input
                                    type="text"
                                    value={formData.bot_name}
                                    onChange={(e) => setFormData({ ...formData, bot_name: e.target.value })}
                                    disabled={isReadOnly}
                                    className={inputClass}
                                    placeholder="例如: Alice"
                                />
                            </div>
                        </div>

                        <div className="glass rounded-xl p-6 border border-white/20">
                            <label className={labelClass}>头像设置</label>
                            <div className="flex items-center gap-8">
                                <div className="relative group">
                                    <img
                                        src={avatarPreview || presetsApi.getAvatarUrl(formData.avatar_filename || 'default_preset.png')}
                                        alt="Avatar"
                                        className="w-32 h-32 rounded-2xl object-cover shadow-lg border-4 border-white/50 dark:border-white/10 transition-transform duration-300 group-hover:scale-105"
                                        onError={(e) => {
                                            (e.target as HTMLImageElement).src = '/api/avatar/bot';
                                        }}
                                    />
                                    {!isReadOnly && (
                                        <button
                                            onClick={() => fileInputRef.current?.click()}
                                            className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-2xl"
                                        >
                                            <span className="text-white font-medium">更换头像</span>
                                        </button>
                                    )}
                                </div>
                                <div className="flex-1">
                                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                                        支持 JPG, PNG 格式。建议尺寸 512x512 像素。
                                    </p>
                                    {!isReadOnly && (
                                        <button
                                            onClick={() => fileInputRef.current?.click()}
                                            className="px-4 py-2 rounded-lg bg-white dark:bg-white/10 border border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/20 transition-colors text-sm font-medium"
                                        >
                                            选择图片
                                        </button>
                                    )}
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        onChange={handleAvatarChange}
                                        className="hidden"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                );
            case 'persona':
                return (
                    <div className="space-y-6 animate-fade-in">
                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <label className={labelClass}>人格设定</label>
                                {isReadOnly && (
                                    <button
                                        onClick={() => copyToClipboard(formData.bot_persona)}
                                        className="text-xs text-blue-500 hover:text-blue-600 flex items-center gap-1"
                                    >
                                        <CopyIcon className="w-3 h-3" /> 复制
                                    </button>
                                )}
                            </div>
                            <textarea
                                value={formData.bot_persona}
                                onChange={(e) => setFormData({ ...formData, bot_persona: e.target.value })}
                                disabled={isReadOnly}
                                rows={8}
                                className={`${inputClass} font-mono text-sm`}
                                placeholder="描述Bot的性格、背景故事和行为方式..."
                            />
                        </div>
                        <div>
                            <label className={labelClass}>说话风格示例</label>
                            <textarea
                                value={formData.speech_pattern || ''}
                                onChange={(e) => setFormData({ ...formData, speech_pattern: e.target.value })}
                                disabled={isReadOnly}
                                rows={6}
                                className={`${inputClass} font-mono text-sm`}
                                placeholder="提供一些Bot的对话示例，帮助AI模仿语气..."
                            />
                        </div>
                    </div>
                );
            case 'advanced':
                return (
                    <div className="space-y-6 animate-fade-in">
                        <div>
                            <label className={labelClass}>表达规则约束</label>
                            <textarea
                                value={formData.constraints || ''}
                                onChange={(e) => setFormData({ ...formData, constraints: e.target.value })}
                                disabled={isReadOnly}
                                rows={6}
                                className={`${inputClass} font-mono text-sm`}
                                placeholder="Bot对话时必须遵守的规则..."
                            />
                        </div>
                        <div>
                            <label className={labelClass}>JSON格式示例</label>
                            <textarea
                                value={formData.format_example || ''}
                                onChange={(e) => setFormData({ ...formData, format_example: e.target.value })}
                                disabled={isReadOnly}
                                rows={8}
                                className={`${inputClass} font-mono text-sm`}
                                placeholder="自定义Bot回复的JSON结构..."
                            />
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className="flex h-screen bg-transparent overflow-hidden">
            {/* Left Sidebar - Presets List */}
            <div className="w-20 md:w-80 glass-strong border-r border-slate-200 dark:border-white/10 flex flex-col z-10 transition-all duration-300">
                <div className="p-2 md:p-6 pb-4 flex flex-col items-center md:items-stretch">
                    <button
                        onClick={() => navigate('/')}
                        className="
                            group
                            text-slate-500 hover:text-slate-800 
                            dark:text-white/60 dark:hover:text-white 
                            transition-all duration-200 
                            flex items-center justify-center md:justify-start gap-2 
                            mb-4 md:mb-6 
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

                    <div className="flex flex-col md:flex-row items-center justify-center md:justify-between mb-4 md:mb-6 gap-2 md:gap-0 w-full">
                        <div className="flex items-center gap-3 text-slate-800 dark:text-white">
                            <ContactsIcon className="w-6 h-6 text-purple-500 dark:text-purple-400" />
                            <h2 className="text-lg font-bold hidden md:block">好友列表</h2>
                        </div>
                        <div className="flex flex-col md:flex-row gap-2">
                            <button
                                onClick={handleSaveCurrentConfig}
                                className="
                                    p-2 rounded-lg 
                                    text-slate-500 hover:text-slate-800 hover:bg-white/50 
                                    dark:text-white/60 dark:hover:text-white dark:hover:bg-white/10 
                                    transition-all duration-200
                                "
                                title="保存当前配置为好友"
                            >
                                <HomeIcon className="w-5 h-5" />
                            </button>
                            <button
                                onClick={handleCreateNew}
                                className="
                                    p-2 rounded-lg 
                                    text-slate-500 hover:text-slate-800 hover:bg-white/50 
                                    dark:text-white/60 dark:hover:text-white dark:hover:bg-white/10 
                                    transition-all duration-200
                                "
                                title="添加好友"
                            >
                                <PlusIcon className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto px-2 md:px-4 pb-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/20 scrollbar-track-transparent">
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragStart={handleDragStart}
                        onDragEnd={handleDragEnd}
                    >
                        <SortableContext
                            items={presets.map(p => p.id)}
                            strategy={verticalListSortingStrategy}
                        >
                            {presets.map((preset) => (
                                <SortablePresetItem
                                    key={preset.id}
                                    preset={preset}
                                    isSelected={selectedPreset?.id === preset.id}
                                    onSelect={handlePresetSelect}
                                />
                            ))}
                        </SortableContext>
                        <DragOverlay>
                            {activeId ? (
                                <PresetItem
                                    preset={presets.find(p => p.id === activeId)!}
                                    isSelected={true}
                                    style={{ cursor: 'grabbing' }}
                                />
                            ) : null}
                        </DragOverlay>
                    </DndContext>
                </div>
            </div>

            {/* Right Content Area */}
            <div className="flex-1 flex flex-col min-w-0 bg-slate-50/50 dark:bg-[#1a1b26]/50">
                {(selectedPreset || isCreating) ? (
                    <>
                        {/* Header Profile Section */}
                        <div className="relative pt-8 md:pt-12 pb-6 md:pb-8 px-4 md:px-8 border-b border-slate-200 dark:border-white/5">
                            <div className="absolute top-0 left-0 w-full h-24 md:h-32 bg-gradient-to-r from-blue-500/10 to-purple-500/10 pointer-events-none" />

                            <div className="relative flex flex-col md:flex-row md:items-end justify-between gap-4">
                                <div className="flex items-center md:items-end gap-4 md:gap-6">
                                    <img
                                        src={avatarPreview || presetsApi.getAvatarUrl(formData.avatar_filename || 'default_preset.png')}
                                        alt="Profile"
                                        className="w-20 h-20 md:w-24 md:h-24 rounded-2xl object-cover shadow-xl border-4 border-white dark:border-[#1a1b26]"
                                        onError={(e) => {
                                            (e.target as HTMLImageElement).src = '/api/avatar/bot';
                                        }}
                                    />
                                    <div className="mb-1 md:mb-2">
                                        <h1 className="text-2xl md:text-3xl font-bold text-slate-800 dark:text-white">
                                            {isCreating ? '创建新好友' : formData.name}
                                        </h1>
                                        <p className="text-slate-500 dark:text-slate-400 flex items-center gap-2 text-sm md:text-base">
                                            <span className="inline-block w-2 h-2 rounded-full bg-green-500"></span>
                                            {isCreating ? '填写以下信息' : formData.bot_name}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex gap-2 md:gap-3 mb-0 md:mb-2 self-end md:self-auto">
                                    {!isEditing && !isCreating ? (
                                        <>
                                            {!selectedPreset?.is_active && (
                                                <button
                                                    onClick={handleActivate}
                                                    className="px-3 md:px-4 py-1.5 md:py-2 rounded-lg bg-green-500 text-white hover:bg-green-600 transition-colors shadow-lg shadow-green-500/20 font-medium text-sm md:text-base"
                                                >
                                                    激活
                                                </button>
                                            )}
                                            <button
                                                onClick={() => setIsEditing(true)}
                                                className="px-3 md:px-4 py-1.5 md:py-2 rounded-lg bg-white dark:bg-white/10 border border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/20 transition-colors font-medium text-slate-700 dark:text-white text-sm md:text-base"
                                            >
                                                编辑
                                            </button>
                                            <button
                                                onClick={handleDelete}
                                                className="px-3 md:px-4 py-1.5 md:py-2 rounded-lg bg-red-50 text-red-600 hover:bg-red-100 border border-red-200 dark:bg-red-500/10 dark:text-red-400 dark:hover:bg-red-500/20 dark:border-red-500/20 transition-colors font-medium text-sm md:text-base"
                                            >
                                                删除
                                            </button>
                                        </>
                                    ) : (
                                        <>
                                            <button
                                                onClick={handleSave}
                                                className="px-4 md:px-6 py-1.5 md:py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-lg shadow-blue-500/20 font-medium text-sm md:text-base"
                                            >
                                                保存
                                            </button>
                                            <button
                                                onClick={() => {
                                                    if (isCreating) {
                                                        setSelectedPreset(null);
                                                        setIsCreating(false);
                                                    } else {
                                                        setIsEditing(false);
                                                        // Reset form
                                                        if (selectedPreset) {
                                                            setFormData({
                                                                name: selectedPreset.name,
                                                                bot_name: selectedPreset.bot_name,
                                                                bot_persona: selectedPreset.bot_persona,
                                                                speech_pattern: selectedPreset.speech_pattern,
                                                                constraints: selectedPreset.constraints,
                                                                format_example: selectedPreset.format_example,
                                                                avatar_filename: selectedPreset.avatar_filename,
                                                            });
                                                        }
                                                    }
                                                }}
                                                className="px-3 md:px-4 py-1.5 md:py-2 rounded-lg bg-white dark:bg-white/10 border border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/20 transition-colors font-medium text-slate-700 dark:text-white text-sm md:text-base"
                                            >
                                                取消
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>

                            {/* Tabs */}
                            <div className="flex gap-4 md:gap-6 mt-6 md:mt-8 border-b border-slate-200 dark:border-white/10 overflow-x-auto scrollbar-none">
                                <button
                                    onClick={() => setActiveTab('overview')}
                                    className={`pb-3 px-2 text-sm font-medium transition-colors relative whitespace-nowrap ${activeTab === 'overview'
                                        ? 'text-blue-600 dark:text-blue-400'
                                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <AvatarIcon className="w-4 h-4" />
                                        基础信息
                                    </div>
                                    {activeTab === 'overview' && (
                                        <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600 dark:bg-blue-400 rounded-t-full" />
                                    )}
                                </button>
                                <button
                                    onClick={() => setActiveTab('persona')}
                                    className={`pb-3 px-2 text-sm font-medium transition-colors relative whitespace-nowrap ${activeTab === 'persona'
                                        ? 'text-blue-600 dark:text-blue-400'
                                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <AgentIcon className="w-4 h-4" />
                                        人格设定
                                    </div>
                                    {activeTab === 'persona' && (
                                        <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600 dark:bg-blue-400 rounded-t-full" />
                                    )}
                                </button>
                                <button
                                    onClick={() => setActiveTab('advanced')}
                                    className={`pb-3 px-2 text-sm font-medium transition-colors relative whitespace-nowrap ${activeTab === 'advanced'
                                        ? 'text-blue-600 dark:text-blue-400'
                                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <SettingsIcon className="w-4 h-4" />
                                        高级配置
                                    </div>
                                    {activeTab === 'advanced' && (
                                        <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600 dark:bg-blue-400 rounded-t-full" />
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* Tab Content */}
                        <div className="flex-1 overflow-y-auto p-4 md:p-8 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/20 scrollbar-track-transparent">
                            <div className="max-w-4xl mx-auto">
                                {renderTabContent()}
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-slate-400 dark:text-slate-500">
                        <div className="w-20 h-20 md:w-24 md:h-24 bg-slate-100 dark:bg-white/5 rounded-full flex items-center justify-center mb-4">
                            <ContactsIcon className="w-8 h-8 md:w-10 md:h-10 opacity-50" />
                        </div>
                        <h3 className="text-lg md:text-xl font-medium text-slate-600 dark:text-slate-300 mb-2">
                            没有选择好友
                        </h3>
                        <p className="max-w-xs text-center text-sm px-4">
                            点击左上角的<span className="text-green-500 font-medium">绿色按钮</span>保存当前配置为好友，或点击<span className="text-blue-500 font-medium">蓝色按钮</span>创建新好友。
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};
