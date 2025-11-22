# -*- coding: utf-8 -*-
"""
预设(好友)数据模型和管理器
用于管理不同的AI人格预设,每个预设包含独立的配置和头像
"""
import os
import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger("PresetManager")


class Preset(BaseModel):
    """预设数据模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="预设名称(显示名称)")
    avatar_filename: str = Field(default="default_preset.png", description="头像文件名")
    
    # Bot基础配置
    bot_name: str = Field(..., description="Bot名称")
    bot_persona: str = Field(..., description="Bot人格设定")
    speech_pattern: str = Field(default="", description="说话风格示例")
    constraints: str = Field(default="", description="表达规则约束")
    format_example: str = Field(default="", description="JSON格式示例")
    
    # 元数据
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = Field(default=False, description="是否为当前激活的预设")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "爱丽丝",
                "avatar_filename": "alice.png",
                "bot_name": "爱丽丝",
                "bot_persona": "你是爱丽丝,一个可爱但有点毒舌的AI助手...",
                "speech_pattern": "\"哈……?你是笨蛋吗?\"",
                "constraints": "【表达规则】...",
                "format_example": "{...}",
                "is_active": True
            }
        }


class PresetManager:
    """预设管理器 - 负责预设的CRUD操作和文件存储"""
    
    def __init__(self, presets_dir: str = "data/presets", avatars_dir: str = "data/presets/avatars", config_manager=None):
        """
        初始化预设管理器
        
        Args:
            presets_dir: 预设JSON文件存储目录
            avatars_dir: 预设头像存储目录
            config_manager: ConfigManager实例,用于创建默认预设
        """
        self.presets_dir = presets_dir
        self.avatars_dir = avatars_dir
        self.presets_file = os.path.join(presets_dir, "presets.json")
        self.config_manager = config_manager
        
        # 确保目录存在
        os.makedirs(self.presets_dir, exist_ok=True)
        os.makedirs(self.avatars_dir, exist_ok=True)
        
        # 初始化预设文件
        if not os.path.exists(self.presets_file):
            self._save_presets([])
            logger.info(f"创建新的预设文件: {self.presets_file}")
        
        # 如果没有预设且提供了config_manager,创建默认预设
        presets = self._load_presets()
        if len(presets) == 0 and config_manager is not None:
            self._create_default_preset_from_config()
        
        logger.info(f"PresetManager 已初始化: presets_dir={presets_dir}, avatars_dir={avatars_dir}")
    
    def _create_default_preset_from_config(self) -> Optional[Preset]:
        """从当前ConfigManager配置创建默认预设"""
        try:
            if self.config_manager is None:
                logger.warning("无法创建默认预设: 未提供ConfigManager")
                return None
            
            # 从配置读取当前设置 - 使用ConfigManager的标准接口
            bot_name = self.config_manager.get_bot_name()
            bot_persona = self.config_manager.get_bot_persona()
            speech_pattern = self.config_manager.get_bot_speech_pattern()
            constraints = self.config_manager.get_bot_constraints()
            format_example = self.config_manager.get_bot_format_example()
            
            # 创建默认预设
            default_preset_data = {
                "name": bot_name,
                "bot_name": bot_name,
                "bot_persona": bot_persona,
                "speech_pattern": speech_pattern,
                "constraints": constraints,
                "format_example": format_example,
                "avatar_filename": "default_preset.png",
                "is_active": True  # 默认激活
            }
            
            preset = self.create_preset(default_preset_data)
            if preset:
                logger.info(f"已从当前配置创建默认预设: {preset.name}")
            return preset
        except Exception as e:
            logger.error(f"创建默认预设失败: {e}", exc_info=True)
            return None
    
    def _load_presets(self) -> List[Preset]:
        """从文件加载所有预设"""
        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Preset(**preset_data) for preset_data in data]
        except Exception as e:
            logger.error(f"加载预设文件失败: {e}", exc_info=True)
            return []
    
    def _save_presets(self, presets: List[Preset]) -> bool:
        """保存所有预设到文件"""
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                data = [preset.model_dump() for preset in presets]
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存预设文件失败: {e}", exc_info=True)
            return False
    
    def get_all_presets(self) -> List[Preset]:
        """获取所有预设"""
        return self._load_presets()
    
    def get_preset_by_id(self, preset_id: str) -> Optional[Preset]:
        """根据ID获取预设"""
        presets = self._load_presets()
        for preset in presets:
            if preset.id == preset_id:
                return preset
        return None
    
    def get_active_preset(self) -> Optional[Preset]:
        """获取当前激活的预设"""
        presets = self._load_presets()
        for preset in presets:
            if preset.is_active:
                return preset
        return None
    
    def create_preset(self, preset_data: Dict[str, Any]) -> Optional[Preset]:
        """
        创建新预设
        
        Args:
            preset_data: 预设数据字典
            
        Returns:
            创建的预设对象,失败返回None
        """
        try:
            # 创建预设对象
            preset = Preset(**preset_data)
            
            # 加载现有预设
            presets = self._load_presets()
            
            # 添加新预设
            presets.append(preset)
            
            # 保存
            if self._save_presets(presets):
                logger.info(f"创建预设成功: id={preset.id}, name={preset.name}")
                return preset
            else:
                logger.error(f"创建预设失败: 保存文件时出错")
                return None
        except Exception as e:
            logger.error(f"创建预设失败: {e}", exc_info=True)
            return None
    
    def update_preset(self, preset_id: str, update_data: Dict[str, Any]) -> Optional[Preset]:
        """
        更新预设
        
        Args:
            preset_id: 预设ID
            update_data: 要更新的数据字典
            
        Returns:
            更新后的预设对象,失败返回None
        """
        try:
            presets = self._load_presets()
            
            # 查找要更新的预设
            preset_index = None
            for i, preset in enumerate(presets):
                if preset.id == preset_id:
                    preset_index = i
                    break
            
            if preset_index is None:
                logger.warning(f"更新预设失败: 未找到ID为 {preset_id} 的预设")
                return None
            
            # 更新预设数据
            preset_dict = presets[preset_index].model_dump()
            preset_dict.update(update_data)
            preset_dict['updated_at'] = datetime.now().isoformat()
            
            # 创建新的预设对象
            updated_preset = Preset(**preset_dict)
            presets[preset_index] = updated_preset
            
            # 保存
            if self._save_presets(presets):
                logger.info(f"更新预设成功: id={preset_id}")
                return updated_preset
            else:
                logger.error(f"更新预设失败: 保存文件时出错")
                return None
        except Exception as e:
            logger.error(f"更新预设失败: {e}", exc_info=True)
            return None
    
    def delete_preset(self, preset_id: str) -> bool:
        """
        删除预设
        
        Args:
            preset_id: 预设ID
            
        Returns:
            是否删除成功
        """
        try:
            presets = self._load_presets()
            
            # 查找并删除预设
            original_count = len(presets)
            presets = [p for p in presets if p.id != preset_id]
            
            if len(presets) == original_count:
                logger.warning(f"删除预设失败: 未找到ID为 {preset_id} 的预设")
                return False
            
            # 保存
            if self._save_presets(presets):
                logger.info(f"删除预设成功: id={preset_id}")
                return True
            else:
                logger.error(f"删除预设失败: 保存文件时出错")
                return False
        except Exception as e:
            logger.error(f"删除预设失败: {e}", exc_info=True)
            return False
    
    def activate_preset(self, preset_id: str) -> Optional[Preset]:
        """
        激活预设(将其他预设设为非激活状态)
        
        Args:
            preset_id: 要激活的预设ID
            
        Returns:
            激活的预设对象,失败返回None
        """
        try:
            presets = self._load_presets()
            
            # 查找目标预设
            target_preset = None
            for preset in presets:
                if preset.id == preset_id:
                    preset.is_active = True
                    target_preset = preset
                else:
                    preset.is_active = False
            
            if target_preset is None:
                logger.warning(f"激活预设失败: 未找到ID为 {preset_id} 的预设")
                return None
            
            # 保存
            if self._save_presets(presets):
                logger.info(f"激活预设成功: id={preset_id}, name={target_preset.name}")
                return target_preset
            else:
                logger.error(f"激活预设失败: 保存文件时出错")
                return None
        except Exception as e:
            logger.error(f"激活预设失败: {e}", exc_info=True)
            return None
    
    def get_avatar_path(self, avatar_filename: str) -> str:
        """获取头像的完整路径"""
        return os.path.join(self.avatars_dir, avatar_filename)
    
    def avatar_exists(self, avatar_filename: str) -> bool:
        """检查头像文件是否存在"""
        return os.path.exists(self.get_avatar_path(avatar_filename))

    def reorder_presets(self, preset_ids: List[str]) -> bool:
        """
        重新排序预设
        
        Args:
            preset_ids: 排序后的预设ID列表
            
        Returns:
            是否成功
        """
        try:
            presets = self._load_presets()
            
            # 创建ID到预设的映射
            preset_map = {p.id: p for p in presets}
            
            # 检查所有ID是否都存在
            if len(preset_ids) != len(presets):
                # 如果长度不一致，可能是有些预设未包含在列表中，或者列表中有无效ID
                # 这里我们只重新排序存在的ID，并把未提到的ID放到最后
                existing_ids = set(p.id for p in presets)
                provided_ids = set(preset_ids)
                
                # 过滤无效ID
                valid_ordered_ids = [pid for pid in preset_ids if pid in existing_ids]
                
                # 找出未提到的ID
                missing_ids = list(existing_ids - provided_ids)
                
                # 合并ID列表
                final_ids = valid_ordered_ids + missing_ids
            else:
                final_ids = preset_ids
            
            # 构建新的预设列表
            new_presets = []
            for pid in final_ids:
                if pid in preset_map:
                    new_presets.append(preset_map[pid])
            
            # 确保没有丢失预设 (双重检查)
            if len(new_presets) != len(presets):
                logger.error("重新排序失败: 预设数量不匹配")
                return False
                
            # 保存
            if self._save_presets(new_presets):
                logger.info("预设重新排序成功")
                return True
            else:
                logger.error("预设重新排序失败: 保存文件时出错")
                return False
        except Exception as e:
            logger.error(f"预设重新排序失败: {e}", exc_info=True)
            return False
