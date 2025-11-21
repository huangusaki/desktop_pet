"""
Configuration service layer for reading, validating, and saving configurations.
"""
import os
import logging
import configparser
from typing import Dict, List, Any, Optional
from src.utils.config_schema import ConfigCategory, ConfigItem, CONFIG_SCHEMA
from src.utils.config_manager import ConfigManager

logger = logging.getLogger("ConfigService")


class ConfigService:
    """Service for managing configuration read/write operations."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the configuration service.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.config_path = os.path.join(
            config_manager.get_project_root(),
            "config",
            "settings.ini"
        )
    
    def get_all_configs(self) -> Dict[str, List[ConfigItem]]:
        """
        Get all configuration items organized by category.
        
        Returns:
            Dictionary mapping category names to lists of ConfigItem objects
        """
        all_configs = {}
        
        for category, schema_items in CONFIG_SCHEMA.items():
            config_items = []
            
            for schema in schema_items:
                # Read current value from config manager
                current_value = self._read_config_value(
                    schema["section"],
                    schema["key"],
                    schema["default"],
                    schema["type"]
                )
                
                config_item = ConfigItem(
                    section=schema["section"],
                    key=schema["key"],
                    value=current_value,
                    value_type=schema["type"],
                    default_value=schema["default"],
                    label=schema["label"],
                    description=schema["description"],
                    category=category,
                    required=schema.get("required", False),
                    sensitive=schema.get("sensitive", False),
                    options=schema.get("options", None)
                )
                config_items.append(config_item)
            
            all_configs[category.value] = config_items
        
        return all_configs
    
    def get_category_configs(self, category: str) -> List[ConfigItem]:
        """
        Get configuration items for a specific category.
        
        Args:
            category: Category name (from ConfigCategory enum)
            
        Returns:
            List of ConfigItem objects for the category
        """
        try:
            cat_enum = ConfigCategory(category)
            all_configs = self.get_all_configs()
            return all_configs.get(cat_enum.value, [])
        except ValueError:
            logger.error(f"Invalid category: {category}")
            return []
    
    def _read_config_value(
        self,
        section: str,
        key: str,
        default: Any,
        value_type: str
    ) -> Any:
        """
        Read a config value from the config manager.
        
        Args:
            section: INI section name
            key: INI key name
            default: Default value if not found
            value_type: Type of the value ('string', 'int', 'float', 'bool', 'text', 'password')
            
        Returns:
            The configuration value
        """
        try:
            config = self.config_manager.config
            
            if not config.has_section(section):
                return default
            
            if not config.has_option(section, key):
                return default
            
            # Read based on type
            if value_type == "bool":
                return config.getboolean(section, key, fallback=default)
            elif value_type == "int":
                return config.getint(section, key, fallback=default)
            elif value_type == "float":
                return config.getfloat(section, key, fallback=default)
            else:  # string, text, password
                value = config.get(section, key, fallback=str(default))
                # Strip quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                return value
        
        except Exception as e:
            logger.error(f"Error reading config {section}.{key}: {e}")
            return default
    
    def validate_config_value(
        self,
        section: str,
        key: str,
        value: Any,
        value_type: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a configuration value.
        
        Args:
            section: INI section name
            key: INI key name
            value: Value to validate
            value_type: Expected type
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if value_type == "bool":
                if not isinstance(value, bool):
                    if isinstance(value, str):
                        if value.lower() not in ["true", "false", "1", "0", "yes", "no"]:
                            return False, "布尔值必须是 true/false"
                    else:
                        return False, "布尔值必须是 true/false"
            
            elif value_type == "int":
                try:
                    int(value)
                except (ValueError, TypeError):
                    return False, "必须是整数"
            
            elif value_type == "float":
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, "必须是数字"
            
            # Add specific validation rules
            if section == "MONGODB" and key == "CONNECTION_STRING":
                if not str(value).startswith("mongodb://"):
                    return False, "MongoDB连接字符串必须以 mongodb:// 开头"
            
            if section == "SCREEN_ANALYSIS" and key == "CHANCE":
                val = float(value)
                if val < 0 or val > 1:
                    return False, "概率必须在 0-1 之间"
            
            return True, None
        
        except Exception as e:
            logger.error(f"Validation error for {section}.{key}: {e}")
            return False, str(e)
    
    def update_configs(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration values in memory.
        
        Args:
            updates: Dictionary of config updates in format "SECTION.KEY": value
            
        Returns:
            Dictionary with results: {
                "success": True/False,
                "updated": [...],
                "errors": {...}
            }
        """
        updated = []
        errors = {}
        
        for config_key, new_value in updates.items():
            try:
                # Parse section and key
                if "." not in config_key:
                    errors[config_key] = "配置键格式错误,应为 SECTION.KEY"
                    continue
                
                section, key = config_key.split(".", 1)
                
                # Find schema for validation
                value_type = self._find_value_type(section, key)
                if not value_type:
                    errors[config_key] = "未知的配置项"
                    continue
                
                # Validate
                is_valid, error_msg = self.validate_config_value(
                    section, key, new_value, value_type
                )
                
                if not is_valid:
                    errors[config_key] = error_msg
                    continue
                
                # Convert value to appropriate type
                converted_value = self._convert_value(new_value, value_type)
                
                # Update in config object
                config = self.config_manager.config
                if not config.has_section(section):
                    config.add_section(section)
                
                config.set(section, key, str(converted_value))
                updated.append(config_key)
                
            except Exception as e:
                logger.error(f"Error updating {config_key}: {e}")
                errors[config_key] = str(e)
        
        return {
            "success": len(errors) == 0,
            "updated": updated,
            "errors": errors
        }
    
    def _find_value_type(self, section: str, key: str) -> Optional[str]:
        """Find the value type for a config item from schema."""
        for category_items in CONFIG_SCHEMA.values():
            for schema in category_items:
                if schema["section"] == section and schema["key"] == key:
                    return schema["type"]
        return None
    
    def _convert_value(self, value: Any, value_type: str) -> Any:
        """Convert a value to the appropriate type."""
        if value_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ["true", "1", "yes"]
            return bool(value)
        
        elif value_type == "int":
            return int(value)
        
        elif value_type == "float":
            return float(value)
        
        else:  # string, text, password
            return str(value)
    
    def save_to_file(self) -> tuple[bool, Optional[str]]:
        """
        Save the current configuration to the INI file.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create backup before saving
            backup_path = f"{self.config_path}.backup"
            if os.path.exists(self.config_path):
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Write to file
            with open(self.config_path, "w", encoding="utf-8") as f:
                self.config_manager.config.write(f)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True, None
        
        except Exception as e:
            error_msg = f"保存配置文件失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
