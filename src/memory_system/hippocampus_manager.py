import asyncio
from typing import Optional, List, Tuple, Dict, Any
from pymongo.database import Database
import logging
from .memory_config import MemoryConfig
from .hippocampus_core_logic import Hippocampus
from ..utils.prompt_builder import PromptBuilder

logger = logging.getLogger("memory_system.manager")


class HippocampusManager:
    _instance: Optional["HippocampusManager"] = None
    _hippocampus: Optional[Hippocampus] = None
    _initialized: bool = False
    _async_lock = asyncio.Lock()

    def __init__(self):
        if HippocampusManager._instance is not None:
            logger.warning(
                "HippocampusManager __init__ called when _instance already exists. This might be a re-initialization."
            )
        HippocampusManager._instance = self

    @classmethod
    async def get_instance(cls) -> "HippocampusManager":
        if cls._instance is None:
            async with cls._async_lock:
                if cls._instance is None:
                    cls()
        if cls._instance is None:
            raise RuntimeError("Failed to create HippocampusManager instance.")
        return cls._instance

    @classmethod
    def get_hippocampus(cls) -> Hippocampus:
        if not cls._initialized or cls._hippocampus is None:
            raise RuntimeError(
                "HippocampusManager 尚未初始化或 Hippocampus 实例丢失，请先调用 initialize_singleton"
            )
        return cls._hippocampus

    async def initialize_singleton(
        self,
        memory_config: MemoryConfig,
        database_instance: Database,
        chat_collection_name: str,
        pet_name: str,
        prompt_builder: PromptBuilder,
        global_llm_params: Optional[Dict[str, Any]] = None,
    ) -> Hippocampus:
        if HippocampusManager._initialized:
            logger.warning("HippocampusManager 已初始化，跳过重复初始化。")
            if HippocampusManager._hippocampus is None:
                raise RuntimeError(
                    "Manager initialized but Hippocampus instance is None."
                )
            return HippocampusManager._hippocampus
        try:
            hippocampus_instance = Hippocampus()
            hippocampus_instance.initialize(
                memory_config=memory_config,
                database_instance=database_instance,
                chat_history_collection_name=chat_collection_name,
                pet_name=pet_name,
                prompt_builder=prompt_builder,
                global_llm_params=global_llm_params,
            )
            HippocampusManager._hippocampus = hippocampus_instance
            HippocampusManager._initialized = True
            logger.info("海马体管理器 (HippocampusManager) 初始化成功。")
            h_cfg = HippocampusManager._hippocampus.config
            if h_cfg and HippocampusManager._hippocampus.memory_graph:
                nodes = len(HippocampusManager._hippocampus.memory_graph.G.nodes())
                edges = len(HippocampusManager._hippocampus.memory_graph.G.edges())
                llm_emb_config = getattr(h_cfg, "llm_embedding_topic", {})
                llm_emb_name = (
                    llm_emb_config.get("name", "N/A")
                    if isinstance(llm_emb_config, dict)
                    else "N/A"
                )
                logger.info(
                    f"--- 记忆系统配置: 构建(样本数:{h_cfg.build_memory_sample_num}), "
                    f"遗忘(比例:{h_cfg.memory_forget_percentage * 100:.2f}%), 整合(比例:{h_cfg.consolidate_memory_percentage * 100:.2f}%), "
                    f"嵌入模型({llm_emb_name}), 当前图({nodes}节点,{edges}边) ---"
                )
        except Exception as e:
            logger.error(f"HippocampusManager 初始化失败: {e}", exc_info=True)
            HippocampusManager._initialized = False
            HippocampusManager._hippocampus = None
            raise
        return HippocampusManager._hippocampus

    async def build_memory(self):
        hippocampus = self.get_hippocampus()
        if not hippocampus.parahippocampal_gyrus:
            raise RuntimeError("ParahippocampalGyrus 组件未初始化")
        try:
            await hippocampus.parahippocampal_gyrus.operation_build_memory()
        except Exception as e:
            logger.error(f"执行build_memory出错:{e}", exc_info=True)

    async def forget_memory(self, percentage: Optional[float] = None):
        hippocampus = self.get_hippocampus()
        if not hippocampus.parahippocampal_gyrus:
            raise RuntimeError("ParahippocampalGyrus 组件未初始化")
        if not hippocampus.config:
            raise RuntimeError("Hippocampus 配置未加载")
        eff_perc = (
            percentage
            if percentage is not None
            else hippocampus.config.memory_forget_percentage
        )
        try:
            await hippocampus.parahippocampal_gyrus.operation_forget_topic(eff_perc)
        except Exception as e:
            logger.error(f"执行forget_memory(perc={eff_perc})出错:{e}", exc_info=True)

    async def consolidate_memory(self):
        hippocampus = self.get_hippocampus()
        if not hippocampus.parahippocampal_gyrus:
            raise RuntimeError("ParahippocampalGyrus 组件未初始化")
        try:
            await hippocampus.parahippocampal_gyrus.operation_consolidate_memory()
        except Exception as e:
            logger.error(f"执行consolidate_memory出错:{e}", exc_info=True)

    async def get_memory_from_text(
        self,
        txt: str,
        num: Optional[int] = None,
        depth: Optional[int] = None,
        fast_kw: bool = False,
        retrieval_summary_level: str = "L1_core_sentence",
        output_summary_level: str = "L2_paragraph",
    ) -> List[Tuple[str, str]]:
        hippocampus = self.get_hippocampus()
        try:
            return await hippocampus.get_memory_from_text(
                txt,
                num=num,
                depth=depth,
                fast_kw=fast_kw,
                retrieval_summary_level=retrieval_summary_level,
                output_summary_level=output_summary_level,
            )
        except Exception as e:
            logger.error(f"执行get_memory_from_text出错:{e}", exc_info=True)
            return []

    async def get_activation_score_from_text(
        self, txt: str, depth: int = 3, fast_kw: bool = False
    ) -> float:
        hippocampus = self.get_hippocampus()
        try:
            return await hippocampus.get_activation_score_from_text(
                txt, depth=depth, fast_kw=fast_kw
            )
        except Exception as e:
            logger.error(f"执行get_activation_score_from_text出错:{e}", exc_info=True)
            return 0.0

    async def get_memory_from_keyword(
        self, kw: str, depth: int = 2, summary_level: str = "L1_core_sentence"
    ) -> List[Tuple[str, List[str], float]]:
        hippocampus = self.get_hippocampus()
        try:
            return await hippocampus.get_memory_from_keyword(
                kw, depth=depth, summary_level=summary_level
            )
        except Exception as e:
            logger.error(f"执行get_memory_from_keyword出错:{e}", exc_info=True)
            return []

    def get_all_node_names(self) -> List[str]:
        hippocampus = self.get_hippocampus()
        try:
            return hippocampus.get_all_node_names()
        except Exception as e:
            logger.error(f"执行get_all_node_names出错:{e}", exc_info=True)
            return []
