import logging

if not logging.getLogger("memory_system").hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
from .hippocampus_utils import calculate_information_content, cosine_similarity
from .hippocampus_graph import MemoryGraph
from .hippocampus_io import EntorhinalCortex
from .hippocampus_processing import ParahippocampalGyrus
from .hippocampus_core_logic import Hippocampus
from .hippocampus_manager import HippocampusManager
from ..llm.llm_request import LLM_request, GeminiSDKResponse

__all__ = [
    "calculate_information_content",
    "cosine_similarity",
    "MemoryGraph",
    "EntorhinalCortex",
    "ParahippocampalGyrus",
    "Hippocampus",
    "HippocampusManager",
    "MemoryConfig",
    "LLM_request",
    "GeminiSDKResponse",
]
logger = logging.getLogger(__name__)
logger.info("记忆系统加载完成！")
