from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    from ..utils.config_manager import ConfigManager
except ImportError:
    try:
        from utils.config_manager import ConfigManager
    except ImportError:
        print(
            "CRITICAL: Could not import ConfigManager for MemoryConfig. Ensure paths are correct."
        )
        ConfigManager = None


@dataclass
class MemoryConfig:
    """记忆系统配置类"""

    memory_build_distribution: Tuple[float, float, float, float, float, float]
    build_memory_sample_num: int
    build_memory_sample_length: int
    memory_compress_rate: float
    memory_forget_time_hours: float
    memory_forget_percentage: float
    memory_ban_words: List[str]
    consolidate_memory_percentage: float
    consolidation_similarity_threshold: float
    llm_topic_judge: Dict[str, any] = field(default_factory=dict)
    llm_summary_by_topic: Dict[str, any] = field(default_factory=dict)
    llm_embedding_topic: Dict[str, any] = field(default_factory=dict)
    llm_re_rank: Dict[str, any] = field(default_factory=dict)
    max_memorized_time_per_msg: int = 3
    topic_similarity_threshold_for_connection: float = 0.7
    max_similar_topics_to_connect: int = 3
    node_summary_forget_time_hours: float = 360.0
    rpm_limit_delay_summary_sec: float = 2.0
    embedding_update_delay_sec: float = 0.1
    embedding_update_batch_size: int = 50
    max_topics_per_snippet: int = 8
    retrieval_max_final_memories: int = 5
    retrieval_activation_depth: int = 3
    fast_retrieval_max_keywords: int = 5
    retrieval_input_max_topics: int = 5
    retrieval_input_compress_rate: float = 0.05
    activation_decay_per_link: float = 0.5
    activation_link_decay_base: float = 0.5
    activation_min_threshold_for_spread: float = 0.1
    retrieval_top_activated_nodes_to_scan: int = 20
    retrieval_min_summary_similarity: float = 0.65
    retrieval_max_candidates_for_llm_rerank: int = 15
    rerank_target_selection_count: int = 5
    keyword_retrieval_node_similarity_threshold: float = 0.8

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager):
        if config_manager is None:
            raise ValueError(
                "ConfigManager is not available for MemoryConfig creation."
            )
        llm_topic_judge_cfg = config_manager.get_memory_llm_config("LLM_TOPIC_JUDGE")
        llm_summary_by_topic_cfg = config_manager.get_memory_llm_config(
            "LLM_SUMMARY_BY_TOPIC"
        )
        llm_embedding_topic_cfg = config_manager.get_memory_llm_config(
            "LLM_EMBEDDING_TOPIC"
        )
        llm_re_rank_cfg = config_manager.get_memory_llm_config("LLM_RE_RANK")

        def get_param(method_name_suffix, fallback_value, param_type=float):
            method_to_call = f"get_memory_{method_name_suffix}"
            if hasattr(config_manager, method_to_call):
                try:
                    return getattr(config_manager, method_to_call)()
                except Exception as e:
                    raw_val = config_manager.config.get(
                        "MEMORY_SYSTEM_PARAMS",
                        method_name_suffix.upper(),
                        fallback=None,
                    )
                    if raw_val is not None:
                        try:
                            return param_type(raw_val)
                        except ValueError:
                            return fallback_value
                    return fallback_value
            else:
                raw_val = config_manager.config.get(
                    "MEMORY_SYSTEM_PARAMS", method_name_suffix.upper(), fallback=None
                )
                if raw_val is not None:
                    try:
                        return param_type(raw_val)
                    except ValueError:
                        return fallback_value
                return fallback_value

        return cls(
            memory_build_distribution=config_manager.get_memory_build_distribution(),
            build_memory_sample_num=config_manager.get_memory_build_sample_num(),
            build_memory_sample_length=config_manager.get_memory_build_sample_length(),
            memory_compress_rate=config_manager.get_memory_compress_rate(),
            memory_forget_time_hours=config_manager.get_memory_forget_time_hours(),
            memory_forget_percentage=config_manager.get_memory_forget_percentage(),
            memory_ban_words=config_manager.get_memory_ban_words(),
            consolidate_memory_percentage=config_manager.get_memory_consolidate_percentage(),
            consolidation_similarity_threshold=config_manager.get_memory_consolidation_similarity_threshold(),
            llm_topic_judge=llm_topic_judge_cfg,
            llm_summary_by_topic=llm_summary_by_topic_cfg,
            llm_embedding_topic=llm_embedding_topic_cfg,
            llm_re_rank=llm_re_rank_cfg,
            max_memorized_time_per_msg=get_param("max_memorized_time_per_msg", 3, int),
            topic_similarity_threshold_for_connection=get_param(
                "topic_similarity_threshold_for_connection", 0.7
            ),
            max_similar_topics_to_connect=get_param(
                "max_similar_topics_to_connect", 3, int
            ),
            node_summary_forget_time_hours=get_param(
                "node_summary_forget_time_hours", 360.0
            ),
            rpm_limit_delay_summary_sec=get_param("rpm_limit_delay_summary_sec", 2.0),
            embedding_update_delay_sec=get_param("embedding_update_delay_sec", 0.1),
            embedding_update_batch_size=get_param(
                "embedding_update_batch_size", 50, int
            ),
            max_topics_per_snippet=get_param("max_topics_per_snippet", 8, int),
            retrieval_max_final_memories=get_param(
                "retrieval_max_final_memories", 5, int
            ),
            retrieval_activation_depth=get_param("retrieval_activation_depth", 3, int),
            fast_retrieval_max_keywords=get_param(
                "fast_retrieval_max_keywords", 5, int
            ),
            retrieval_input_max_topics=get_param("retrieval_input_max_topics", 5, int),
            retrieval_input_compress_rate=get_param(
                "retrieval_input_compress_rate", 0.05
            ),
            activation_decay_per_link=get_param("activation_decay_per_link", 0.5),
            activation_link_decay_base=get_param("activation_link_decay_base", 0.5),
            activation_min_threshold_for_spread=get_param(
                "activation_min_threshold_for_spread", 0.1
            ),
            retrieval_top_activated_nodes_to_scan=get_param(
                "retrieval_top_activated_nodes_to_scan", 20, int
            ),
            retrieval_min_summary_similarity=get_param(
                "retrieval_min_summary_similarity", 0.65
            ),
            retrieval_max_candidates_for_llm_rerank=get_param(
                "retrieval_max_candidates_for_llm_rerank", 15, int
            ),
            rerank_target_selection_count=get_param(
                "rerank_target_selection_count", 5, int
            ),
            keyword_retrieval_node_similarity_threshold=get_param(
                "keyword_retrieval_node_similarity_threshold", 0.8
            ),
        )
