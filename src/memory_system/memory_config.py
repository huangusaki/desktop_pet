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
    min_topics_per_snippet: int = 1
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
    topic_calc_len_base: float = 1.0
    topic_calc_len_log_factor: float = 1.0
    topic_calc_len_short_thresh: int = 5
    topic_calc_len_very_short_val: int = 1
    topic_calc_info_baseline: float = 2.5
    topic_calc_info_scale: float = 1.5
    topic_calc_info_max_absolute: int = 5
    sample_time_gap_threshold_seconds: int = 1800
    sample_max_anchor_attempts: int = 10
    sample_min_snippet_messages: int = 2
    sample_max_snippet_messages: int = 50

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

        def get_param_from_cm(method_name_suffix, fallback_value, param_type=float):
            method_to_call_str = f"get_memory_{method_name_suffix}"
            if hasattr(config_manager, method_to_call_str):
                try:
                    return getattr(config_manager, method_to_call_str)()
                except Exception as e:
                    pass
            config_key_name = method_name_suffix.upper()
            raw_val = None
            if config_manager.config.has_option(
                config_manager._PARAMS_SECTION, config_key_name
            ):
                raw_val = config_manager.config.get(
                    config_manager._PARAMS_SECTION,
                    config_key_name,
                    fallback=None,
                )
            if raw_val is not None:
                try:
                    if param_type == bool:
                        if raw_val.lower() in ["true", "yes", "1"]:
                            return True
                        elif raw_val.lower() in ["false", "no", "0"]:
                            return False
                        else:
                            return fallback_value
                    return param_type(raw_val)
                except ValueError:
                    return fallback_value
            return fallback_value

        return cls(
            memory_build_distribution=config_manager.get_memory_build_distribution(),
            build_memory_sample_num=config_manager.get_memory_build_sample_num(),
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
            max_memorized_time_per_msg=get_param_from_cm(
                "max_memorized_time_per_msg", cls.max_memorized_time_per_msg, int
            ),
            topic_similarity_threshold_for_connection=get_param_from_cm(
                "topic_similarity_threshold_for_connection",
                cls.topic_similarity_threshold_for_connection,
                float,
            ),
            max_similar_topics_to_connect=get_param_from_cm(
                "max_similar_topics_to_connect", cls.max_similar_topics_to_connect, int
            ),
            node_summary_forget_time_hours=get_param_from_cm(
                "node_summary_forget_time_hours",
                cls.node_summary_forget_time_hours,
                float,
            ),
            rpm_limit_delay_summary_sec=get_param_from_cm(
                "rpm_limit_delay_summary_sec", cls.rpm_limit_delay_summary_sec, float
            ),
            embedding_update_delay_sec=get_param_from_cm(
                "embedding_update_delay_sec", cls.embedding_update_delay_sec, float
            ),
            embedding_update_batch_size=get_param_from_cm(
                "embedding_update_batch_size", cls.embedding_update_batch_size, int
            ),
            max_topics_per_snippet=get_param_from_cm(
                "max_topics_per_snippet", cls.max_topics_per_snippet, int
            ),
            min_topics_per_snippet=get_param_from_cm(
                "min_topics_per_snippet", cls.min_topics_per_snippet, int
            ),
            retrieval_max_final_memories=get_param_from_cm(
                "retrieval_max_final_memories", cls.retrieval_max_final_memories, int
            ),
            retrieval_activation_depth=get_param_from_cm(
                "retrieval_activation_depth", cls.retrieval_activation_depth, int
            ),
            fast_retrieval_max_keywords=get_param_from_cm(
                "fast_retrieval_max_keywords", cls.fast_retrieval_max_keywords, int
            ),
            retrieval_input_max_topics=get_param_from_cm(
                "retrieval_input_max_topics", cls.retrieval_input_max_topics, int
            ),
            retrieval_input_compress_rate=get_param_from_cm(
                "retrieval_input_compress_rate",
                cls.retrieval_input_compress_rate,
                float,
            ),
            activation_decay_per_link=get_param_from_cm(
                "activation_decay_per_link", cls.activation_decay_per_link, float
            ),
            activation_link_decay_base=get_param_from_cm(
                "activation_link_decay_base", cls.activation_link_decay_base, float
            ),
            activation_min_threshold_for_spread=get_param_from_cm(
                "activation_min_threshold_for_spread",
                cls.activation_min_threshold_for_spread,
                float,
            ),
            retrieval_top_activated_nodes_to_scan=get_param_from_cm(
                "retrieval_top_activated_nodes_to_scan",
                cls.retrieval_top_activated_nodes_to_scan,
                int,
            ),
            retrieval_min_summary_similarity=get_param_from_cm(
                "retrieval_min_summary_similarity",
                cls.retrieval_min_summary_similarity,
                float,
            ),
            retrieval_max_candidates_for_llm_rerank=get_param_from_cm(
                "retrieval_max_candidates_for_llm_rerank",
                cls.retrieval_max_candidates_for_llm_rerank,
                int,
            ),
            rerank_target_selection_count=get_param_from_cm(
                "rerank_target_selection_count", cls.rerank_target_selection_count, int
            ),
            keyword_retrieval_node_similarity_threshold=get_param_from_cm(
                "keyword_retrieval_node_similarity_threshold",
                cls.keyword_retrieval_node_similarity_threshold,
                float,
            ),
            topic_calc_len_base=get_param_from_cm(
                "topic_calc_len_base", cls.topic_calc_len_base, float
            ),
            topic_calc_len_log_factor=get_param_from_cm(
                "topic_calc_len_log_factor", cls.topic_calc_len_log_factor, float
            ),
            topic_calc_len_short_thresh=get_param_from_cm(
                "topic_calc_len_short_thresh", cls.topic_calc_len_short_thresh, int
            ),
            topic_calc_len_very_short_val=get_param_from_cm(
                "topic_calc_len_very_short_val", cls.topic_calc_len_very_short_val, int
            ),
            topic_calc_info_baseline=get_param_from_cm(
                "topic_calc_info_baseline", cls.topic_calc_info_baseline, float
            ),
            topic_calc_info_scale=get_param_from_cm(
                "topic_calc_info_scale", cls.topic_calc_info_scale, float
            ),
            topic_calc_info_max_absolute=get_param_from_cm(
                "topic_calc_info_max_absolute", cls.topic_calc_info_max_absolute, int
            ),
            sample_time_gap_threshold_seconds=get_param_from_cm(
                "sample_time_gap_threshold_seconds",
                cls.sample_time_gap_threshold_seconds,
                int,
            ),
            sample_max_anchor_attempts=get_param_from_cm(
                "sample_max_anchor_attempts", cls.sample_max_anchor_attempts, int
            ),
            sample_min_snippet_messages=get_param_from_cm(
                "sample_min_snippet_messages", cls.sample_min_snippet_messages, int
            ),
            sample_max_snippet_messages=get_param_from_cm(
                "sample_max_snippet_messages", cls.sample_max_snippet_messages, int
            ),
        )
