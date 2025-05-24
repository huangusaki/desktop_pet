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


if __name__ == "__main__":
    print("Testing MemoryConfig with extended parameters...")
    import os
    import sys

    dummy_project_root = "dummy_project_for_mem_config_test"
    dummy_config_dir = os.path.join(dummy_project_root, "config")
    dummy_settings_file = os.path.join(dummy_config_dir, "settings.ini")
    dummy_src_dir = os.path.join(dummy_project_root, "src")
    dummy_utils_dir = os.path.join(dummy_src_dir, "utils")
    os.makedirs(dummy_utils_dir, exist_ok=True)
    os.makedirs(dummy_config_dir, exist_ok=True)
    config_manager_py_content = """
import configparser
import os
class ConfigManager:
    def __init__(self, config_file="config/settings.ini"):
        self.config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        # Simplified project root for dummy
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        actual_config_path = os.path.join(self.project_root, config_file)
        if not os.path.exists(actual_config_path):
            # Fallback if run from inside dummy_project_root/src/memory_system
            alt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            actual_config_path = os.path.join(alt_root, config_file)
            if os.path.exists(actual_config_path):
                 self.project_root = alt_root
            else:
                raise FileNotFoundError(f"Dummy ConfigManager: Config file not found at {actual_config_path} (tried from {self.project_root})")
        self.config.read(actual_config_path, encoding="utf-8")
        # print(f"Dummy ConfigManager loaded: {actual_config_path}")
    def get_memory_build_distribution(self): return tuple(map(float, self.config.get("MEMORY_SYSTEM", "BUILD_DISTRIBUTION").split(',')))
    def get_memory_build_sample_num(self): return self.config.getint("MEMORY_SYSTEM", "BUILD_SAMPLE_NUM")
    def get_memory_build_sample_length(self): return self.config.getint("MEMORY_SYSTEM", "BUILD_SAMPLE_LENGTH")
    def get_memory_compress_rate(self): return self.config.getfloat("MEMORY_SYSTEM", "COMPRESS_RATE")
    def get_memory_forget_time_hours(self): return self.config.getfloat("MEMORY_SYSTEM", "FORGET_TIME_HOURS")
    def get_memory_forget_percentage(self): return self.config.getfloat("MEMORY_SYSTEM", "FORGET_PERCENTAGE")
    def get_memory_ban_words(self): return [w.strip() for w in self.config.get("MEMORY_SYSTEM", "BAN_WORDS").split(',')]
    def get_memory_consolidate_percentage(self): return self.config.getfloat("MEMORY_SYSTEM", "CONSOLIDATE_PERCENTAGE")
    def get_memory_consolidation_similarity_threshold(self): return self.config.getfloat("MEMORY_SYSTEM", "CONSOLIDATION_SIMILARITY_THRESHOLD")
    def get_memory_llm_config(self, llm_type_key_base: str) -> dict: # llm_type_key_base is like "LLM_TOPIC_JUDGE"
        # In settings.ini, it's LLM_TOPIC_JUDGE_NICKNAME, so we need to adjust the key
        nickname_key = f"{llm_type_key_base}_NICKNAME" # e.g. LLM_TOPIC_JUDGE_NICKNAME
        nickname = self.config.get("MEMORY_LLMS", nickname_key, fallback=None)
        if not nickname: return {}
        section_name = f"MEMORY_LLM_{nickname}"
        if self.config.has_section(section_name):
            cfg = dict(self.config.items(section_name))
            # Convert relevant fields if needed (e.g. pri_in/out were removed)
            return cfg
        return {}
    # For fine-grained params, MemoryConfig's get_param will try these first if methods exist
    # If not, it falls back to reading from [MEMORY_SYSTEM_PARAMS] section.
    # Example:
    # def get_memory_max_memorized_time_per_msg(self):
    #     return self.config.getint("MEMORY_SYSTEM_PARAMS", "MAX_MEMORIZED_TIME_PER_MSG", fallback=3)
"""
    dummy_config_manager_file = os.path.join(dummy_utils_dir, "config_manager.py")
    with open(dummy_config_manager_file, "w", encoding="utf-8") as f:
        f.write(config_manager_py_content)
    settings_ini_content = """
[MEMORY_SYSTEM]
BUILD_DISTRIBUTION = 1,1,1,1,1,1
BUILD_SAMPLE_NUM = 1
BUILD_SAMPLE_LENGTH = 1
COMPRESS_RATE = 0.1
FORGET_TIME_HOURS = 1.0
FORGET_PERCENTAGE = 0.1
BAN_WORDS = a,b
CONSOLIDATE_PERCENTAGE = 0.1
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.1
[MEMORY_LLMS]
LLM_TOPIC_JUDGE_NICKNAME = test_judge
LLM_SUMMARY_BY_TOPIC_NICKNAME = test_summary
LLM_EMBEDDING_TOPIC_NICKNAME = test_embed
LLM_RE_RANK_NICKNAME = test_rerank
[MEMORY_LLM_test_judge]
name = judge_model_name
key = JUDGE_KEY
[MEMORY_LLM_test_summary]
name = summary_model_name
key = SUMMARY_KEY
[MEMORY_LLM_test_embed]
name = embed_model_name
key = EMBED_KEY
[MEMORY_LLM_test_rerank]
name = rerank_model_name
key = RERANK_KEY
; Section for fine-grained params if ConfigManager doesn't have specific getters
[MEMORY_SYSTEM_PARAMS]
MAX_MEMORIZED_TIME_PER_MSG = 4
TOPIC_SIMILARITY_THRESHOLD_FOR_CONNECTION = 0.75
MAX_SIMILAR_TOPICS_TO_CONNECT = 2
NODE_SUMMARY_FORGET_TIME_HOURS = 200.0
RPM_LIMIT_DELAY_SUMMARY_SEC = 1.5
EMBEDDING_UPDATE_DELAY_SEC = 0.05
EMBEDDING_UPDATE_BATCH_SIZE = 25
MAX_TOPICS_PER_SNIPPET = 6
RETRIEVAL_MAX_FINAL_MEMORIES = 3
RETRIEVAL_ACTIVATION_DEPTH = 2
FAST_RETRIEVAL_MAX_KEYWORDS = 3
RETRIEVAL_INPUT_MAX_TOPICS = 3
RETRIEVAL_INPUT_COMPRESS_RATE = 0.03
ACTIVATION_DECAY_PER_LINK = 0.6
ACTIVATION_LINK_DECAY_BASE = 0.4
ACTIVATION_MIN_THRESHOLD_FOR_SPREAD = 0.15
RETRIEVAL_TOP_ACTIVATED_NODES_TO_SCAN = 10
RETRIEVAL_MIN_SUMMARY_SIMILARITY = 0.7
RETRIEVAL_MAX_CANDIDATES_FOR_LLM_RERANK = 10
RERANK_TARGET_SELECTION_COUNT = 3
KEYWORD_RETRIEVAL_NODE_SIMILARITY_THRESHOLD = 0.85
"""
    with open(dummy_settings_file, "w", encoding="utf-8") as f:
        f.write(settings_ini_content)
    _original_file = __file__
    __file__ = os.path.join(
        dummy_project_root, "src", "memory_system", "memory_config.py"
    )
    sys.path.insert(0, os.path.join(dummy_project_root, "src"))
    try:
        from utils.config_manager import ConfigManager as DummyCfgMgr

        ConfigManager.__file__ = dummy_config_manager_file
        cfg_manager_instance = DummyCfgMgr(
            config_file=os.path.join("config", "settings.ini")
        )
        mem_cfg = MemoryConfig.from_config_manager(cfg_manager_instance)
        print(f"Loaded MemoryConfig: {mem_cfg}")
        print(
            f"Topic Similarity Threshold: {mem_cfg.topic_similarity_threshold_for_connection}"
        )
        print(f"LLM for Topic Judge: {mem_cfg.llm_topic_judge}")
        assert mem_cfg.max_memorized_time_per_msg == 4
        assert mem_cfg.topic_similarity_threshold_for_connection == 0.75
    except Exception as e:
        print(f"Error in MemoryConfig test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        __file__ = _original_file
        if os.path.join(dummy_project_root, "src") in sys.path:
            sys.path.remove(os.path.join(dummy_project_root, "src"))
        if os.path.exists(dummy_config_manager_file):
            os.remove(dummy_config_manager_file)
        if os.path.exists(dummy_utils_dir):
            os.rmdir(dummy_utils_dir)
        if os.path.exists(dummy_src_dir):
            os.rmdir(dummy_src_dir)
        if os.path.exists(dummy_settings_file):
            os.remove(dummy_settings_file)
        if os.path.exists(dummy_config_dir):
            os.rmdir(dummy_config_dir)
        if os.path.exists(dummy_project_root):
            os.rmdir(dummy_project_root)
