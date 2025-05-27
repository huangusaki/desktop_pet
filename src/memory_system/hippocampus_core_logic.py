import asyncio
import math
import time
import logging
import re
import networkx as nx
import uuid
import json
from datetime import datetime
from pymongo import UpdateOne
from pymongo.database import Database
from typing import Optional, List, Tuple, Set, Dict, Any
from .memory_config import MemoryConfig
from .hippocampus_utils import calculate_information_content, cosine_similarity
from .hippocampus_graph import MemoryGraph
from .hippocampus_io import EntorhinalCortex
from .hippocampus_processing import ParahippocampalGyrus
from ..utils.prompt_builder import PromptBuilder
from ..llm.llm_request import LLM_request, GeminiSDKResponse

logger = logging.getLogger("memory_system")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class Hippocampus:
    def __init__(self):
        self.memory_graph: MemoryGraph = MemoryGraph()
        self.config: Optional[MemoryConfig] = None
        self.db_instance: Optional[Database] = None
        self.chat_collection_name: Optional[str] = None
        self.pet_name_for_history_filter: Optional[str] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.llm_topic_judge: Optional[LLM_request] = None
        self.llm_summary_by_topic: Optional[LLM_request] = None
        self.llm_embedding_topic: Optional[LLM_request] = None
        self.llm_re_rank: Optional[LLM_request] = None
        self.entorhinal_cortex: Optional[EntorhinalCortex] = None
        self.parahippocampal_gyrus: Optional[ParahippocampalGyrus] = None
        self._nodes_needing_embedding_update: List[str] = []
        self._items_needing_summary_embedding_update: Dict[
            str, List[Tuple[str, str]]
        ] = {}
        self._db_graph_lock = asyncio.Lock()

    def initialize(
        self,
        memory_config: MemoryConfig,
        database_instance: Database,
        chat_history_collection_name: str,
        pet_name: str,
        prompt_builder: PromptBuilder,
        global_llm_params: Optional[Dict[str, Any]] = None,
    ):
        self.config = memory_config
        self.db_instance = database_instance
        self.chat_collection_name = chat_history_collection_name
        self.pet_name_for_history_filter = pet_name
        self.prompt_builder = prompt_builder
        logger.info("海马体开始初始化...")
        if not LLM_request:
            logger.error("LLM_request 未成功导入，LLM功能将不可用。")
        if self.config.llm_topic_judge and LLM_request:
            try:
                self.llm_topic_judge = LLM_request(
                    self.config.llm_topic_judge,
                    self.db_instance,
                    global_llm_params,
                    "memory_topic_judge",
                )
            except Exception as e:
                logger.error(f"初始化主题判断LLM失败: {e}", exc_info=True)
        if self.config.llm_summary_by_topic and LLM_request:
            try:
                self.llm_summary_by_topic = LLM_request(
                    self.config.llm_summary_by_topic,
                    self.db_instance,
                    global_llm_params,
                    "memory_hier_summary",
                )
            except Exception as e:
                logger.error(f"初始化层级摘要LLM失败: {e}", exc_info=True)
        if self.config.llm_embedding_topic and LLM_request:
            try:
                self.llm_embedding_topic = LLM_request(
                    self.config.llm_embedding_topic,
                    self.db_instance,
                    global_llm_params,
                    "memory_embedding",
                )
            except Exception as e:
                logger.error(f"初始化文本嵌入模型失败: {e}", exc_info=True)
        if self.config.llm_re_rank and LLM_request:
            try:
                self.llm_re_rank = LLM_request(
                    self.config.llm_re_rank,
                    self.db_instance,
                    global_llm_params,
                    "memory_re_rank",
                )
            except Exception as e:
                logger.error(f"初始化记忆重排LLM失败: {e}", exc_info=True)
        for llm_attr_name, name_in_cfg in [
            ("llm_topic_judge", "llm_topic_judge"),
            ("llm_summary_by_topic", "llm_summary_by_topic"),
            ("llm_embedding_topic", "llm_embedding_topic"),
            ("llm_re_rank", "llm_re_rank"),
        ]:
            llm_instance = getattr(self, llm_attr_name)
            if llm_instance:
                logger.info(
                    f"{name_in_cfg} LLM ({getattr(self.config, name_in_cfg, {}).get('name')}) 初始化成功。"
                )
            else:
                logger.warning(
                    f"{name_in_cfg} LLM 未能初始化 (配置: {getattr(self.config, name_in_cfg, {})})。"
                )
        self.entorhinal_cortex = EntorhinalCortex(self)
        self.parahippocampal_gyrus = ParahippocampalGyrus(self)
        logger.info("正在从数据库同步记忆图谱...")
        if self.entorhinal_cortex:
            self.entorhinal_cortex.sync_memory_from_db()
        else:
            logger.error("EntorhinalCortex 未初始化，无法从数据库同步记忆。")
        if self.llm_embedding_topic and (
            self._nodes_needing_embedding_update
            or self._items_needing_summary_embedding_update
        ):
            logger.info("检测到启动时有缺失的嵌入，将启动后台任务补充...")
            asyncio.create_task(self._update_missing_embeddings_task())
        elif not self.llm_embedding_topic:
            logger.warning("嵌入模型未配置，无法补充缺失嵌入。将清除待更新列表。")
            self._clear_pending_embeddings()
        else:
            logger.info("无需补充嵌入或嵌入模型未配置。")
        logger.info("海马体初始化完成。")

    async def _update_missing_embeddings_task(self):
        if not self.llm_embedding_topic or not self.config:
            logger.warning("嵌入模型或配置未初始化，无法更新缺失嵌入。")
            self._clear_pending_embeddings()
            return
        delay = self.config.embedding_update_delay_sec
        batch_size = self.config.embedding_update_batch_size
        nodes_to_proc = list(self._nodes_needing_embedding_update)
        self._nodes_needing_embedding_update.clear()
        if nodes_to_proc:
            logger.info(f"开始处理{len(nodes_to_proc)}个节点概念嵌入...")
            upd_c, fail_c = 0, 0
            bulk_db_upd_concept: List[UpdateOne] = []
            for i, concept in enumerate(nodes_to_proc):
                if concept not in self.memory_graph.G:
                    continue
                emb_vec = await self.get_embedding_async(
                    concept, request_type="memory_concept_embed_补"
                )
                if emb_vec:
                    self.memory_graph.G.nodes[concept]["embedding"] = emb_vec
                    self.memory_graph.G.nodes[concept][
                        "last_modified"
                    ] = datetime.now().timestamp()
                    bulk_db_upd_concept.append(
                        UpdateOne(
                            {"concept": concept},
                            {
                                "$set": {
                                    "embedding": emb_vec,
                                    "last_modified": self.memory_graph.G.nodes[concept][
                                        "last_modified"
                                    ],
                                }
                            },
                        )
                    )
                    upd_c += 1
                else:
                    fail_c += 1
                if (
                    (i + 1) % batch_size == 0
                    and bulk_db_upd_concept
                    and self.db_instance is not None
                ):
                    try:
                        async with self._db_graph_lock:
                            self.db_instance.graph_data_nodes.bulk_write(
                                bulk_db_upd_concept, ordered=False
                            )
                        bulk_db_upd_concept = []
                    except Exception as e:
                        logger.error(f"批量更新概念嵌入DB出错:{e}")
                await asyncio.sleep(delay)
            if bulk_db_upd_concept and self.db_instance is not None:
                try:
                    async with self._db_graph_lock:
                        self.db_instance.graph_data_nodes.bulk_write(
                            bulk_db_upd_concept, ordered=False
                        )
                except Exception as e:
                    logger.error(f"最后批量更新概念嵌入DB出错:{e}")
            logger.info(f"概念嵌入更新完成。成功:{upd_c},失败:{fail_c}。")
        items_to_proc_hierarchical = dict(self._items_needing_summary_embedding_update)
        self._items_needing_summary_embedding_update.clear()
        if items_to_proc_hierarchical:
            total_sum_levels_to_update = sum(
                len(v) for v in items_to_proc_hierarchical.values()
            )
            logger.info(
                f"开始处理概念中缺失的 {total_sum_levels_to_update} 个层级摘要嵌入..."
            )
            upd_s, fail_s = 0, 0
            for concept, event_level_pairs in items_to_proc_hierarchical.items():
                if not event_level_pairs or concept not in self.memory_graph.G:
                    continue
                node_tuple = self.memory_graph.get_dot(concept)
                if not node_tuple:
                    logger.warning(f"补摘要嵌入时，无法获取概念'{concept}'的节点数据。")
                    continue
                _c, node_data = node_tuple
                current_memory_events = list(node_data.get("memory_events", []))
                if not current_memory_events:
                    continue
                node_modified_in_graph_and_db = False
                event_id_to_idx_map = {
                    event.get("event_id"): i
                    for i, event in enumerate(current_memory_events)
                }
                for event_id, level_name in event_level_pairs:
                    event_idx = event_id_to_idx_map.get(event_id)
                    if event_idx is None or not (
                        0 <= event_idx < len(current_memory_events)
                    ):
                        logger.warning(
                            f"补摘要嵌入时，概念'{concept}'中未找到事件ID'{event_id}'。"
                        )
                        continue
                    target_event = current_memory_events[event_idx]
                    hier_summaries = target_event.get("hierarchical_summaries", {})
                    summary_tuple = hier_summaries.get(level_name)
                    if (
                        summary_tuple
                        and isinstance(summary_tuple, tuple)
                        and len(summary_tuple) == 2
                    ):
                        text_to_embed, current_embedding = summary_tuple
                        if current_embedding is None and text_to_embed:
                            summary_emb_vec = await self.get_embedding_async(
                                text_to_embed,
                                request_type=f"memory_sum_embed_补_{level_name}",
                            )
                            if summary_emb_vec:
                                hier_summaries[level_name] = (
                                    text_to_embed,
                                    summary_emb_vec,
                                )
                                upd_s += 1
                                node_modified_in_graph_and_db = True
                            else:
                                fail_s += 1
                            await asyncio.sleep(delay)
                if node_modified_in_graph_and_db and self.db_instance is not None:
                    self.memory_graph.G.nodes[concept][
                        "memory_events"
                    ] = current_memory_events
                    self.memory_graph.G.nodes[concept][
                        "last_modified"
                    ] = datetime.now().timestamp()
                    try:
                        async with self._db_graph_lock:
                            self.db_instance.graph_data_nodes.update_one(
                                {"concept": concept},
                                {
                                    "$set": {
                                        "memory_events": current_memory_events,
                                        "last_modified": self.memory_graph.G.nodes[
                                            concept
                                        ]["last_modified"],
                                    }
                                },
                            )
                    except Exception as e:
                        logger.error(f"更新节点'{concept}'层级摘要嵌入到DB出错:{e}")
            logger.info(f"层级摘要嵌入更新完成。成功:{upd_s},失败:{fail_s}。")
        logger.info("后台嵌入更新任务结束。")

    def _clear_pending_embeddings(self):
        self._nodes_needing_embedding_update.clear()
        self._items_needing_summary_embedding_update.clear()

    async def get_embedding_async(
        self, text: str, user_id: str = "system", request_type: str = "memory_embedding"
    ) -> Optional[List[float]]:
        if not self.llm_embedding_topic:
            logger.warning("嵌入LLM未初始化。")
            return None
        if not text or not text.strip():
            return None
        try:
            emb_vec = await self.llm_embedding_topic.get_embedding_async(
                text, user_id=user_id, request_type=request_type
            )
            if isinstance(emb_vec, list) and all(
                isinstance(x, (int, float)) for x in emb_vec
            ):
                return emb_vec
            logger.error(
                f"嵌入格式不正确({request_type}):{type(emb_vec)} for '{text[:50]}...'"
            )
            return None
        except Exception as e:
            logger.error(
                f"获取嵌入出错({request_type}) for '{text[:50]}...':{e}", exc_info=True
            )
            return None

    def get_all_node_names(self) -> List[str]:
        return list(self.memory_graph.G.nodes())

    def calculate_node_hash(
        self, concept_name: str, memory_events_list: List[Dict]
    ) -> int:
        hash_parts = [f"c:{concept_name}"]
        sorted_memory_events = sorted(
            memory_events_list, key=lambda ev: ev.get("event_id", "")
        )
        for event in sorted_memory_events:
            event_id = event.get("event_id", "unknown_event_id")
            hs = event.get("hierarchical_summaries", {})
            event_summary_parts = [f"ev_id:{event_id}"]
            sorted_levels = sorted(hs.keys())
            for level_name in sorted_levels:
                summary_tuple = hs.get(level_name)
                if (
                    summary_tuple
                    and isinstance(summary_tuple, tuple)
                    and len(summary_tuple) > 0
                ):
                    summary_text = summary_tuple[0]
                    event_summary_parts.append(f"{level_name}_txt:{summary_text}")
            hash_parts.append("|".join(event_summary_parts))
        full_hash_string = "||".join(hash_parts)
        return hash(full_hash_string)

    def calculate_edge_hash(self, s: str, t: str) -> int:
        return hash(f"e:{'-'.join(sorted([s,t]))}")

    def _create_hierarchical_summary_prompt(
        self, text_to_summarize: str, time_info: str, topic: str
    ) -> str:
        if not self.prompt_builder:
            logger.error("PromptBuilder未初始化，无法创建层级摘要提示。")
            return ""
        return self.prompt_builder.build_hierarchical_summary_prompt(
            text_to_summarize=text_to_summarize, time_info=time_info, topic=topic
        )

    async def generate_hierarchical_summaries(
        self, text_to_summarize: str, time_info: str, topic: str
    ) -> Optional[Dict[str, str]]:
        if not self.llm_summary_by_topic:
            logger.warning("层级摘要LLM未初始化。")
            return None
        if not text_to_summarize or not text_to_summarize.strip() or not topic:
            return None
        prompt = self._create_hierarchical_summary_prompt(
            text_to_summarize, time_info, topic
        )
        if not prompt:
            return None
        try:
            resp: Optional[GeminiSDKResponse] = (
                await self.llm_summary_by_topic.generate_response_async(
                    prompt, request_type="memory_gen_hier_summary"
                )
            )
            if resp and resp.content and resp.content.strip():
                try:
                    cleaned_content = resp.content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content[3:]
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]
                    hier_summaries = json.loads(cleaned_content.strip())
                    if isinstance(hier_summaries, dict):
                        valid_structure = True
                        for k, v in hier_summaries.items():
                            if not isinstance(k, str):
                                valid_structure = False
                                break
                            if k == "L3_details_list":
                                if not (
                                    isinstance(v, str)
                                    or (
                                        isinstance(v, list)
                                        and all(isinstance(item, str) for item in v)
                                    )
                                ):
                                    valid_structure = False
                                    break
                            elif not isinstance(v, str):
                                valid_structure = False
                                break
                        if valid_structure:
                            expected_keys = {
                                "L0_keywords",
                                "L1_core_sentence",
                                "L2_paragraph",
                                "L3_details_list",
                            }
                            if expected_keys.issubset(hier_summaries.keys()):
                                if isinstance(
                                    hier_summaries.get("L3_details_list"), list
                                ):
                                    hier_summaries["L3_details_list"] = "\n".join(
                                        hier_summaries["L3_details_list"]
                                    )
                                return hier_summaries
                            else:
                                logger.warning(
                                    f"LLM为主题'{topic}'返回的层级摘要JSON缺少预期键。得到: {hier_summaries.keys()}"
                                )
                                return None
                        else:
                            logger.warning(
                                f"LLM为主题'{topic}'返回的层级摘要JSON中键或值类型不正确。内容: {cleaned_content}"
                            )
                            return None
                    else:
                        logger.warning(
                            f"LLM为主题'{topic}'返回的层级摘要解析后非字典类型。内容: {cleaned_content}"
                        )
                        return None
                except json.JSONDecodeError as e:
                    logger.error(
                        f"LLM为主题'{topic}'返回的层级摘要JSON解析失败: {e}。内容: {resp.content.strip()}"
                    )
                    return None
        except Exception as e:
            logger.error(f"为主题'{topic}'生成层级摘要出错:{e}", exc_info=True)
            return None
        return None

    def _create_find_topic_prompt(self, txt: str, num: int) -> str:
        if not self.prompt_builder:
            logger.error("PromptBuilder未初始化，无法创建主题提取提示。")
            return ""
        return self.prompt_builder.build_find_topics_prompt(
            text_to_analyze=txt, num_topics=num
        )

    async def extract_topics_from_text(self, txt: str, num: int) -> List[str]:
        if not self.llm_topic_judge:
            logger.warning("主题判断LLM未初始化。")
            return []
        if not txt or not txt.strip():
            return []
        prompt = self._create_find_topic_prompt(txt, num)
        if not prompt:
            return []
        try:
            resp: Optional[GeminiSDKResponse] = (
                await self.llm_topic_judge.generate_response_async(
                    prompt, request_type="memory_extract_topics"
                )
            )
            if not resp or not resp.content:
                logger.warning("LLM提取主题返回空。")
                return []
            extracted = re.findall(r"<([^>]+)>", resp.content)
            if not extracted:
                logger.info(f"LLM未提取到任何内容（空列表）。响应: '{resp.content}'")
                return []
            if len(extracted) == 1 and extracted[0].strip().lower() == "none":
                logger.info(
                    f"LLM提取到<none>，判定为无有效主题。响应: '{resp.content}'"
                )
                return []
            topics = set()
            for item in extracted:
                split_topics = re.split(r"[,，、]", item)
                for t in split_topics:
                    cleaned_topic = t.strip()
                    if cleaned_topic:
                        topics.add(cleaned_topic)
            logger.debug(f"LLM提取清洗后主题: {list(topics)}")
            return list(topics)
        except Exception as e:
            logger.error(f"提取主题异常:{e}", exc_info=True)
            return []

    def calculate_topic_num(self, txt: str, rate: Optional[float] = None) -> int:
        if not txt or not self.config:
            return 1
        lines = txt.count("\n") + 1
        info = calculate_information_content(txt)
        base_topics_for_length = self.config.topic_calc_len_base
        log_lines_factor = self.config.topic_calc_len_log_factor
        short_text_line_threshold = self.config.topic_calc_len_short_thresh
        topics_for_very_short_text = self.config.topic_calc_len_very_short_val
        info_baseline = self.config.topic_calc_info_baseline
        info_scale = self.config.topic_calc_info_scale
        max_topics_from_info_absolute = self.config.topic_calc_info_max_absolute
        min_overall_topics = self.config.min_topics_per_snippet
        max_overall_topics = self.config.max_topics_per_snippet
        if lines <= short_text_line_threshold:
            len_topic_est = float(topics_for_very_short_text)
        else:
            len_topic_est = (
                base_topics_for_length + math.log10(lines) * log_lines_factor
            )
        len_topic = max(1, math.ceil(len_topic_est))
        info_topic_est = (info - info_baseline) * info_scale
        info_topic = max(1, math.ceil(info_topic_est))
        info_topic = min(info_topic, max_topics_from_info_absolute)
        combined_estimate = max(len_topic, info_topic)
        final_num = max(min_overall_topics, math.ceil(combined_estimate))
        final_num = min(max_overall_topics, final_num)
        logger.debug(
            f"Calculate topic num: lines={lines}, info={info:.2f} -> "
            f"len_topic_est={len_topic_est:.2f} (ceil->{len_topic}), "
            f"info_topic_est={info_topic_est:.2f} (ceil->{info_topic}) -> "
            f"combined_est={combined_estimate:.2f} -> final_num={int(final_num)}"
        )
        return int(final_num)

    async def get_memory_from_keyword(
        self, kw: str, depth: int = 2, summary_level: str = "L1_core_sentence"
    ) -> List[Tuple[str, List[str], float]]:
        if not kw or not kw.strip() or not self.llm_embedding_topic or not self.config:
            return []
        kw_emb = await self.get_embedding_async(
            kw, request_type="memory_kw_embed_retrieve"
        )
        if not kw_emb:
            logger.warning(f"无法获取关键词'{kw}'嵌入。")
            return []
        similar_nodes: List[Tuple[str, List[str], float]] = []
        for node_name, attr in self.memory_graph.G.nodes(data=True):
            node_emb = attr.get("embedding")
            if node_emb and isinstance(node_emb, list):
                try:
                    sim_to_concept = cosine_similarity(kw_emb, node_emb)
                    if (
                        sim_to_concept
                        >= self.config.keyword_retrieval_node_similarity_threshold
                    ):
                        node_tuple = self.memory_graph.get_dot(node_name)
                        if node_tuple:
                            _concept, node_data = node_tuple
                            summaries_from_events = []
                            for event in node_data.get("memory_events", []):
                                hs = event.get("hierarchical_summaries", {})
                                summary_data = hs.get(summary_level)
                                if (
                                    summary_data
                                    and isinstance(summary_data, tuple)
                                    and len(summary_data) > 0
                                    and isinstance(summary_data[0], str)
                                ):
                                    summaries_from_events.append(summary_data[0])
                            if summaries_from_events:
                                similar_nodes.append(
                                    (node_name, summaries_from_events, sim_to_concept)
                                )
                except Exception as e:
                    logger.error(f"计算关键词'{kw}'与节点'{node_name}'相似度出错:{e}")
        similar_nodes.sort(key=lambda x: x[2], reverse=True)
        return similar_nodes

    def _create_bulk_relevance_check_prompt(
        self, txt: str, candidates: List[Tuple[str, str, float]]
    ) -> str:
        if not self.prompt_builder or not self.config:
            logger.error("PromptBuilder或Config未初始化，无法创建重排提示。")
            return ""
        return self.prompt_builder.build_bulk_relevance_check_prompt(
            current_dialog_text=txt,
            candidate_memories=candidates,
            target_selection_count=self.config.rerank_target_selection_count,
        )

    async def get_memory_from_text(
        self,
        txt: str,
        num: Optional[int] = None,
        depth: Optional[int] = None,
        fast_kw: bool = False,
        retrieval_summary_level: str = "L1_core_sentence",
        output_summary_level: str = "L2_paragraph",
    ) -> List[Tuple[str, str]]:
        if not txt or not txt.strip() or not self.config:
            return []
        start_t = time.time()
        max_mem = num if num is not None else self.config.retrieval_max_final_memories
        act_depth = (
            depth if depth is not None else self.config.retrieval_activation_depth
        )
        kws: List[str] = []
        if fast_kw:
            try:
                import jieba

                stopwords = set(
                    self.config.memory_ban_words + list("的了和是就都而及或个吧啦呀吗")
                )
                kws = list(
                    set(
                        w
                        for w in jieba.cut(txt, cut_all=False)
                        if len(w) > 1 and w not in stopwords
                    )
                )[: self.config.fast_retrieval_max_keywords]
            except ImportError:
                logger.warning("jieba 未安装，fast_kw 将回退到 LLM 提取关键词。")
                fast_kw = False
            except Exception as e:
                logger.error(f"jieba 分词出错: {e}，fast_kw 将回退。")
                fast_kw = False
        if not fast_kw:
            topic_n = min(
                self.config.retrieval_input_max_topics,
                max(
                    1,
                    self.calculate_topic_num(
                        txt, self.config.retrieval_input_compress_rate
                    ),
                ),
            )
            kws = await self.extract_topics_from_text(txt, topic_n)
        if not kws:
            logger.info("未提取到有效关键词用于记忆检索。")
            return []
        seeds = [kw for kw in kws if kw in self.memory_graph.G]
        if not seeds:
            logger.info("提取关键词在记忆图中均不存在。")
            return []
        act_map: Dict[str, float] = {}
        for seed in seeds:
            q: List[Tuple[str, float, int]] = [(seed, 1.0, 0)]
            visited_spread: Set[str] = {seed}
            head = 0
            while head < len(q):
                curr_n, curr_act, curr_d = q[head]
                head += 1
                act_map[curr_n] = act_map.get(curr_n, 0.0) + curr_act
                if curr_d >= act_depth:
                    continue
                try:
                    for neighbor in self.memory_graph.G.neighbors(curr_n):
                        if neighbor not in visited_spread:
                            edge_data = self.memory_graph.G[curr_n][neighbor]
                            strength = edge_data.get("strength", 1)
                            decay_base = self.config.activation_link_decay_base
                            eff_strength = max(1, strength)
                            actual_decay = decay_base**eff_strength
                            new_act = curr_act * (1.0 - actual_decay)
                            if (
                                new_act
                                > self.config.activation_min_threshold_for_spread
                            ):
                                visited_spread.add(neighbor)
                                q.append((neighbor, new_act, curr_d + 1))
                except nx.NetworkXError as e:
                    logger.warning(f"激活扩散时访问邻居节点 '{curr_n}' 出错: {e}")
                except Exception as e:
                    logger.error(
                        f"激活扩散时发生未知错误在节点 '{curr_n}': {e}", exc_info=True
                    )
        if not act_map:
            logger.info("激活扩散未能激活任何节点。")
            return []
        sorted_act_nodes = sorted(act_map.items(), key=lambda x: x[1], reverse=True)
        if not self.llm_embedding_topic:
            logger.warning("嵌入模型不可用，无法进行基于摘要相似度的排序。")
            return []
        txt_emb = await self.get_embedding_async(
            txt, request_type="memory_input_embed_retrieve"
        )
        if not txt_emb:
            logger.warning("无法获取输入文本嵌入，无法进行基于摘要相似度的排序。")
            return []
        candidates_for_rerank: List[Tuple[str, str, str, float, str]] = []
        top_n_scan = self.config.retrieval_top_activated_nodes_to_scan
        min_sum_sim = self.config.retrieval_min_summary_similarity
        for node_name, _act_score in sorted_act_nodes[:top_n_scan]:
            node_tuple = self.memory_graph.get_dot(node_name)
            if not node_tuple:
                continue
            _concept, node_data = node_tuple
            for event in node_data.get("memory_events", []):
                hs = event.get("hierarchical_summaries", {})
                event_id = event.get(
                    "event_id", f"unknown_event_{uuid.uuid4().hex[:8]}"
                )
                retrieval_summary_data = hs.get(retrieval_summary_level)
                output_summary_data = hs.get(output_summary_level)
                if (
                    not retrieval_summary_data
                    or not isinstance(retrieval_summary_data, tuple)
                    or len(retrieval_summary_data) < 2
                    or not retrieval_summary_data[0]
                    or not retrieval_summary_data[1]
                ):
                    continue
                retrieval_s_text, retrieval_s_emb = retrieval_summary_data
                output_s_text_str = ""
                if (
                    output_summary_data
                    and isinstance(output_summary_data, tuple)
                    and len(output_summary_data) > 0
                    and isinstance(output_summary_data[0], str)
                    and output_summary_data[0].strip()
                ):
                    output_s_text_str = output_summary_data[0]
                elif retrieval_s_text.strip():
                    output_s_text_str = retrieval_s_text
                else:
                    logger.warning(
                        f"在节点 {node_name} 事件 {event_id} 中，无法获取有效的 output({output_summary_level}) 或 retrieval({retrieval_summary_level}) 摘要文本。"
                    )
                    continue
                if retrieval_s_emb and isinstance(retrieval_s_emb, list):
                    try:
                        sim = cosine_similarity(txt_emb, retrieval_s_emb)
                        if sim >= min_sum_sim:
                            candidates_for_rerank.append(
                                (
                                    node_name,
                                    event_id,
                                    retrieval_s_text,
                                    sim,
                                    output_s_text_str,
                                )
                            )
                    except Exception as e:
                        logger.error(
                            f"计算摘要相似度时出错 (节点 {node_name}, 事件 {event_id}): {e}"
                        )
                        pass
        if not candidates_for_rerank:
            logger.info(
                f"无摘要 (层级 {retrieval_summary_level}) 通过相似度阈值 {min_sum_sim:.2f}。"
            )
            return []
        candidates_for_rerank.sort(key=lambda x: x[3], reverse=True)
        to_llm_rerank_input_full = candidates_for_rerank[
            : self.config.retrieval_max_candidates_for_llm_rerank
        ]
        to_llm_rerank_prompt_candidates = [
            (topic, retrieval_text, score)
            for topic, _eid, retrieval_text, score, _out_text in to_llm_rerank_input_full
        ]
        candidates_output_map: Dict[Tuple[str, str], str] = {
            (cand_topic, cand_event_id): output_text
            for cand_topic, cand_event_id, _r_text, _score, output_text in to_llm_rerank_input_full
        }
        final_mem_tuples: List[Tuple[str, str]] = []
        if self.llm_re_rank and to_llm_rerank_prompt_candidates:
            logger.info(
                f"LLM({self.llm_re_rank.model_name})重排{len(to_llm_rerank_prompt_candidates)}条候选记忆 (基于 {retrieval_summary_level} 摘要)..."
            )
            prompt = self._create_bulk_relevance_check_prompt(
                txt, to_llm_rerank_prompt_candidates
            )
            if not prompt:
                logger.warning("LLM重排提示创建失败, 回退到原始相似度排序。")
                for (
                    topic,
                    event_id,
                    _r_text,
                    _score,
                    _o_text_mapped_by_key,
                ) in to_llm_rerank_input_full:
                    output_text_for_final = candidates_output_map.get((topic, event_id))
                    if output_text_for_final:
                        final_mem_tuples.append((topic, output_text_for_final))
            else:
                try:
                    resp = await self.llm_re_rank.generate_response_async(
                        prompt, request_type="memory_rerank"
                    )
                    llm_order_str = resp.content if resp else ""
                    if llm_order_str.lower().strip() == "无":
                        logger.info("LLM判断无相关记忆。")
                    elif llm_order_str:
                        try:
                            indices = [
                                int(i.strip()) - 1
                                for i in re.sub(r"[^\d,]", "", llm_order_str).split(",")
                                if i.strip().isdigit()
                            ]
                            for idx in indices:
                                if 0 <= idx < len(to_llm_rerank_input_full):
                                    (
                                        selected_topic,
                                        selected_event_id,
                                        _selected_retrieval_text,
                                        _selected_score,
                                        _selected_output_text_direct,
                                    ) = to_llm_rerank_input_full[idx]
                                    output_text_for_final = candidates_output_map.get(
                                        (selected_topic, selected_event_id)
                                    )
                                    if output_text_for_final:
                                        final_mem_tuples.append(
                                            (selected_topic, output_text_for_final)
                                        )
                            logger.info(
                                f"LLM成功重排并选择了{len(final_mem_tuples)}条记忆。"
                            )
                        except Exception as e:
                            logger.error(
                                f"解析LLM重排索引失败:'{llm_order_str}' ({e}). 回退到原始相似度排序。"
                            )
                            for (
                                topic,
                                event_id,
                                _r_text,
                                _score,
                                _o_text_mapped,
                            ) in to_llm_rerank_input_full:
                                output_text_for_final = candidates_output_map.get(
                                    (topic, event_id)
                                )
                                if output_text_for_final:
                                    final_mem_tuples.append(
                                        (topic, output_text_for_final)
                                    )
                    else:
                        logger.warning("LLM重排响应为空, 回退到原始相似度排序。")
                        for (
                            topic,
                            event_id,
                            _r_text,
                            _score,
                            _o_text_mapped,
                        ) in to_llm_rerank_input_full:
                            output_text_for_final = candidates_output_map.get(
                                (topic, event_id)
                            )
                            if output_text_for_final:
                                final_mem_tuples.append((topic, output_text_for_final))
                except Exception as e:
                    logger.error(f"LLM重排调用失败:{e}", exc_info=True)
                    for (
                        topic,
                        event_id,
                        _r_text,
                        _score,
                        _o_text_mapped,
                    ) in to_llm_rerank_input_full:
                        output_text_for_final = candidates_output_map.get(
                            (topic, event_id)
                        )
                        if output_text_for_final:
                            final_mem_tuples.append((topic, output_text_for_final))
        else:
            if not self.llm_re_rank:
                logger.debug("重排LLM未配置，使用初步筛选结果。")
            for (
                topic,
                event_id,
                _r_text,
                _score,
                _o_text_mapped,
            ) in to_llm_rerank_input_full:
                output_text_for_final = candidates_output_map.get((topic, event_id))
                if output_text_for_final:
                    final_mem_tuples.append((topic, output_text_for_final))
        seen_summaries, unique_final_mems = set(), []
        for topic_name, summary_text in final_mem_tuples:
            if summary_text not in seen_summaries:
                seen_summaries.add(summary_text)
                unique_final_mems.append((topic_name, summary_text))
            if len(unique_final_mems) >= max_mem:
                break
        logger.info(
            f"记忆检索完成 (基于 {retrieval_summary_level}, 输出 {output_summary_level}), 找到{len(unique_final_mems)}条。耗时:{time.time()-start_t:.3f}s"
        )
        return unique_final_mems

    async def get_activation_score_from_text(
        self, txt: str, depth: int = 3, fast_kw: bool = False
    ) -> float:
        if not txt or not txt.strip() or not self.config:
            return 0.0
        kws: List[str] = []
        if fast_kw:
            try:
                import jieba

                stopwords = set(
                    self.config.memory_ban_words + list("的了和是就都而及或个吧啦呀吗")
                )
                kws = list(
                    set(
                        w
                        for w in jieba.cut(txt, cut_all=False)
                        if len(w) > 1 and w not in stopwords
                    )
                )[: self.config.fast_retrieval_max_keywords]
            except ImportError:
                logger.warning(
                    "jieba 未安装，fast_kw (激活分数) 将回退到 LLM 提取关键词。"
                )
                fast_kw = False
            except Exception as e:
                logger.error(f"jieba 分词 (激活分数) 出错: {e}，fast_kw 将回退。")
                fast_kw = False
        if not fast_kw:
            topic_n = min(
                self.config.retrieval_input_max_topics,
                max(
                    1,
                    self.calculate_topic_num(
                        txt, self.config.retrieval_input_compress_rate
                    ),
                ),
            )
            kws = await self.extract_topics_from_text(txt, topic_n)
        if not kws:
            return 0.0
        seeds = [kw for kw in kws if kw in self.memory_graph.G]
        if not seeds:
            return 0.0
        act_map: Dict[str, float] = {}
        for seed in seeds:
            q: List[Tuple[str, float, int]] = [(seed, 1.0, 0)]
            visited_spread: Set[str] = {seed}
            head = 0
            while head < len(q):
                curr_n, curr_act, curr_d = q[head]
                head += 1
                act_map[curr_n] = act_map.get(curr_n, 0.0) + curr_act
                if curr_d >= depth:
                    continue
                try:
                    for neighbor in self.memory_graph.G.neighbors(curr_n):
                        if neighbor not in visited_spread:
                            edge_data = self.memory_graph.G[curr_n][neighbor]
                            strength = edge_data.get("strength", 1)
                            decay_base = self.config.activation_link_decay_base
                            eff_strength = max(1, strength)
                            actual_decay = decay_base**eff_strength
                            new_act = curr_act * (1.0 - actual_decay)
                            if (
                                new_act
                                > self.config.activation_min_threshold_for_spread
                            ):
                                visited_spread.add(neighbor)
                                q.append((neighbor, new_act, curr_d + 1))
                except nx.NetworkXError as e:
                    logger.warning(f"计算激活分数时访问邻居节点 '{curr_n}' 出错: {e}")
                except Exception as e:
                    logger.error(
                        f"计算激活分数时发生未知错误在节点 '{curr_n}': {e}",
                        exc_info=True,
                    )
        if not act_map:
            return 0.0
        total_act = sum(act_map.values())
        num_active = len(act_map)
        total_nodes = len(self.memory_graph.G.nodes())
        if total_nodes == 0:
            return 0.0
        prop_active = num_active / total_nodes
        avg_act_active = total_act / num_active if num_active > 0 else 0.0
        score = avg_act_active * prop_active
        return max(0.0, min(1.0, score))
