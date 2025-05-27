import asyncio
import random
import time
import uuid
import logging
from .memory_config import MemoryConfig
from .hippocampus_graph import MemoryGraph
from .hippocampus_utils import calculate_information_content, cosine_similarity
from datetime import datetime
from typing import Optional, List, Tuple, Set, Dict, Any, TYPE_CHECKING
from itertools import combinations

logger = logging.getLogger("memory_system.processing")
if not logger.hasHandlers():
    parent_logger = logging.getLogger("memory_system")
    if parent_logger.hasHandlers():
        logger.handlers = parent_logger.handlers
        logger.setLevel(parent_logger.level)
        logger.propagate = False
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.propagate = False
if TYPE_CHECKING:
    from .hippocampus_core_logic import Hippocampus
    from .hippocampus_io import EntorhinalCortex


class ParahippocampalGyrus:
    def __init__(self, hippocampus: "Hippocampus"):
        self.hippocampus = hippocampus
        self.memory_graph: MemoryGraph = hippocampus.memory_graph
        self.config: MemoryConfig = hippocampus.config

    async def extract_topics_and_find_similar(
        self, input_text: str, compress_rate: Optional[float] = None
    ) -> Tuple[str, Dict[str, List[float]], Dict[str, List[Tuple[str, float]]]]:
        if not input_text or not input_text.strip():
            return input_text, {}, {}
        effective_compress_rate = (
            compress_rate
            if compress_rate is not None
            else self.config.memory_compress_rate
        )
        topic_num = self.hippocampus.calculate_topic_num(
            input_text, effective_compress_rate
        )
        topics = await self.hippocampus.extract_topics_from_text(input_text, topic_num)
        filtered_topics = [
            t for t in topics if not any(kw in t for kw in self.config.memory_ban_words)
        ]
        if not filtered_topics:
            return input_text, {}, {}
        topic_embeddings_map: Dict[str, List[float]] = {}
        similar_topics_map: Dict[str, List[Tuple[str, float]]] = {}
        existing_nodes_with_embed = {
            node: data.get("embedding")
            for node, data in self.memory_graph.G.nodes(data=True)
            if data.get("embedding") and isinstance(data.get("embedding"), list)
        }
        embed_tasks = [
            self.hippocampus.get_embedding_async(t, request_type="memory_topic_embed")
            for t in filtered_topics
        ]
        results = await asyncio.gather(*embed_tasks, return_exceptions=True)
        for topic, emb_or_err in zip(filtered_topics, results):
            if isinstance(emb_or_err, Exception) or emb_or_err is None:
                logger.warning(f"无法为新主题 '{topic}' 获取嵌入: {emb_or_err}")
                continue
            topic_embeddings_map[topic] = emb_or_err
            similar_found = []
            for existing_topic, existing_emb in existing_nodes_with_embed.items():
                if topic == existing_topic or not existing_emb:
                    continue
                try:
                    sim = cosine_similarity(emb_or_err, existing_emb)
                    if sim >= self.config.topic_similarity_threshold_for_connection:
                        similar_found.append((existing_topic, sim))
                except Exception as e:
                    logger.error(f"计算主题相似度({topic},{existing_topic})出错: {e}")
            similar_found.sort(key=lambda x: x[1], reverse=True)
            similar_topics_map[topic] = similar_found[
                : self.config.max_similar_topics_to_connect
            ]
        return input_text, topic_embeddings_map, similar_topics_map

    async def operation_build_memory(self):
        start_time = time.time()
        logger.info("开始构建记忆 (层级化)...")
        if not self.hippocampus.entorhinal_cortex:
            logger.error("EntorhinalCortex 未初始化，无法构建记忆。")
            return
        memory_samples = self.hippocampus.entorhinal_cortex.get_memory_sample()
        if not memory_samples:
            logger.info("无记忆样本，跳过构建。")
            return
        added_nodes_concepts = set()
        connected_edges_pairs = set()
        processed_sample_count = 0
        for i, messages_in_sample in enumerate(memory_samples):
            sample_process_start_time = time.time()
            logger.info(
                f"处理样本 {i+1}/{len(memory_samples)}. 原始消息数: {len(messages_in_sample) if messages_in_sample else 'None'}"
            )
            if not messages_in_sample:
                logger.info(f"样本 {i+1} 为空，跳过。")
                continue
            input_text_for_sample, earliest_ts, latest_ts, valid_msgs_count = (
                "",
                float("inf"),
                float("-inf"),
                0,
            )
            for msg_idx, msg_data in enumerate(messages_in_sample):
                txt, ts = msg_data.get("message_text", ""), msg_data.get("timestamp")
                sender = msg_data.get("sender")
                if (
                    isinstance(txt, str)
                    and txt.strip()
                    and isinstance(ts, (int, float))
                ):
                    input_text_for_sample += f"{sender}:{txt}\n"
                    earliest_ts = min(earliest_ts, ts)
                    latest_ts = max(latest_ts, ts)
                    valid_msgs_count += 1
            if not input_text_for_sample.strip() or valid_msgs_count == 0:
                logger.info(f"样本 {i+1} 无有效文本内容或消息，跳过LLM处理。")
                continue
            processed_sample_count += 1
            time_info_for_summary = "时间未知"
            if earliest_ts != float("inf"):
                e_dt, l_dt = datetime.fromtimestamp(
                    earliest_ts
                ), datetime.fromtimestamp(latest_ts)
                time_info_for_summary = (
                    f"{e_dt.year}年, {e_dt.strftime('%m月%d日 %H:%M')}到{l_dt.strftime('%m月%d日 %H:%M')}"
                    if e_dt.year == l_dt.year
                    else f"{e_dt.strftime('%Y年%m月%d日 %H:%M')}到{l_dt.strftime('%Y年%m月%d日 %H:%M')}"
                )
            try:
                (
                    _original_text,
                    topic_to_embedding_map,
                    topic_to_similar_existing_topics_map,
                ) = await self.extract_topics_and_find_similar(
                    input_text_for_sample, self.config.memory_compress_rate
                )
            except Exception as e:
                logger.error(f"样本 {i+1} 提取主题和相似项失败: {e}", exc_info=True)
                continue
            if not topic_to_embedding_map:
                logger.info(f"样本 {i+1} 未提取到有效主题。")
                continue
            logger.info(
                f"样本 {i+1}: 提取到 {len(topic_to_embedding_map)} 个主题，将为它们生成层级摘要。"
            )
            newly_added_or_updated_topics_in_this_sample = []
            for (
                topic_concept,
                topic_concept_embedding,
            ) in topic_to_embedding_map.items():
                hierarchical_summaries_text_only: Optional[Dict[str, str]] = None
                try:
                    hierarchical_summaries_text_only = (
                        await self.hippocampus.generate_hierarchical_summaries(
                            input_text_for_sample, time_info_for_summary, topic_concept
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"为主题 '{topic_concept}' (样本 {i+1}) 生成层级摘要文本时出错: {e}",
                        exc_info=True,
                    )
                    continue
                if not hierarchical_summaries_text_only or not any(
                    hierarchical_summaries_text_only.values()
                ):
                    logger.warning(
                        f"主题 '{topic_concept}' (样本 {i+1}) 的层级摘要文本为空或无效。"
                    )
                    continue
                hierarchical_summaries_with_embeddings: Dict[
                    str, Tuple[str, Optional[List[float]]]
                ] = {}
                has_at_least_one_summary_with_embedding = False
                summary_gen_start_ts = time.monotonic()
                summaries_embedded_this_batch = 0
                for (
                    level_name,
                    summary_text,
                ) in hierarchical_summaries_text_only.items():
                    if not summary_text or not summary_text.strip():
                        hierarchical_summaries_with_embeddings[level_name] = (
                            summary_text,
                            None,
                        )
                        continue
                    summary_embedding_vec = await self.hippocampus.get_embedding_async(
                        summary_text, request_type=f"memory_sum_embed_{level_name}"
                    )
                    hierarchical_summaries_with_embeddings[level_name] = (
                        summary_text,
                        summary_embedding_vec,
                    )
                    if summary_embedding_vec:
                        has_at_least_one_summary_with_embedding = True
                    summaries_embedded_this_batch += 1
                    elapsed_batch_time = time.monotonic() - summary_gen_start_ts
                    expected_time_for_rpms = (
                        summaries_embedded_this_batch
                        * self.config.rpm_limit_delay_summary_sec
                    )
                    wait_time_needed = max(
                        0, expected_time_for_rpms - elapsed_batch_time
                    )
                    if wait_time_needed > 0:
                        await asyncio.sleep(wait_time_needed)
                if (
                    not has_at_least_one_summary_with_embedding
                    and not hierarchical_summaries_with_embeddings.get(
                        "L1_core_sentence", (None, None)
                    )[1]
                ):
                    logger.warning(
                        f"主题 '{topic_concept}' (样本 {i+1}) 所有层级摘要均未能成功获取嵌入，跳过此主题。"
                    )
                    continue
                current_event_id = f"event_{i}_{topic_concept.replace(' ','_')}_{str(uuid.uuid4())[:8]}"
                source_details = {
                    "sample_index": i,
                    "valid_message_count": valid_msgs_count,
                }
                self.memory_graph.add_dot(
                    concept=topic_concept,
                    hierarchical_summaries_data=hierarchical_summaries_with_embeddings,
                    concept_embedding=topic_concept_embedding,
                    event_id=current_event_id,
                    source_info=source_details,
                )
                added_nodes_concepts.add(topic_concept)
                newly_added_or_updated_topics_in_this_sample.append(topic_concept)
            if len(newly_added_or_updated_topics_in_this_sample) >= 2:
                for t1, t2 in combinations(
                    newly_added_or_updated_topics_in_this_sample, 2
                ):
                    self.memory_graph.connect_dot(t1, t2)
                    connected_edges_pairs.add("-".join(sorted((t1, t2))))
            for topic_concept in newly_added_or_updated_topics_in_this_sample:
                if topic_concept in topic_to_similar_existing_topics_map:
                    for (
                        similar_existing_topic,
                        _similarity_score,
                    ) in topic_to_similar_existing_topics_map[topic_concept]:
                        if self.memory_graph.G.has_node(similar_existing_topic):
                            self.memory_graph.connect_dot(
                                topic_concept, similar_existing_topic
                            )
                            connected_edges_pairs.add(
                                "-".join(
                                    sorted((topic_concept, similar_existing_topic))
                                )
                            )
            logger.info(
                f"样本 {i+1} 处理耗时: {time.time() - sample_process_start_time:.2f}s"
            )
        if processed_sample_count > 0 and (
            added_nodes_concepts or connected_edges_pairs
        ):
            logger.info(
                f"构建记忆处理了 {processed_sample_count} 个有效样本。"
                f"新增/更新概念数: {len(added_nodes_concepts)}, "
                f"新增/强化连接数: {len(connected_edges_pairs)}。即将同步到数据库..."
            )
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info(
                "本次构建记忆无有效样本处理或无图节点/连接变化，跳过数据库同步。"
            )
        logger.info(f"记忆构建 (层级化) 完成，总耗时: {time.time() - start_time:.2f}s")

    async def operation_forget_topic(self, percentage: Optional[float] = None):
        start_time = time.time()
        eff_percentage = (
            percentage
            if percentage is not None and 0 < percentage <= 1
            else self.config.memory_forget_percentage
        )
        nodes_in_graph = list(self.memory_graph.G.nodes())
        edges_in_graph = list(self.memory_graph.G.edges())
        if not nodes_in_graph and not edges_in_graph:
            logger.info("记忆图为空，无需执行遗忘操作。")
            return
        num_nodes_to_check = (
            max(1, int(len(nodes_in_graph) * eff_percentage)) if nodes_in_graph else 0
        )
        num_edges_to_check = (
            max(1, int(len(edges_in_graph) * eff_percentage)) if edges_in_graph else 0
        )
        logger.info(
            f"遗忘操作: 检查 {num_nodes_to_check}/{len(nodes_in_graph)} 个节点, "
            f"{num_edges_to_check}/{len(edges_in_graph)} 条边。"
        )
        if num_nodes_to_check == 0 and num_edges_to_check == 0:
            logger.info("需检查的节点和边数量均为0，跳过遗忘。")
            return
        nodes_sample_for_forget = (
            random.sample(nodes_in_graph, num_nodes_to_check)
            if num_nodes_to_check > 0 and len(nodes_in_graph) >= num_nodes_to_check
            else (nodes_in_graph if num_nodes_to_check > 0 else [])
        )
        edges_sample_for_forget = (
            random.sample(edges_in_graph, num_edges_to_check)
            if num_edges_to_check > 0 and len(edges_in_graph) >= num_edges_to_check
            else (edges_in_graph if num_edges_to_check > 0 else [])
        )
        edge_changes = {"weakened": [], "removed": []}
        node_changes = {"memory_event_removed_from_node": [], "node_itself_removed": []}
        current_timestamp = datetime.now().timestamp()
        edge_forget_threshold_seconds = self.config.memory_forget_time_hours * 3600
        node_event_forget_threshold_seconds = (
            self.config.node_summary_forget_time_hours * 3600
        )
        for u, v in edges_sample_for_forget:
            if not self.memory_graph.G.has_edge(u, v):
                continue
            edge_data = self.memory_graph.G[u][v]
            if (
                current_timestamp - edge_data.get("last_modified", 0)
                > edge_forget_threshold_seconds
            ):
                current_strength = edge_data.get("strength", 1) - 1
                if current_strength <= 0:
                    self.memory_graph.G.remove_edge(u, v)
                    edge_changes["removed"].append(f"{u}-{v}")
                else:
                    self.memory_graph.G[u][v]["strength"] = current_strength
                    self.memory_graph.G[u][v]["last_modified"] = current_timestamp
                    edge_changes["weakened"].append(
                        f"{u}-{v} (S:{current_strength+1}->{current_strength})"
                    )
        for node_name in nodes_sample_for_forget:
            if node_name not in self.memory_graph.G:
                continue
            node_tuple = self.memory_graph.get_dot(node_name)
            if not node_tuple:
                continue
            _concept, node_data = node_tuple
            if (
                current_timestamp - node_data.get("last_modified", 0)
                > node_event_forget_threshold_seconds
            ):
                removed_event_l1_summary = self.memory_graph.forget_memory_event(
                    node_name
                )
                if removed_event_l1_summary:
                    if node_name in self.memory_graph.G:
                        node_changes["memory_event_removed_from_node"].append(
                            f"{node_name} (L1: {removed_event_l1_summary[:30]}...)"
                        )
                    else:
                        node_changes["node_itself_removed"].append(
                            f"{node_name} (因最后事件被遗忘)"
                        )
                elif (
                    node_name not in self.memory_graph.G
                    and node_name not in node_changes["node_itself_removed"]
                ):
                    node_changes["node_itself_removed"].append(
                        f"{node_name} (可能原无事件或数据问题)"
                    )
        if any(edge_changes.values()) or any(node_changes.values()):
            change_summary_parts = []
            if edge_changes["weakened"]:
                change_summary_parts.append(f"边削弱:{len(edge_changes['weakened'])}")
            if edge_changes["removed"]:
                change_summary_parts.append(f"边移除:{len(edge_changes['removed'])}")
            if node_changes["memory_event_removed_from_node"]:
                change_summary_parts.append(
                    f"节点内事件移除:{len(node_changes['memory_event_removed_from_node'])}"
                )
            if node_changes["node_itself_removed"]:
                change_summary_parts.append(
                    f"节点移除:{len(node_changes['node_itself_removed'])}"
                )
            logger.info(
                f"遗忘操作检测到变化: {'; '.join(change_summary_parts)}. 即将同步到数据库..."
            )
            if not self.hippocampus.entorhinal_cortex:
                logger.error("EntorhinalCortex 未初始化，无法同步遗忘操作。")
            else:
                await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次遗忘操作未发现满足条件的边或节点内记忆事件。")
        logger.info(f"遗忘操作完成，耗时: {time.time() - start_time:.2f}s")

    async def operation_consolidate_memory(self):
        start_time = time.time()
        consolidation_percentage_to_check = self.config.consolidate_memory_percentage
        consolidation_similarity_threshold = (
            self.config.consolidation_similarity_threshold
        )
        logger.info(
            f"开始整合记忆 (层级化)... 检查比例:{consolidation_percentage_to_check:.1%}, "
            f"合并阈值 (基于L1摘要):{consolidation_similarity_threshold:.2f}"
        )
        eligible_nodes_for_consolidation = []
        for node_name in list(self.memory_graph.G.nodes()):
            node_tuple = self.memory_graph.get_dot(node_name)
            if not node_tuple:
                continue
            _concept, node_data = node_tuple
            memory_events = node_data.get("memory_events", [])
            valid_events_for_comparison_count = 0
            for event in memory_events:
                hs = event.get("hierarchical_summaries", {})
                l1_data = hs.get("L1_core_sentence")
                if (
                    l1_data
                    and isinstance(l1_data, tuple)
                    and len(l1_data) == 2
                    and l1_data[0]
                    and l1_data[1]
                ):
                    valid_events_for_comparison_count += 1
            if valid_events_for_comparison_count >= 2:
                eligible_nodes_for_consolidation.append(node_name)
        if not eligible_nodes_for_consolidation:
            logger.info("无包含至少2个带有效L1摘要和嵌入的记忆事件的节点，无需整合。")
            return
        num_nodes_to_actually_check = max(
            1,
            int(
                len(eligible_nodes_for_consolidation)
                * consolidation_percentage_to_check
            ),
        )
        nodes_to_check_sample = (
            random.sample(eligible_nodes_for_consolidation, num_nodes_to_actually_check)
            if len(eligible_nodes_for_consolidation) > num_nodes_to_actually_check
            else eligible_nodes_for_consolidation
        )
        logger.info(
            f"将检查 {len(nodes_to_check_sample)}/{len(eligible_nodes_for_consolidation)} 个合格节点进行记忆事件整合。"
        )
        total_merged_event_pairs_count = 0
        modified_node_names_set = set()
        current_timestamp = datetime.now().timestamp()
        for node_name in nodes_to_check_sample:
            node_tuple = self.memory_graph.get_dot(node_name)
            if not node_tuple:
                continue
            _concept, node_data = node_tuple
            memory_events_in_node = list(node_data.get("memory_events", []))
            events_with_valid_l1_for_comparison = []
            for event_idx, event_data in enumerate(memory_events_in_node):
                hs = event_data.get("hierarchical_summaries", {})
                l1_summary_tuple = hs.get("L1_core_sentence")
                if (
                    l1_summary_tuple
                    and isinstance(l1_summary_tuple, tuple)
                    and len(l1_summary_tuple) == 2
                    and l1_summary_tuple[0]
                    and l1_summary_tuple[1]
                ):
                    events_with_valid_l1_for_comparison.append(
                        (
                            event_idx,
                            l1_summary_tuple[0],
                            l1_summary_tuple[1],
                            event_data,
                        )
                    )
            if len(events_with_valid_l1_for_comparison) < 2:
                continue
            events_with_valid_l1_for_comparison.sort(
                key=lambda x: calculate_information_content(x[1]), reverse=True
            )
            final_events_for_this_node = []
            processed_original_indices = set()
            for i in range(len(events_with_valid_l1_for_comparison)):
                original_idx_i, summary_i_text, embedding_i, event_data_i = (
                    events_with_valid_l1_for_comparison[i]
                )
                if original_idx_i in processed_original_indices:
                    continue
                current_group_representative_event = event_data_i
                for j in range(i + 1, len(events_with_valid_l1_for_comparison)):
                    original_idx_j, summary_j_text, embedding_j, event_data_j = (
                        events_with_valid_l1_for_comparison[j]
                    )
                    if original_idx_j in processed_original_indices:
                        continue
                    if (
                        cosine_similarity(embedding_i, embedding_j)
                        >= consolidation_similarity_threshold
                    ):
                        logger.info(
                            f"整合节点'{node_name}': 事件 (L1: '{summary_j_text[:30]}...') 与 事件 (L1: '{summary_i_text[:30]}...') 相似. "
                            f"保留信息熵较高者 (即 '{summary_i_text[:30]}...')."
                        )
                        processed_original_indices.add(original_idx_j)
                        total_merged_event_pairs_count += 1
                        modified_node_names_set.add(node_name)
                final_events_for_this_node.append(current_group_representative_event)
                processed_original_indices.add(original_idx_i)
            if len(final_events_for_this_node) < len(memory_events_in_node):
                original_event_ids_kept = {
                    ev.get("event_id") for ev in final_events_for_this_node
                }
                truly_final_events = list(final_events_for_this_node)
                for original_event in memory_events_in_node:
                    is_comparable_and_processed = False
                    for idx, _, _, _ in events_with_valid_l1_for_comparison:
                        if memory_events_in_node[idx].get(
                            "event_id"
                        ) == original_event.get("event_id"):
                            if idx in processed_original_indices:
                                is_comparable_and_processed = True
                            break
                    if (
                        not is_comparable_and_processed
                        and original_event.get("event_id")
                        not in original_event_ids_kept
                    ):
                        hs_original = original_event.get("hierarchical_summaries", {})
                        l1_data_original = hs_original.get("L1_core_sentence")
                        if not (
                            l1_data_original
                            and isinstance(l1_data_original, tuple)
                            and len(l1_data_original) == 2
                            and l1_data_original[0]
                            and l1_data_original[1]
                        ):
                            truly_final_events.append(original_event)
                self.memory_graph.G.nodes[node_name][
                    "memory_events"
                ] = truly_final_events
                self.memory_graph.G.nodes[node_name][
                    "last_modified"
                ] = current_timestamp
            elif modified_node_names_set:
                self.memory_graph.G.nodes[node_name][
                    "last_modified"
                ] = current_timestamp
        if total_merged_event_pairs_count > 0:
            logger.info(
                f"记忆整合共合并移除了 {total_merged_event_pairs_count} 个相似的记忆事件，"
                f"分布在 {len(modified_node_names_set)} 个概念节点中。即将同步到数据库..."
            )
            if not self.hippocampus.entorhinal_cortex:
                logger.error("EntorhinalCortex 未初始化，无法同步整合操作。")
            else:
                await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次记忆整合未发现需要合并的记忆事件。")
        logger.info(f"整合记忆 (层级化) 完成，耗时: {time.time() - start_time:.2f}s")
