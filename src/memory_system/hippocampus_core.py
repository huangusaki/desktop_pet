import asyncio
import math
import random
import time
from datetime import datetime, timedelta, timezone
import re
import networkx as nx
import numpy as np
from collections import Counter
from pymongo import UpdateOne, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError
from pymongo.database import Database
from typing import Optional, List, Tuple, Set, Dict, Any
from itertools import combinations
from .memory_config import MemoryConfig
from ..utils.prompt_builder import PromptBuilder
import uuid
import json

try:
    from ..llm.llm_request import LLM_request, GeminiSDKResponse
except ImportError:
    try:
        from src.llm.llm_request import LLM_request, GeminiSDKResponse
    except ImportError:
        print(
            "CRITICAL: Could not import LLM_request for hippocampus_core. Ensure paths are correct."
        )
        LLM_request = None
        GeminiSDKResponse = None
import logging

logger = logging.getLogger("memory_system")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def calculate_information_content(text: str) -> float:
    if not text:
        return 0.0
    char_count = Counter(text)
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    entropy = 0.0
    for count in char_count.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    v1_arr = np.asarray(v1).flatten()
    v2_arr = np.asarray(v2).flatten()
    if v1_arr.shape != v2_arr.shape or v1_arr.size == 0:
        return 0.0
    dot_product = np.dot(v1_arr, v2_arr)
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = dot_product / (norm1 * norm2)
    return max(0.0, min(1.0, float(similarity)))


class MemoryGraph:
    def __init__(self):
        self.G = nx.Graph()

    def connect_dot(self, concept1: str, concept2: str):
        if concept1 == concept2:
            return
        current_time = datetime.now().timestamp()
        if self.G.has_edge(concept1, concept2):
            self.G[concept1][concept2]["strength"] = (
                self.G[concept1][concept2].get("strength", 1) + 1
            )
            self.G[concept1][concept2]["last_modified"] = current_time
        else:
            self.G.add_edge(
                concept1,
                concept2,
                strength=1,
                created_time=current_time,
                last_modified=current_time,
            )
        logger.debug(f"连接/强化概念点: {concept1} <-> {concept2}")

    def add_dot(
        self,
        concept: str,
        hierarchical_summaries_data: Dict[str, Tuple[str, Optional[List[float]]]],
        concept_embedding: Optional[List[float]] = None,
        event_id: Optional[str] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ):
        current_time = datetime.now().timestamp()
        new_event_id = event_id if event_id else str(uuid.uuid4())
        new_memory_event = {
            "event_id": new_event_id,
            "hierarchical_summaries": hierarchical_summaries_data,
            "source_info": source_info if source_info else {},
            "created_time": current_time,
            "last_accessed_time": current_time,
        }
        if concept in self.G:
            if "memory_events" not in self.G.nodes[concept] or not isinstance(
                self.G.nodes[concept]["memory_events"], list
            ):
                self.G.nodes[concept]["memory_events"] = []
            existing_l1_summaries = []
            for mev in self.G.nodes[concept]["memory_events"]:
                hs = mev.get("hierarchical_summaries", {})
                l1_data = hs.get("L1_core_sentence")
                if (
                    l1_data
                    and isinstance(l1_data, tuple)
                    and len(l1_data) > 0
                    and isinstance(l1_data[0], str)
                ):
                    existing_l1_summaries.append(l1_data[0])
            current_l1_summary_tuple = hierarchical_summaries_data.get(
                "L1_core_sentence"
            )
            current_l1_summary_text = ""
            if (
                current_l1_summary_tuple
                and isinstance(current_l1_summary_tuple, tuple)
                and len(current_l1_summary_tuple) > 0
            ):
                current_l1_summary_text = current_l1_summary_tuple[0]
            if (
                current_l1_summary_text
                and current_l1_summary_text not in existing_l1_summaries
            ):
                self.G.nodes[concept]["memory_events"].append(new_memory_event)
                self.G.nodes[concept]["last_modified"] = current_time
                logger.debug(
                    f"为概念 '{concept}' 添加新记忆事件 (ID: {new_event_id}): '{current_l1_summary_text[:50]}...'"
                )
            elif not current_l1_summary_text:
                logger.warning(
                    f"尝试为概念 '{concept}' 添加记忆事件，但L1摘要为空，已跳过。"
                )
            else:
                logger.debug(
                    f"概念 '{concept}' 已存在相似L1摘要的记忆事件: '{current_l1_summary_text[:50]}...'，未重复添加。"
                )
            if "created_time" not in self.G.nodes[concept]:
                self.G.nodes[concept]["created_time"] = current_time
            if (
                concept_embedding is not None
                and self.G.nodes[concept].get("embedding") is None
            ):
                self.G.nodes[concept]["embedding"] = concept_embedding
                self.G.nodes[concept]["last_modified"] = current_time
                logger.debug(f"为概念 '{concept}' 添加/更新了概念嵌入。")
        else:
            node_attrs = {
                "memory_events": [new_memory_event],
                "embedding": concept_embedding,
                "created_time": current_time,
                "last_modified": current_time,
            }
            self.G.add_node(concept, **node_attrs)
            current_l1_summary_tuple = hierarchical_summaries_data.get(
                "L1_core_sentence"
            )
            current_l1_summary_text = ""
            if (
                current_l1_summary_tuple
                and isinstance(current_l1_summary_tuple, tuple)
                and len(current_l1_summary_tuple) > 0
            ):
                current_l1_summary_text = current_l1_summary_tuple[0]
            logger.info(
                f"创建新概念节点: '{concept}' 并添加了首个记忆事件 (L1: '{current_l1_summary_text[:50]}...')。"
            )

    def get_dot(self, concept: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if concept in self.G:
            original_node_data = self.G.nodes[concept]
            memory_events = original_node_data.get("memory_events", [])
            if not isinstance(memory_events, list):
                logger.warning(
                    f"概念 '{concept}' 的 memory_events 格式不正确 ({type(memory_events)})，将返回空列表。"
                )
                memory_events = []
            validated_events = []
            for event_idx, event_item in enumerate(memory_events):
                if not isinstance(event_item, dict):
                    logger.warning(
                        f"概念 '{concept}' 中发现格式不正确的记忆事件 (非字典): {event_item}，已跳过。"
                    )
                    continue
                event_id = event_item.get("event_id", f"missing_id_{event_idx}")
                hierarchical_summaries = event_item.get("hierarchical_summaries", {})
                validated_hierarchical_summaries = {}
                valid_event_structure = True
                if not isinstance(hierarchical_summaries, dict):
                    logger.warning(
                        f"概念 '{concept}', 事件 '{event_id}' 的 hierarchical_summaries 格式不正确 (非字典): {hierarchical_summaries}，已跳过此事件。"
                    )
                    valid_event_structure = False
                else:
                    for level, summary_tuple in hierarchical_summaries.items():
                        if (
                            isinstance(summary_tuple, tuple)
                            and len(summary_tuple) == 2
                            and isinstance(summary_tuple[0], str)
                            and (
                                summary_tuple[1] is None
                                or isinstance(summary_tuple[1], list)
                            )
                        ):
                            validated_hierarchical_summaries[level] = summary_tuple
                        else:
                            logger.warning(
                                f"概念 '{concept}', 事件 '{event_id}', 层级 '{level}' 的摘要格式不正确: {summary_tuple} (类型: {type(summary_tuple)})，已跳过此层级。"
                            )
                if valid_event_structure:
                    validated_event = {
                        "event_id": event_id,
                        "hierarchical_summaries": validated_hierarchical_summaries,
                        "source_info": event_item.get("source_info", {}),
                        "created_time": event_item.get("created_time"),
                        "last_accessed_time": event_item.get("last_accessed_time"),
                    }
                    validated_events.append(validated_event)
            data_to_return = {
                "memory_events": validated_events,
                "embedding": original_node_data.get("embedding"),
                "created_time": original_node_data.get("created_time"),
                "last_modified": original_node_data.get("last_modified"),
            }
            return concept, data_to_return
        return None

    def get_related_summaries(
        self, topic: str, depth: int = 1, summary_level: str = "L1_core_sentence"
    ) -> Tuple[List[str], List[str]]:
        if topic not in self.G:
            return [], []
        first_layer_summaries, second_layer_summaries = [], []
        node_data_tuple = self.get_dot(topic)
        if node_data_tuple:
            _, node_data = node_data_tuple
            for event in node_data.get("memory_events", []):
                hs = event.get("hierarchical_summaries", {})
                summary_tuple = hs.get(summary_level)
                if (
                    summary_tuple
                    and isinstance(summary_tuple, tuple)
                    and len(summary_tuple) > 0
                    and isinstance(summary_tuple[0], str)
                ):
                    first_layer_summaries.append(summary_tuple[0])
        if depth >= 2:
            try:
                for neighbor in self.G.neighbors(topic):
                    neighbor_data_tuple = self.get_dot(neighbor)
                    if neighbor_data_tuple:
                        _, neighbor_data = neighbor_data_tuple
                        for event in neighbor_data.get("memory_events", []):
                            hs = event.get("hierarchical_summaries", {})
                            summary_tuple = hs.get(summary_level)
                            if (
                                summary_tuple
                                and isinstance(summary_tuple, tuple)
                                and len(summary_tuple) > 0
                                and isinstance(summary_tuple[0], str)
                            ):
                                second_layer_summaries.append(summary_tuple[0])
            except nx.NetworkXError as e:
                logger.error(f"获取 '{topic}' 的邻居节点时出错: {e}")
        return first_layer_summaries, second_layer_summaries

    @property
    def dots(self) -> List[Optional[Tuple[str, Dict[str, Any]]]]:
        return [self.get_dot(node) for node in self.G.nodes()]

    def forget_memory_event(self, topic: str) -> Optional[str]:
        if topic not in self.G:
            logger.warning(f"尝试遗忘主题 '{topic}' 时，节点不存在。")
            return None
        node_data_tuple = self.get_dot(topic)
        if not node_data_tuple:
            if topic in self.G:
                self.G.remove_node(topic)
                logger.warning(
                    f"主题 '{topic}' get_dot 返回 None，节点数据可能损坏，已移除节点。"
                )
            return None
        _concept_name, node_data = node_data_tuple
        memory_events = node_data.get("memory_events", [])
        if not memory_events:
            self.G.remove_node(topic)
            logger.info(f"主题 '{topic}' 无可遗忘的记忆事件，节点移除。")
            return None
        removed_event_idx = random.randrange(len(memory_events))
        removed_event = memory_events.pop(removed_event_idx)
        removed_summary_text = "一个记忆事件被移除"
        hs = removed_event.get("hierarchical_summaries", {})
        l1_summary_tuple = hs.get("L1_core_sentence")
        if (
            l1_summary_tuple
            and isinstance(l1_summary_tuple, tuple)
            and len(l1_summary_tuple) > 0
        ):
            removed_summary_text = l1_summary_tuple[0]
        if not memory_events:
            self.G.remove_node(topic)
            logger.info(
                f"主题 '{topic}' 的最后一个记忆事件 (L1: '{removed_summary_text[:50]}...') 已遗忘，节点移除。"
            )
        else:
            self.G.nodes[topic]["memory_events"] = memory_events
            self.G.nodes[topic]["last_modified"] = datetime.now().timestamp()
            logger.info(
                f"主题 '{topic}' 的一个记忆事件 (L1: '{removed_summary_text[:50]}...') 已遗忘。剩余 {len(memory_events)} 个事件。"
            )
        return removed_summary_text


class EntorhinalCortex:
    def __init__(self, hippocampus: "Hippocampus"):
        self.hippocampus = hippocampus
        self.memory_graph = hippocampus.memory_graph
        self.config: MemoryConfig = hippocampus.config
        self.db: Optional[Database] = hippocampus.db_instance

    def _get_chat_history_for_memory_sample(
        self, sample_length: int
    ) -> List[Dict[str, Any]]:
        if self.db is None or not hasattr(
            self.db, self.hippocampus.chat_collection_name
        ):
            logger.error("数据库或聊天集合未配置，无法获取记忆样本。")
            return []
        try:
            candidate_messages = list(
                self.db[self.hippocampus.chat_collection_name]
                .find(
                    {
                        "role_play_character": self.hippocampus.pet_name_for_history_filter
                    }
                )
                .sort("timestamp", DESCENDING)
                .limit(sample_length * 5)
            )
            candidate_messages.reverse()
            if not candidate_messages:
                return []
            max_mem_time = self.config.max_memorized_time_per_msg
            for _ in range(10):
                if len(candidate_messages) < sample_length:
                    valid_short_snippet = all(
                        msg.get("memorized_times", 0) < max_mem_time
                        for msg in candidate_messages
                    )
                    if valid_short_snippet and candidate_messages:
                        logger.info(
                            f"获取到少于请求长度 ({len(candidate_messages)}<{sample_length}) 但有效的记忆样本。"
                        )
                        ids_to_update = [
                            msg["_id"] for msg in candidate_messages if "_id" in msg
                        ]
                        if ids_to_update:
                            self.db[self.hippocampus.chat_collection_name].update_many(
                                {"_id": {"$in": ids_to_update}},
                                {"$inc": {"memorized_times": 1}},
                            )
                        return candidate_messages
                    else:
                        continue
                start_index = random.randint(0, len(candidate_messages) - sample_length)
                snippet = candidate_messages[start_index : start_index + sample_length]
                if all(msg.get("memorized_times", 0) < max_mem_time for msg in snippet):
                    ids_to_update = [msg["_id"] for msg in snippet if "_id" in msg]
                    if ids_to_update:
                        self.db[self.hippocampus.chat_collection_name].update_many(
                            {"_id": {"$in": ids_to_update}},
                            {"$inc": {"memorized_times": 1}},
                        )
                    return snippet
            logger.warning("多次尝试后未能找到合适的记忆样本片段。")
            return []
        except Exception as e:
            logger.error(f"获取聊天样本时出错: {e}", exc_info=True)
            return []

    def get_memory_sample(self) -> List[List[Dict[str, Any]]]:
        chat_samples = []
        for _ in range(self.config.build_memory_sample_num):
            messages = self._get_chat_history_for_memory_sample(
                self.config.build_memory_sample_length
            )
            if messages:
                chat_samples.append(messages)
            else:
                logger.info("本次未能获取到一个有效的记忆样本片段。")
        logger.info(f"共获取到 {len(chat_samples)} 个记忆样本片段。")
        return chat_samples

    async def sync_memory_to_db(self):
        start_time = time.time()
        if self.db is None:
            logger.error("数据库未初始化，无法同步记忆到数据库。")
            return
        mem_graph_instance = self.memory_graph
        memory_nodes_data_from_graph: Dict[str, Dict[str, Any]] = {}
        for node_concept in list(mem_graph_instance.G.nodes()):
            node_tuple = mem_graph_instance.get_dot(node_concept)
            if node_tuple:
                memory_nodes_data_from_graph[node_tuple[0]] = node_tuple[1]
        memory_edges_data = {}
        for u, v, data in mem_graph_instance.G.edges(data=True):
            memory_edges_data[tuple(sorted((u, v)))] = {
                "source": u,
                "target": v,
                "data": data,
            }
        db_nodes_map, db_edges_map = {}, {}
        try:
            if memory_nodes_data_from_graph:
                for doc in self.db.graph_data_nodes.find(
                    {"concept": {"$in": list(memory_nodes_data_from_graph.keys())}}
                ):
                    db_nodes_map[doc["concept"]] = doc
            if memory_edges_data:
                edge_queries = [
                    {"source": k[0], "target": k[1]} for k in memory_edges_data.keys()
                ]
                if edge_queries:
                    for doc in self.db.graph_data_edges.find({"$or": edge_queries}):
                        db_edges_map[tuple(sorted((doc["source"], doc["target"])))] = (
                            doc
                        )
        except Exception as e:
            logger.error(f"查询数据库图数据时出错: {e}", exc_info=True)
        nodes_to_insert, nodes_to_update, edges_to_insert, edges_to_update = (
            [],
            [],
            [],
            [],
        )
        current_time_ts = datetime.now().timestamp()
        for concept, data_from_graph_node in memory_nodes_data_from_graph.items():
            current_memory_events = data_from_graph_node.get("memory_events", [])
            node_hash = self.hippocampus.calculate_node_hash(
                concept, current_memory_events
            )
            payload = {
                "concept": concept,
                "memory_events": current_memory_events,
                "embedding": data_from_graph_node.get("embedding"),
                "hash": node_hash,
                "created_time": data_from_graph_node.get(
                    "created_time", current_time_ts
                ),
                "last_modified": data_from_graph_node.get(
                    "last_modified", current_time_ts
                ),
            }
            db_node = db_nodes_map.get(concept)
            if not db_node:
                nodes_to_insert.append(payload)
            elif (
                db_node.get("hash") != node_hash
                or db_node.get("embedding") != payload["embedding"]
                or db_node.get("memory_events") != payload["memory_events"]
            ):
                nodes_to_update.append(
                    UpdateOne({"concept": concept}, {"$set": payload})
                )
        for key, edge_info in memory_edges_data.items():
            s, t, data = edge_info["source"], edge_info["target"], edge_info["data"]
            edge_hash = self.hippocampus.calculate_edge_hash(s, t)
            payload = {
                "source": key[0],
                "target": key[1],
                "strength": data.get("strength", 1),
                "hash": edge_hash,
                "created_time": data.get("created_time", current_time_ts),
                "last_modified": data.get("last_modified", current_time_ts),
            }
            db_edge = db_edges_map.get(key)
            if not db_edge:
                edges_to_insert.append(payload)
            elif (
                db_edge.get("strength") != payload["strength"]
                or db_edge.get("last_modified") != payload["last_modified"]
            ):
                edges_to_update.append(
                    UpdateOne({"source": key[0], "target": key[1]}, {"$set": payload})
                )
        nodes_to_delete_names = list(
            set(db_nodes_map.keys()) - set(memory_nodes_data_from_graph.keys())
        )
        edges_to_delete_db_keys = list(
            set(db_edges_map.keys()) - set(memory_edges_data.keys())
        )
        edges_to_delete_filters = (
            [{"source": k[0], "target": k[1]} for k in edges_to_delete_db_keys]
            if edges_to_delete_db_keys
            else []
        )
        ops_summary = {
            "nodes_inserted": 0,
            "nodes_updated": 0,
            "nodes_deleted": 0,
            "edges_inserted": 0,
            "edges_updated": 0,
            "edges_deleted": 0,
        }
        if (
            nodes_to_insert
            or nodes_to_update
            or nodes_to_delete_names
            or edges_to_insert
            or edges_to_update
            or edges_to_delete_filters
        ):
            logger.info(
                f"同步图: N(I:{len(nodes_to_insert)} U:{len(nodes_to_update)} D:{len(nodes_to_delete_names)}) E(I:{len(edges_to_insert)} U:{len(edges_to_update)} D:{len(edges_to_delete_filters)})"
            )
            async with self.hippocampus._db_graph_lock:
                try:
                    if nodes_to_insert:
                        ops_summary["nodes_inserted"] = len(
                            self.db.graph_data_nodes.insert_many(
                                nodes_to_insert, ordered=False
                            ).inserted_ids
                        )
                    if nodes_to_update:
                        ops_summary["nodes_updated"] = (
                            self.db.graph_data_nodes.bulk_write(
                                nodes_to_update, ordered=False
                            ).modified_count
                            or 0
                        )
                    if nodes_to_delete_names:
                        ops_summary["nodes_deleted"] = (
                            self.db.graph_data_nodes.delete_many(
                                {"concept": {"$in": nodes_to_delete_names}}
                            ).deleted_count
                        )
                    if edges_to_insert:
                        ops_summary["edges_inserted"] = len(
                            self.db.graph_data_edges.insert_many(
                                edges_to_insert, ordered=False
                            ).inserted_ids
                        )
                    if edges_to_update:
                        ops_summary["edges_updated"] = (
                            self.db.graph_data_edges.bulk_write(
                                edges_to_update, ordered=False
                            ).modified_count
                            or 0
                        )
                    if edges_to_delete_filters:
                        ops_summary["edges_deleted"] = (
                            self.db.graph_data_edges.delete_many(
                                {"$or": edges_to_delete_filters}
                            ).deleted_count
                        )
                    logger.info(f"数据库同步完成: {ops_summary}")
                except BulkWriteError as bwe:
                    logger.error(f"同步时批量写入错误: {bwe.details}", exc_info=True)
                except Exception as e:
                    logger.error(f"同步时意外错误: {e}", exc_info=True)
        else:
            logger.info("内存图与数据库一致，无需同步。")
        logger.debug(f"sync_memory_to_db 耗时: {time.time() - start_time:.3f}s")

    def sync_memory_from_db(self):
        start_time = time.time()
        if self.db is None:
            logger.error("数据库未初始化，无法加载记忆。")
            return
        self.memory_graph.G.clear()
        current_time_ts = datetime.now().timestamp()
        nodes_needing_concept_emb: List[str] = []
        items_needing_hier_summary_emb: Dict[str, List[Tuple[str, str]]] = {}
        node_db_updates, edge_db_updates = [], []
        try:
            db_nodes = list(self.db.graph_data_nodes.find())
        except Exception as e:
            logger.error(f"从数据库读取节点时出错: {e}", exc_info=True)
            return
        for node_doc in db_nodes:
            concept = node_doc["concept"]
            db_memory_events_raw = node_doc.get("memory_events", [])
            concept_emb = node_doc.get("embedding")
            parsed_graph_memory_events = []
            db_event_struct_changed_for_node = False
            if not isinstance(db_memory_events_raw, list):
                logger.warning(
                    f"节点 '{concept}' 的 memory_events 在DB中格式非列表 ({type(db_memory_events_raw)})，尝试修正。"
                )
                db_memory_events_raw = []
                db_event_struct_changed_for_node = True
            for event_idx, event_doc_item in enumerate(db_memory_events_raw):
                if not isinstance(event_doc_item, dict):
                    logger.warning(
                        f"节点 '{concept}', 事件索引 {event_idx} 格式非字典 ({type(event_doc_item)})，已跳过。"
                    )
                    db_event_struct_changed_for_node = True
                    continue
                event_id = event_doc_item.get("event_id")
                if not event_id:
                    event_id = str(uuid.uuid4())
                    event_doc_item["event_id"] = event_id
                    logger.info(
                        f"节点 '{concept}', 事件索引 {event_idx} 缺少event_id, 已生成: {event_id}"
                    )
                    db_event_struct_changed_for_node = True
                raw_hier_summaries = event_doc_item.get("hierarchical_summaries", {})
                parsed_hier_summaries_for_graph = {}
                db_hier_summary_struct_changed_for_event = False
                if not isinstance(raw_hier_summaries, dict):
                    logger.warning(
                        f"节点 '{concept}', 事件ID '{event_id}' 的 hierarchical_summaries 非字典 ({type(raw_hier_summaries)})，已跳过此事件的摘要。"
                    )
                    db_hier_summary_struct_changed_for_event = True
                    raw_hier_summaries = {}
                for level_name, summary_data_from_db in raw_hier_summaries.items():
                    s_text, s_emb = None, None
                    item_ok_for_graph = False
                    if (
                        isinstance(summary_data_from_db, (tuple, list))
                        and len(summary_data_from_db) == 2
                    ):
                        s_text, s_emb = summary_data_from_db
                        if not isinstance(s_text, str):
                            logger.warning(
                                f"节点'{concept}', 事件'{event_id}', 层级'{level_name}'摘要文本非字符串。"
                            )
                            db_hier_summary_struct_changed_for_event = True
                            continue
                        if s_emb is not None and not isinstance(s_emb, list):
                            logger.warning(
                                f"节点'{concept}', 事件'{event_id}', 层级'{level_name}'嵌入无效，已置为None。"
                            )
                            s_emb = None
                            db_hier_summary_struct_changed_for_event = True
                        item_ok_for_graph = True
                        if isinstance(summary_data_from_db, list):
                            db_hier_summary_struct_changed_for_event = True
                    elif isinstance(summary_data_from_db, str):
                        s_text, s_emb, item_ok_for_graph = (
                            summary_data_from_db,
                            None,
                            True,
                        )
                        db_hier_summary_struct_changed_for_event = True
                        logger.info(
                            f"节点'{concept}', 事件'{event_id}', 层级'{level_name}'只有文本，将补嵌入。"
                        )
                    else:
                        logger.warning(
                            f"节点'{concept}', 事件'{event_id}', 层级'{level_name}'格式无法识别: {type(summary_data_from_db)}。"
                        )
                        db_hier_summary_struct_changed_for_event = True
                        continue
                    if item_ok_for_graph:
                        parsed_hier_summaries_for_graph[level_name] = (s_text, s_emb)
                        if s_emb is None and s_text:
                            if concept not in items_needing_hier_summary_emb:
                                items_needing_hier_summary_emb[concept] = []
                            items_needing_hier_summary_emb[concept].append(
                                (event_id, level_name)
                            )
                if db_hier_summary_struct_changed_for_event:
                    event_doc_item["hierarchical_summaries"] = (
                        parsed_hier_summaries_for_graph
                    )
                    db_event_struct_changed_for_node = True
                parsed_graph_memory_events.append(
                    {
                        "event_id": event_id,
                        "hierarchical_summaries": parsed_hier_summaries_for_graph,
                        "source_info": event_doc_item.get("source_info", {}),
                        "created_time": event_doc_item.get(
                            "created_time", current_time_ts
                        ),
                        "last_accessed_time": event_doc_item.get(
                            "last_accessed_time", current_time_ts
                        ),
                    }
                )
            created_time = node_doc.get("created_time")
            last_modified_time = node_doc.get("last_modified")
            time_payload_for_db_update = {}
            if created_time is None:
                created_time = current_time_ts
                time_payload_for_db_update["created_time"] = current_time_ts
            if last_modified_time is None:
                last_modified_time = current_time_ts
                time_payload_for_db_update["last_modified"] = current_time_ts
            db_updates_for_current_node = {}
            if time_payload_for_db_update:
                db_updates_for_current_node.update(time_payload_for_db_update)
            if db_event_struct_changed_for_node:
                db_updates_for_current_node["memory_events"] = (
                    parsed_graph_memory_events
                )
            if db_updates_for_current_node:
                node_db_updates.append(
                    UpdateOne(
                        {"_id": node_doc["_id"]}, {"$set": db_updates_for_current_node}
                    )
                )
            if concept_emb is None:
                nodes_needing_concept_emb.append(concept)
            self.memory_graph.G.add_node(
                concept,
                memory_events=parsed_graph_memory_events,
                embedding=concept_emb,
                created_time=created_time,
                last_modified=last_modified_time,
            )
        try:
            db_edges = list(self.db.graph_data_edges.find())
        except Exception as e:
            logger.error(f"从数据库读取边时出错: {e}", exc_info=True)
            db_edges = []
        for edge_doc in db_edges:
            s, t = edge_doc["source"], edge_doc["target"]
            if s not in self.memory_graph.G or t not in self.memory_graph.G:
                logger.debug(f"跳过边 ({s}-{t})，因为一个或两个节点在图中不存在。")
                continue
            strength = edge_doc.get("strength", 1)
            created_time_edge = edge_doc.get("created_time")
            last_modified_time_edge = edge_doc.get("last_modified")
            time_payload_edge_update = {}
            if created_time_edge is None:
                created_time_edge = current_time_ts
                time_payload_edge_update["created_time"] = current_time_ts
            if last_modified_time_edge is None:
                last_modified_time_edge = current_time_ts
                time_payload_edge_update["last_modified"] = current_time_ts
            if time_payload_edge_update:
                edge_db_updates.append(
                    UpdateOne(
                        {"_id": edge_doc["_id"]}, {"$set": time_payload_edge_update}
                    )
                )
            self.memory_graph.G.add_edge(
                s,
                t,
                strength=strength,
                created_time=created_time_edge,
                last_modified=last_modified_time_edge,
            )
        if node_db_updates:
            try:
                self.db.graph_data_nodes.bulk_write(node_db_updates, ordered=False)
                logger.info(
                    f"从DB加载时，修正并更新了 {len(node_db_updates)} 个数据库中的节点结构。"
                )
            except Exception as e:
                logger.error(f"从DB加载时，修正数据库节点结构失败: {e}", exc_info=True)
        if edge_db_updates:
            try:
                self.db.graph_data_edges.bulk_write(edge_db_updates, ordered=False)
                logger.info(
                    f"从DB加载时，修正并更新了 {len(edge_db_updates)} 条数据库中的边结构。"
                )
            except Exception as e:
                logger.error(f"从DB加载时，修正数据库边结构失败: {e}", exc_info=True)
        if nodes_needing_concept_emb:
            self.hippocampus._nodes_needing_embedding_update = list(
                set(nodes_needing_concept_emb)
            )
        if items_needing_hier_summary_emb:
            self.hippocampus._items_needing_summary_embedding_update = dict(
                items_needing_hier_summary_emb
            )
        logger.info(
            f"从DB同步记忆完成。加载 {len(self.memory_graph.G.nodes())} 节点, {len(self.memory_graph.G.edges())} 边。耗时: {time.time()-start_time:.3f}s"
        )
        total_summary_levels_to_update = sum(
            len(v)
            for v in self.hippocampus._items_needing_summary_embedding_update.values()
        )
        logger.info(
            f"需补概念嵌入: {len(self.hippocampus._nodes_needing_embedding_update)} 个节点。需补层级摘要嵌入: {total_summary_levels_to_update} 个层级。"
        )


class ParahippocampalGyrus:
    def __init__(self, hippocampus: "Hippocampus"):
        self.hippocampus = hippocampus
        self.memory_graph = hippocampus.memory_graph
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
            if num_nodes_to_check > 0
            else []
        )
        edges_sample_for_forget = (
            random.sample(edges_in_graph, num_edges_to_check)
            if num_edges_to_check > 0
            else []
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
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次记忆整合未发现需要合并的记忆事件。")
        logger.info(f"整合记忆 (层级化) 完成，耗时: {time.time() - start_time:.2f}s")


class Hippocampus:
    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.config: Optional[MemoryConfig] = None
        self.db_instance: Optional[Database] = None
        self.chat_collection_name: Optional[str] = None
        self.pet_name_for_history_filter: Optional[str] = None
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
        for llm_attr, name_in_cfg in [
            (self.llm_topic_judge, "llm_topic_judge"),
            (self.llm_summary_by_topic, "llm_summary_by_topic"),
            (self.llm_embedding_topic, "llm_embedding_topic"),
            (self.llm_re_rank, "llm_re_rank"),
        ]:
            if llm_attr:
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
        self.entorhinal_cortex.sync_memory_from_db()
        if self.llm_embedding_topic and (
            self._nodes_needing_embedding_update
            or self._items_needing_summary_embedding_update
        ):
            logger.info("检测到启动时有缺失的嵌入，将启动后台任务补充...")
            asyncio.create_task(self._update_missing_embeddings_task())
        else:
            logger.info("无需补充嵌入或嵌入模型未配置。")
        logger.info("海马体初始化完成。")

    async def _update_missing_embeddings_task(self):
        if not self.llm_embedding_topic:
            logger.warning("嵌入模型未初始化，无法更新缺失嵌入。")
            self._clear_pending_embeddings()
            return
        delay = self.config.embedding_update_delay_sec
        batch_size = self.config.embedding_update_batch_size
        nodes_to_proc = list(self._nodes_needing_embedding_update)
        self._nodes_needing_embedding_update.clear()
        if nodes_to_proc:
            logger.info(f"开始处理{len(nodes_to_proc)}个节点概念嵌入...")
            upd_c, fail_c = 0, 0
            bulk_db_upd_concept = []
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
                        self.db_instance.graph_data_nodes.bulk_write(
                            bulk_db_upd_concept, ordered=False
                        )
                        bulk_db_upd_concept = []
                    except Exception as e:
                        logger.error(f"批量更新概念嵌入DB出错:{e}")
                await asyncio.sleep(delay)
            if bulk_db_upd_concept and self.db_instance is not None:
                try:
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
                    continue
                _, node_data = node_tuple
                current_memory_events = list(node_data.get("memory_events", []))
                if not current_memory_events:
                    continue
                node_modified_in_db = False
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
                                node_modified_in_db = True
                            else:
                                fail_s += 1
                            await asyncio.sleep(delay)
                if node_modified_in_db and self.db_instance is not None:
                    self.memory_graph.G.nodes[concept][
                        "memory_events"
                    ] = current_memory_events
                    self.memory_graph.G.nodes[concept][
                        "last_modified"
                    ] = datetime.now().timestamp()
                    try:
                        self.db_instance.graph_data_nodes.update_one(
                            {"concept": concept},
                            {
                                "$set": {
                                    "memory_events": current_memory_events,
                                    "last_modified": self.memory_graph.G.nodes[concept][
                                        "last_modified"
                                    ],
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

    def _create_find_topic_prompt(self, txt: str, num: int) -> str:
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
            topics = set(
                t.strip()
                for item in extracted
                for t in re.split(r"[,，、]", item)
                if t.strip()
            )
            logger.debug(f"LLM提取清洗后主题: {list(topics)}")
            return list(topics)
        except Exception as e:
            logger.error(f"提取主题异常:{e}", exc_info=True)
            return []

    def calculate_topic_num(self, txt: str, rate: float) -> int:
        if not txt:
            return 1
        info, lines = calculate_information_content(txt), txt.count("\n") + 1
        len_topic = max(1, math.ceil(lines * rate * 0.3))
        info_topic = max(1, min(5, math.ceil((info - 2.5) * 1.5)))
        num = max(
            1, min(self.config.max_topics_per_snippet, max(len_topic, info_topic))
        )
        return num

    async def get_memory_from_keyword(
        self, kw: str, depth: int = 2, summary_level: str = "L1_core_sentence"
    ) -> List[Tuple[str, List[str], float]]:
        if not kw or not kw.strip() or not self.llm_embedding_topic:
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
                            _, node_data = node_tuple
                            summaries_from_events = []
                            for event in node_data.get("memory_events", []):
                                hs = event.get("hierarchical_summaries", {})
                                summary_data = hs.get(summary_level)
                                if (
                                    summary_data
                                    and isinstance(summary_data, tuple)
                                    and len(summary_data) > 0
                                    and isinstance(summary_data, str)
                                ):
                                    summaries_from_events.append(summary_data)
                            if summaries_from_events:
                                similar_nodes.append(
                                    (node_name, summaries_from_events, sim_to_concept)
                                )
                except Exception as e:
                    logger.error(f"计算关键词'{kw}'与节点'{node_name}'相似度出错:{e}")
        similar_nodes.sort(key=lambda x: x, reverse=True)
        return similar_nodes

    def _create_bulk_relevance_check_prompt(
        self, txt: str, candidates: List[Tuple[str, str, float]]
    ) -> str:
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
        if not txt or not txt.strip():
            return []
        start_t = time.time()
        max_mem, act_depth = (
            num if num is not None else self.config.retrieval_max_final_memories
        ), (depth if depth is not None else self.config.retrieval_activation_depth)
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
            except Exception:
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
            while q:
                curr_n, curr_act, curr_d = q.pop(0)
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
                except Exception:
                    pass
        if not act_map:
            logger.info("激活扩散未能激活任何节点。")
            return []
        sorted_act_nodes = sorted(act_map.items(), key=lambda x: x, reverse=True)
        if not self.llm_embedding_topic:
            logger.warning("嵌入模型不可用，无法相似度排序。")
            return []
        txt_emb = await self.get_embedding_async(
            txt, request_type="memory_input_embed_retrieve"
        )
        if not txt_emb:
            logger.warning("无法获取输入文本嵌入。")
            return []
        candidates_for_rerank: List[Tuple[str, str, float, str, str]] = []
        top_n_scan = self.config.retrieval_top_activated_nodes_to_scan
        min_sum_sim = self.config.retrieval_min_summary_similarity
        for node_name, _act_score in sorted_act_nodes[:top_n_scan]:
            node_tuple = self.memory_graph.get_dot(node_name)
            if not node_tuple:
                continue
            _concept, node_data = node_tuple
            for event in node_data.get("memory_events", []):
                hs = event.get("hierarchical_summaries", {})
                event_id = event.get("event_id", "unknown_event")
                retrieval_summary_data = hs.get(retrieval_summary_level)
                output_summary_data = hs.get(output_summary_level)
                if (
                    not retrieval_summary_data
                    or not isinstance(retrieval_summary_data, tuple)
                    or len(retrieval_summary_data) < 2
                ):
                    continue
                retrieval_s_text, retrieval_s_emb = retrieval_summary_data
                output_s_text_str = ""
                if (
                    output_summary_data
                    and isinstance(output_summary_data, tuple)
                    and len(output_summary_data) > 0
                    and isinstance(output_summary_data[0], str)
                ):
                    output_s_text_str = output_summary_data[0]
                elif (
                    retrieval_summary_data
                    and isinstance(retrieval_summary_data, tuple)
                    and len(retrieval_summary_data) > 0
                    and isinstance(retrieval_summary_data[0], str)
                ):
                    output_s_text_str = retrieval_summary_data[0]
                else:
                    logger.warning(
                        f"在节点 {node_name} 事件 {event_id} 中，无法获取有效的 output 或 retrieval 摘要文本。"
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
                    except Exception:
                        pass
        if not candidates_for_rerank:
            logger.info(f"无摘要 (层级 {retrieval_summary_level}) 通过相似度阈值。")
            return []
        candidates_for_rerank.sort(key=lambda x: x, reverse=True)
        to_llm_rerank_input = [
            (topic, event_id, retrieval_text, score)
            for topic, event_id, retrieval_text, score, _output_text in candidates_for_rerank[
                : self.config.retrieval_max_candidates_for_llm_rerank
            ]
        ]
        candidates_map_for_output = {
            (cand_topic, cand_event_id): output_text
            for cand_topic, cand_event_id, _r_text, _score, output_text in candidates_for_rerank[
                : self.config.retrieval_max_candidates_for_llm_rerank
            ]
        }
        final_mem_tuples: List[Tuple[str, str]] = []
        if self.llm_re_rank and to_llm_rerank_input:
            logger.info(
                f"LLM({self.llm_re_rank.model_name})重排{len(to_llm_rerank_input)}条候选记忆 (基于 {retrieval_summary_level} 摘要)..."
            )
            prompt_candidates = [
                (topic, text, score) for topic, _eid, text, score in to_llm_rerank_input
            ]
            prompt = self._create_bulk_relevance_check_prompt(txt, prompt_candidates)
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
                            if 0 <= idx < len(to_llm_rerank_input):
                                (
                                    selected_topic,
                                    selected_event_id,
                                    _selected_retrieval_text,
                                    _selected_score,
                                ) = to_llm_rerank_input[idx]
                                output_text_for_final = candidates_map_for_output.get(
                                    (selected_topic, selected_event_id)
                                )
                                if output_text_for_final:
                                    final_mem_tuples.append(
                                        (selected_topic, output_text_for_final)
                                    )
                                else:
                                    final_mem_tuples.append(
                                        (selected_topic, _selected_retrieval_text)
                                    )
                        logger.info(
                            f"LLM成功重排并选择了{len(final_mem_tuples)}条记忆。"
                        )
                    except Exception as e:
                        logger.error(f"解析LLM重排索引失败:{e}. 回退到原始相似度排序。")
                        final_mem_tuples = [
                            (
                                topic,
                                candidates_map_for_output.get(
                                    (topic, event_id), r_text
                                ),
                            )
                            for topic, event_id, r_text, _score, _o_text in candidates_for_rerank[
                                : self.config.retrieval_max_candidates_for_llm_rerank
                            ]
                        ]
                else:
                    logger.warning("LLM重排响应为空, 回退到原始相似度排序。")
                    final_mem_tuples = []
                    for (
                        topic,
                        event_id,
                        r_text,
                        _score,
                        o_text_str,
                    ) in candidates_for_rerank[
                        : self.config.retrieval_max_candidates_for_llm_rerank
                    ]:
                        final_mem_tuples.append(
                            (
                                topic,
                                candidates_map_for_output.get(
                                    (topic, event_id),
                                    o_text_str if o_text_str else r_text,
                                ),
                            )
                        )
            except Exception as e:
                logger.error(f"LLM重排调用失败:{e}", exc_info=True)
                final_mem_tuples = []
                for (
                    topic,
                    event_id,
                    r_text,
                    _score,
                    o_text_str,
                ) in candidates_for_rerank[
                    : self.config.retrieval_max_candidates_for_llm_rerank
                ]:
                    final_mem_tuples.append(
                        (
                            topic,
                            candidates_map_for_output.get(
                                (topic, event_id), o_text_str if o_text_str else r_text
                            ),
                        )
                    )
        else:
            if not self.llm_re_rank:
                logger.debug("重排LLM未配置，使用初步筛选结果。")
            final_mem_tuples = []
            for topic, event_id, r_text, _score, o_text_str in candidates_for_rerank[
                : self.config.retrieval_max_candidates_for_llm_rerank
            ]:
                final_mem_tuples.append(
                    (
                        topic,
                        candidates_map_for_output.get(
                            (topic, event_id), o_text_str if o_text_str else r_text
                        ),
                    )
                )
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
        if not txt or not txt.strip():
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
            except Exception:
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
            while q:
                curr_n, curr_act, curr_d = q.pop(0)
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
                except Exception:
                    pass
        if not act_map:
            return 0.0
        total_act, num_active, total_nodes = (
            sum(act_map.values()),
            len(act_map),
            len(self.memory_graph.G.nodes()),
        )
        if total_nodes == 0:
            return 0.0
        prop_active = num_active / total_nodes
        avg_act_active = total_act / num_active if num_active > 0 else 0.0
        score = avg_act_active * prop_active
        return max(0.0, min(1.0, score))


class HippocampusManager:
    _instance: Optional["HippocampusManager"] = None
    _hippocampus: Optional[Hippocampus] = None
    _initialized: bool = False
    _async_lock = asyncio.Lock()

    def __init__(self):
        if HippocampusManager._instance is not None:
            raise RuntimeError("HippocampusManager is a singleton, use get_instance().")

    @classmethod
    async def get_instance(cls) -> "HippocampusManager":
        if cls._instance is None:
            async with cls._async_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_hippocampus(cls) -> Hippocampus:
        if not cls._initialized or cls._hippocampus is None:
            raise RuntimeError(
                "HippocampusManager 尚未初始化，请先调用 initialize_singleton"
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
    ):
        if self._initialized:
            logger.warning("HippocampusManager 已初始化，跳过。")
            return self._hippocampus
        try:
            self._hippocampus = Hippocampus()
            self._hippocampus.initialize(
                memory_config=memory_config,
                database_instance=database_instance,
                chat_history_collection_name=chat_collection_name,
                pet_name=pet_name,
                prompt_builder=prompt_builder,
                global_llm_params=global_llm_params,
            )
            self._initialized = True
            logger.info("海马体管理器 (HippocampusManager) 初始化成功。")
            h_cfg = self._hippocampus.config
            if h_cfg:
                nodes, edges = len(self._hippocampus.memory_graph.G.nodes()), len(
                    self._hippocampus.memory_graph.G.edges()
                )
                logger.info(
                    f"--- 记忆系统配置: 构建(样本数:{h_cfg.build_memory_sample_num},长度:{h_cfg.build_memory_sample_length}), 遗忘(比例:{h_cfg.memory_forget_percentage}), 整合(比例:{h_cfg.consolidate_memory_percentage}), 嵌入模型({h_cfg.llm_embedding_topic.get('name', 'N/A')}), 当前图({nodes}节点,{edges}边) ---"
                )
        except Exception as e:
            logger.error(f"HippocampusManager 初始化失败: {e}", exc_info=True)
            self._initialized = False
            self._hippocampus = None
            raise
        return self._hippocampus

    async def build_memory(self):
        if (
            not self._initialized
            or not self._hippocampus
            or not self._hippocampus.parahippocampal_gyrus
        ):
            raise RuntimeError("Manager/组件未初始化")
        try:
            await self._hippocampus.parahippocampal_gyrus.operation_build_memory()
        except Exception as e:
            logger.error(f"执行build_memory出错:{e}", exc_info=True)

    async def forget_memory(self, percentage: Optional[float] = None):
        if (
            not self._initialized
            or not self._hippocampus
            or not self._hippocampus.parahippocampal_gyrus
        ):
            raise RuntimeError("Manager/组件未初始化")
        eff_perc = (
            percentage
            if percentage is not None
            else (
                self._hippocampus.config.memory_forget_percentage
                if self._hippocampus.config
                else 0.005
            )
        )
        try:
            await self._hippocampus.parahippocampal_gyrus.operation_forget_topic(
                eff_perc
            )
        except Exception as e:
            logger.error(f"执行forget_memory(perc={eff_perc})出错:{e}", exc_info=True)

    async def consolidate_memory(self):
        if (
            not self._initialized
            or not self._hippocampus
            or not self._hippocampus.parahippocampal_gyrus
        ):
            raise RuntimeError("Manager/组件未初始化")
        try:
            await self._hippocampus.parahippocampal_gyrus.operation_consolidate_memory()
        except Exception as e:
            logger.error(f"执行consolidate_memory出错:{e}", exc_info=True)

    async def get_memory_from_text(
        self,
        txt: str,
        num: Optional[int] = None,
        depth: Optional[int] = None,
        fast_kw: bool = False,
    ) -> List[Tuple[str, str]]:
        if not self._initialized or not self._hippocampus:
            raise RuntimeError("Manager未初始化")
        try:
            return await self._hippocampus.get_memory_from_text(
                txt,
                num=num,
                depth=depth,
                fast_kw=fast_kw,
            )
        except Exception as e:
            logger.error(f"执行get_memory_from_text出错:{e}", exc_info=True)
            return []

    async def get_activation_score_from_text(
        self, txt: str, depth: int = 3, fast_kw: bool = False
    ) -> float:
        if not self._initialized or not self._hippocampus:
            raise RuntimeError("Manager未初始化")
        try:
            return await self._hippocampus.get_activation_score_from_text(
                txt, max_depth=depth, fast_retrieval_keywords=fast_kw
            )
        except Exception as e:
            logger.error(f"执行get_activation_score_from_text出错:{e}", exc_info=True)
            return 0.0

    async def get_memory_from_keyword(
        self, kw: str, depth: int = 2
    ) -> List[Tuple[str, List[str], float]]:
        if not self._initialized or not self._hippocampus:
            raise RuntimeError("Manager未初始化")
        try:
            return await self._hippocampus.get_memory_from_keyword(kw, max_depth=depth)
        except Exception as e:
            logger.error(f"执行get_memory_from_keyword出错:{e}", exc_info=True)
            return []

    def get_all_node_names(self) -> List[str]:
        if not self._initialized or not self._hippocampus:
            raise RuntimeError("Manager未初始化")
        try:
            return self._hippocampus.get_all_node_names()
        except Exception as e:
            logger.error(f"执行get_all_node_names出错:{e}", exc_info=True)
            return []
