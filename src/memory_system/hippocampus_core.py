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
        memory_summary: str,
        concept_embedding: Optional[List[float]] = None,
        summary_embedding: Optional[List[float]] = None,
    ):
        current_time = datetime.now().timestamp()
        memory_item_tuple = (memory_summary, summary_embedding)
        if concept in self.G:
            if "memory_items" not in self.G.nodes[concept] or not isinstance(
                self.G.nodes[concept]["memory_items"], list
            ):
                self.G.nodes[concept]["memory_items"] = []
            existing_summaries = [
                item[0] for item in self.G.nodes[concept]["memory_items"]
            ]
            if memory_summary not in existing_summaries:
                self.G.nodes[concept]["memory_items"].append(memory_item_tuple)
                self.G.nodes[concept]["last_modified"] = current_time
                logger.debug(
                    f"为概念 '{concept}' 添加新记忆项: '{memory_summary[:50]}...'"
                )
            else:
                logger.debug(
                    f"概念 '{concept}' 已存在记忆项: '{memory_summary[:50]}...'，未重复添加。"
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
                "memory_items": [memory_item_tuple],
                "embedding": concept_embedding,
                "created_time": current_time,
                "last_modified": current_time,
            }
            self.G.add_node(concept, **node_attrs)
            logger.info(f"创建新概念节点: '{concept}' 并添加了首个记忆项。")

    def get_dot(self, concept: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if concept in self.G:
            original_node_data = self.G.nodes[concept]
            mem_items = original_node_data.get("memory_items", [])
            if not isinstance(mem_items, list):
                logger.warning(
                    f"概念 '{concept}' 的 memory_items 格式不正确 ({type(mem_items)})，将返回空列表。"
                )
                mem_items = []
            validated_items = []
            for item in mem_items:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and (item[1] is None or isinstance(item[1], list))
                ):
                    validated_items.append(item)
                else:
                    logger.warning(
                        f"概念 '{concept}' 中发现格式不正确的记忆项: {item} (类型: {type(item)})，已跳过。"
                    )
            data_to_return = {
                "memory_items": validated_items,
                "embedding": original_node_data.get("embedding"),
                "created_time": original_node_data.get("created_time"),
                "last_modified": original_node_data.get("last_modified"),
            }
            return concept, data_to_return
        return None

    def get_related_summaries(
        self, topic: str, depth: int = 1
    ) -> Tuple[List[str], List[str]]:
        if topic not in self.G:
            return [], []
        first_layer_summaries, second_layer_summaries = [], []
        node_data_tuple = self.get_dot(topic)
        if node_data_tuple:
            _, node_data = node_data_tuple
            first_layer_summaries.extend(
                [
                    item[0]
                    for item in node_data.get("memory_items", [])
                    if isinstance(item, tuple) and len(item) > 0
                ]
            )
        if depth >= 2:
            try:
                for neighbor in self.G.neighbors(topic):
                    neighbor_data_tuple = self.get_dot(neighbor)
                    if neighbor_data_tuple:
                        _, neighbor_data = neighbor_data_tuple
                        second_layer_summaries.extend(
                            [
                                item[0]
                                for item in neighbor_data.get("memory_items", [])
                                if isinstance(item, tuple) and len(item) > 0
                            ]
                        )
            except nx.NetworkXError as e:
                logger.error(f"获取 '{topic}' 的邻居节点时出错: {e}")
        return first_layer_summaries, second_layer_summaries

    @property
    def dots(self) -> List[Optional[Tuple[str, Dict[str, Any]]]]:
        return [self.get_dot(node) for node in self.G.nodes()]

    def forget_topic_summary(self, topic: str) -> Optional[str]:
        if topic not in self.G:
            return None
        node_data_tuple = self.get_dot(topic)
        if not node_data_tuple:
            if topic in self.G:
                self.G.remove_node(topic)
            logger.warning(
                f"尝试遗忘主题 '{topic}' 时 get_dot 返回 None 或节点数据损坏。"
            )
            return None
        _, node_data = node_data_tuple
        memory_items_tuples = node_data.get("memory_items", [])
        if memory_items_tuples:
            removed_idx = random.randrange(len(memory_items_tuples))
            removed_summary_text = memory_items_tuples.pop(removed_idx)[0]
            if not memory_items_tuples:
                self.G.remove_node(topic)
                logger.info(
                    f"主题 '{topic}' 最后记忆项 '{removed_summary_text[:50]}...' 已遗忘，节点移除。"
                )
            else:
                self.G.nodes[topic]["memory_items"] = memory_items_tuples
                self.G.nodes[topic]["last_modified"] = datetime.now().timestamp()
                logger.info(
                    f"主题 '{topic}' 记忆项 '{removed_summary_text[:50]}...' 已遗忘。剩余 {len(memory_items_tuples)} 项。"
                )
            return removed_summary_text
        else:
            self.G.remove_node(topic)
            logger.info(f"主题 '{topic}' 无可遗忘记忆项，节点移除。")
            return None


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
        mem_graph = self.memory_graph.G
        memory_nodes_data = {}
        for node_concept in list(mem_graph.nodes()):
            node_tuple = self.memory_graph.get_dot(node_concept)
            if node_tuple:
                memory_nodes_data[node_tuple[0]] = node_tuple[1]
        memory_edges_data = {}
        for u, v, data in mem_graph.edges(data=True):
            memory_edges_data[tuple(sorted((u, v)))] = {
                "source": u,
                "target": v,
                "data": data,
            }
        db_nodes_map, db_edges_map = {}, {}
        try:
            if memory_nodes_data:
                for doc in self.db.graph_data_nodes.find(
                    {"concept": {"$in": list(memory_nodes_data.keys())}}
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
        for concept, data in memory_nodes_data.items():
            memory_items = data.get("memory_items", [])
            summaries = [
                item[0]
                for item in memory_items
                if isinstance(item, tuple) and len(item) > 0
            ]
            node_hash = self.hippocampus.calculate_node_hash(concept, summaries)
            payload = {
                "concept": concept,
                "memory_items": memory_items,
                "embedding": data.get("embedding"),
                "hash": node_hash,
                "created_time": data.get("created_time", current_time_ts),
                "last_modified": data.get("last_modified", current_time_ts),
            }
            db_node = db_nodes_map.get(concept)
            if not db_node:
                nodes_to_insert.append(payload)
            elif (
                db_node.get("hash") != node_hash
                or db_node.get("embedding") != payload["embedding"]
                or db_node.get("memory_items") != payload["memory_items"]
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
            set(db_nodes_map.keys()) - set(memory_nodes_data.keys())
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
        nodes_needing_concept_emb, items_needing_summary_emb = [], {}
        node_db_updates, edge_db_updates = [], []
        try:
            db_nodes = list(self.db.graph_data_nodes.find())
        except Exception as e:
            logger.error(f"读取节点时出错: {e}", exc_info=True)
            return
        for node_doc in db_nodes:
            concept = node_doc["concept"]
            db_mem_items = node_doc.get("memory_items", [])
            concept_emb = node_doc.get("embedding")
            valid_graph_items, item_indices_need_emb, db_item_struct_change = (
                [],
                [],
                False,
            )
            if not isinstance(db_mem_items, list):
                db_mem_items = [db_mem_items] if isinstance(db_mem_items, str) else []
                db_item_struct_change = True
            for idx, item_db in enumerate(db_mem_items):
                s_text, s_emb, item_ok = None, None, False
                if isinstance(item_db, (tuple, list)) and len(item_db) == 2:
                    s_text, s_emb = item_db
                    if not isinstance(s_text, str):
                        logger.warning(f"节点'{concept}'项{idx}摘要非字符串")
                        db_item_struct_change = True
                        continue
                    if s_emb is not None and not isinstance(s_emb, list):
                        s_emb = None
                        db_item_struct_change = True
                        logger.warning(f"节点'{concept}'项{idx}嵌入无效")
                    item_ok = True
                    if isinstance(item_db, list):
                        db_item_struct_change = True
                elif isinstance(item_db, str):
                    s_text, s_emb, item_ok, db_item_struct_change = (
                        item_db,
                        None,
                        True,
                        True,
                    )
                else:
                    logger.warning(f"节点'{concept}'项{idx}格式无法识别")
                    db_item_struct_change = True
                    continue
                if item_ok:
                    valid_graph_items.append((s_text, s_emb))
                    if s_emb is None:
                        item_indices_need_emb.append(len(valid_graph_items) - 1)
            if item_indices_need_emb:
                items_needing_summary_emb[concept] = item_indices_need_emb
            created, modified, time_payload = (
                node_doc.get("created_time"),
                node_doc.get("last_modified"),
                {},
            )
            if created is None:
                created, time_payload["created_time"] = current_time_ts, current_time_ts
            if modified is None:
                modified, time_payload["last_modified"] = (
                    current_time_ts,
                    current_time_ts,
                )
            db_updates = {}
            if time_payload:
                db_updates.update(time_payload)
            if db_item_struct_change:
                db_updates["memory_items"] = valid_graph_items
            if db_updates:
                node_db_updates.append(
                    UpdateOne({"_id": node_doc["_id"]}, {"$set": db_updates})
                )
            if concept_emb is None:
                nodes_needing_concept_emb.append(concept)
            self.memory_graph.G.add_node(
                concept,
                memory_items=valid_graph_items,
                embedding=concept_emb,
                created_time=created,
                last_modified=modified,
            )
        try:
            db_edges = list(self.db.graph_data_edges.find())
        except Exception as e:
            logger.error(f"读取边时出错: {e}", exc_info=True)
            db_edges = []
        for edge_doc in db_edges:
            s, t = edge_doc["source"], edge_doc["target"]
            if s not in self.memory_graph.G or t not in self.memory_graph.G:
                continue
            strength, created, modified, time_payload = (
                edge_doc.get("strength", 1),
                edge_doc.get("created_time"),
                edge_doc.get("last_modified"),
                {},
            )
            if created is None:
                created, time_payload["created_time"] = current_time_ts, current_time_ts
            if modified is None:
                modified, time_payload["last_modified"] = (
                    current_time_ts,
                    current_time_ts,
                )
            if time_payload:
                edge_db_updates.append(
                    UpdateOne({"_id": edge_doc["_id"]}, {"$set": time_payload})
                )
            self.memory_graph.G.add_edge(
                s, t, strength=strength, created_time=created, last_modified=modified
            )
        if node_db_updates:
            try:
                self.db.graph_data_nodes.bulk_write(node_db_updates, ordered=False)
                logger.info(f"修正了{len(node_db_updates)}个DB节点")
            except Exception as e:
                logger.error(f"修正DB节点失败: {e}", exc_info=True)
        if edge_db_updates:
            try:
                self.db.graph_data_edges.bulk_write(edge_db_updates, ordered=False)
                logger.info(f"修正了{len(edge_db_updates)}条DB边")
            except Exception as e:
                logger.error(f"修正DB边失败: {e}", exc_info=True)
        if nodes_needing_concept_emb:
            self.hippocampus._nodes_needing_embedding_update = list(
                set(nodes_needing_concept_emb)
            )
        if items_needing_summary_emb:
            self.hippocampus._items_needing_summary_embedding_update = dict(
                items_needing_summary_emb
            )
        logger.info(
            f"从DB同步记忆完成。加载 {len(self.memory_graph.G.nodes())} 节点, {len(self.memory_graph.G.edges())} 边。耗时: {time.time()-start_time:.3f}s"
        )
        logger.info(
            f"需补概念嵌入: {len(self.hippocampus._nodes_needing_embedding_update)}。需补摘要嵌入: {sum(len(v) for v in self.hippocampus._items_needing_summary_embedding_update.values())}项。"
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
        logger.info("开始构建记忆...")
        memory_samples = self.hippocampus.entorhinal_cortex.get_memory_sample()
        if not memory_samples:
            logger.info("无记忆样本，跳过构建。")
            return
        added_nodes, connected_edges = set(), set()
        processed_count = 0
        for i, messages in enumerate(memory_samples):
            sample_time = time.time()
            logger.info(
                f"Processing sample {i+1}/{len(memory_samples)}. Raw messages count: {len(messages) if messages else 'None'}"
            )
            if not messages:
                logger.info(f"Sample {i+1} is empty. Skipping.")
                continue
            input_text, earliest_ts, latest_ts, valid_msgs = (
                "",
                float("inf"),
                float("-inf"),
                0,
            )
            for msg_idx, msg in enumerate(messages):
                txt, ts = msg.get("message_text", ""), msg.get("timestamp")
                logger.debug(
                    f"  Sample {i+1}, Msg {msg_idx+1}: Text='{str(txt)[:50]}...', Timestamp='{ts}', Type(txt)={type(txt)}, Type(ts)={type(ts)}"
                )
                if (
                    isinstance(txt, str)
                    and txt.strip()
                    and isinstance(ts, (int, float))
                ):
                    logger.debug(f"    Msg {msg_idx+1} is valid. Adding to input_text.")
                    input_text += f"{txt}\n"
                    earliest_ts = min(earliest_ts, ts)
                    latest_ts = max(latest_ts, ts)
                    valid_msgs += 1
                else:
                    logger.debug(
                        f"    Msg {msg_idx+1} is NOT valid. Text empty/whitespace after strip: {not txt.strip() if isinstance(txt, str) else 'N/A (not str)'}, Timestamp valid: {isinstance(ts, (int, float))}"
                    )
            logger.info(
                f"Sample {i+1} processing complete. Resulting input_text (first 100 chars): '{input_text[:100]}...', valid_msgs: {valid_msgs}"
            )
            if not input_text.strip() or valid_msgs == 0:
                logger.info(
                    f"Sample {i+1} resulted in empty/invalid input_text or zero valid messages. Skipping LLM processing for this sample."
                )
                continue
            processed_count += 1
            time_info = "时间未知"
            if earliest_ts != float("inf"):
                e_dt, l_dt = datetime.fromtimestamp(
                    earliest_ts
                ), datetime.fromtimestamp(latest_ts)
                time_info = (
                    f"{e_dt.year}年, {e_dt.strftime('%m月%d日 %H:%M')}到{l_dt.strftime('%m月%d日 %H:%M')}"
                    if e_dt.year == l_dt.year
                    else f"{e_dt.strftime('%Y年%m月%d日 %H:%M')}到{l_dt.strftime('%Y年%m月%d日 %H:%M')}"
                )
            try:
                _, topic_embed_map, similar_map = (
                    await self.extract_topics_and_find_similar(
                        input_text, self.config.memory_compress_rate
                    )
                )
            except Exception as e:
                logger.error(f"样本{i+1}提取主题失败: {e}", exc_info=True)
                continue
            if not topic_embed_map:
                continue
            logger.info(f"样本{i+1}: 提取到{len(topic_embed_map)}个主题生成摘要。")
            new_topics_this_sample = []
            summary_tasks_with_info = []
            for topic, topic_emb in topic_embed_map.items():
                summary_tasks_with_info.append(
                    (
                        topic,
                        topic_emb,
                        self.hippocampus.generate_topic_specific_summary(
                            input_text, time_info, topic
                        ),
                    )
                )
            summary_gen_start_ts = time.monotonic()
            summaries_generated_this_batch = 0
            for topic, topic_emb, summary_awaitable in summary_tasks_with_info:
                try:
                    topic_summary = await summary_awaitable
                    summaries_generated_this_batch += 1
                    if not topic_summary:
                        logger.warning(f"主题'{topic}'(样本{i+1})摘要为空。")
                        continue
                    summary_emb = await self.hippocampus.get_embedding_async(
                        topic_summary, request_type="memory_summary_embed"
                    )
                    self.memory_graph.add_dot(
                        topic, topic_summary, topic_emb, summary_emb
                    )
                    added_nodes.add(topic)
                    new_topics_this_sample.append(topic)
                    elapsed_batch = time.monotonic() - summary_gen_start_ts
                    expected_time = (
                        summaries_generated_this_batch
                        * self.config.rpm_limit_delay_summary_sec
                    )
                    wait = max(0, expected_time - elapsed_batch)
                    if wait > 0:
                        await asyncio.sleep(wait)
                except Exception as e:
                    logger.error(
                        f"主题'{topic}'(样本{i+1})摘要/嵌入错误: {e}", exc_info=True
                    )
            if len(new_topics_this_sample) >= 2:
                for t1, t2 in combinations(new_topics_this_sample, 2):
                    self.memory_graph.connect_dot(t1, t2)
                    connected_edges.add("-".join(sorted((t1, t2))))
            for topic in new_topics_this_sample:
                if topic in similar_map:
                    for sim_topic, _ in similar_map[topic]:
                        if self.memory_graph.G.has_node(sim_topic):
                            self.memory_graph.connect_dot(topic, sim_topic)
                            connected_edges.add("-".join(sorted((topic, sim_topic))))
            logger.info(f"样本{i+1}处理耗时: {time.time()-sample_time:.2f}s")
        if processed_count > 0 and (added_nodes or connected_edges):
            logger.info(
                f"构建处理了{processed_count}样本。新增/更新概念:{len(added_nodes)}, 新增/强化连接:{len(connected_edges)}。同步DB..."
            )
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次构建无有效样本或无节点/连接变化，跳过同步。")
        logger.info(f"记忆构建完成，总耗时: {time.time()-start_time:.2f}s")

    async def operation_forget_topic(self, percentage: Optional[float] = None):
        start_time = time.time()
        eff_percentage = (
            percentage
            if percentage is not None
            else self.config.memory_forget_percentage
        )
        if not 0 < eff_percentage <= 1:
            eff_percentage = self.config.memory_forget_percentage
        nodes, edges = list(self.memory_graph.G.nodes()), list(
            self.memory_graph.G.edges()
        )
        if not nodes and not edges:
            logger.info("记忆图为空，无需遗忘。")
            return
        n_check, e_check = (max(1, int(len(nodes) * eff_percentage)) if nodes else 0), (
            max(1, int(len(edges) * eff_percentage)) if edges else 0
        )
        logger.info(
            f"遗忘操作: 检查 {n_check}/{len(nodes)} 节点, {e_check}/{len(edges)} 边。"
        )
        if n_check == 0 and e_check == 0:
            logger.info("需检查节点/边数为0，跳过遗忘。")
            return
        nodes_sample = random.sample(nodes, n_check) if n_check > 0 else []
        edges_sample = random.sample(edges, e_check) if e_check > 0 else []
        edge_chg, node_chg = {"weakened": [], "removed": []}, {
            "summary_removed": [],
            "node_removed": [],
        }
        now_ts = datetime.now().timestamp()
        edge_forget_thresh = self.config.memory_forget_time_hours * 3600
        node_sum_forget_thresh = self.config.node_summary_forget_time_hours * 3600
        for u, v in edges_sample:
            if not self.memory_graph.G.has_edge(u, v):
                continue
            edge_data = self.memory_graph.G[u][v]
            if now_ts - edge_data.get("last_modified", 0) > edge_forget_thresh:
                strength = edge_data.get("strength", 1) - 1
                if strength <= 0:
                    self.memory_graph.G.remove_edge(u, v)
                    edge_chg["removed"].append(f"{u}-{v}")
                else:
                    self.memory_graph.G[u][v]["strength"] = strength
                    self.memory_graph.G[u][v]["last_modified"] = now_ts
                    edge_chg["weakened"].append(f"{u}-{v} (S:{strength+1}->{strength})")
        for node_name in nodes_sample:
            if node_name not in self.memory_graph.G:
                continue
            node_data_t = self.memory_graph.get_dot(node_name)
            if not node_data_t:
                continue
            if now_ts - node_data_t[1].get("last_modified", 0) > node_sum_forget_thresh:
                removed_sum_txt = self.memory_graph.forget_topic_summary(node_name)
                if removed_sum_txt:
                    if node_name in self.memory_graph.G:
                        node_chg["summary_removed"].append(f"{node_name}")
                    else:
                        node_chg["node_removed"].append(f"{node_name}")
                elif (
                    node_name not in self.memory_graph.G
                    and node_name not in node_chg["node_removed"]
                ):
                    node_chg["node_removed"].append(node_name)
        if any(edge_chg.values()) or any(node_chg.values()):
            summary = "; ".join(
                f"{k}:{len(v)}" for k, v in {**edge_chg, **node_chg}.items() if v
            )
            logger.info(f"遗忘检测到变化: {summary}. 同步DB...")
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次遗忘无满足条件项。")
        logger.info(f"遗忘操作完成，耗时: {time.time()-start_time:.2f}s")

    async def operation_consolidate_memory(self):
        start_time = time.time()
        perc_check, sim_thresh = (
            self.config.consolidate_memory_percentage,
            self.config.consolidation_similarity_threshold,
        )
        logger.info(
            f"开始整合记忆... 检查比例:{perc_check:.1%}, 合并阈值:{sim_thresh:.2f}"
        )
        eligible_nodes = []
        for node, data_attr in self.memory_graph.G.nodes(data=True):
            node_t = self.memory_graph.get_dot(node)
            if not node_t:
                continue
            items_with_embed_count = sum(
                1
                for item in node_t[1].get("memory_items", [])
                if isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], list)
            )
            if items_with_embed_count >= 2:
                eligible_nodes.append(node)
        if not eligible_nodes:
            logger.info("无包含>=2个带嵌入记忆项的节点，无需整合。")
            return
        num_to_check = max(1, int(len(eligible_nodes) * perc_check))
        nodes_to_check_sample = (
            random.sample(eligible_nodes, num_to_check)
            if len(eligible_nodes) > num_to_check
            else eligible_nodes
        )
        logger.info(
            f"将检查 {len(nodes_to_check_sample)}/{len(eligible_nodes)} 个节点进行整合。"
        )
        merged_count, modified_nodes = 0, set()
        now_ts = datetime.now().timestamp()
        for node_name in nodes_to_check_sample:
            node_t = self.memory_graph.get_dot(node_name)
            if not node_t:
                continue
            current_mem_items = list(node_t[1].get("memory_items", []))
            items_with_embed_indexed = [
                (i, item)
                for i, item in enumerate(current_mem_items)
                if isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], list)
            ]
            if len(items_with_embed_indexed) < 2:
                continue
            items_with_embed_indexed.sort(
                key=lambda x: calculate_information_content(x[1][0]), reverse=True
            )
            final_items_for_node = [
                item
                for item in current_mem_items
                if not (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], list)
                )
            ]
            processed_original_indices = set()
            for i in range(len(items_with_embed_indexed)):
                original_idx_i, (summary_i, embedding_i) = items_with_embed_indexed[i]
                if original_idx_i in processed_original_indices:
                    continue
                current_group_representative = (summary_i, embedding_i)
                for j in range(i + 1, len(items_with_embed_indexed)):
                    original_idx_j, (summary_j, embedding_j) = items_with_embed_indexed[
                        j
                    ]
                    if original_idx_j in processed_original_indices:
                        continue
                    if cosine_similarity(embedding_i, embedding_j) >= sim_thresh:
                        logger.info(
                            f"整合节点'{node_name}': '{summary_j[:30]}...' 与 '{summary_i[:30]}...'相似. 保留信息量较高者."
                        )
                        processed_original_indices.add(original_idx_j)
                        merged_count += 1
                        modified_nodes.add(node_name)
                final_items_for_node.append(current_group_representative)
                processed_original_indices.add(original_idx_i)
            if len(final_items_for_node) < len(current_mem_items):
                self.memory_graph.G.nodes[node_name][
                    "memory_items"
                ] = final_items_for_node
                self.memory_graph.G.nodes[node_name]["last_modified"] = now_ts
        if merged_count > 0:
            logger.info(
                f"共合并{merged_count}对相似记忆项，分布在{len(modified_nodes)}个节点。同步DB..."
            )
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        else:
            logger.info("本次整合未发现需合并项。")
        logger.info(f"整合记忆完成，耗时: {time.time()-start_time:.2f}s")


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
        self._items_needing_summary_embedding_update: Dict[str, List[int]] = {}
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
                    "memory_summary_topic",
                )
            except Exception as e:
                logger.error(f"初始化主题摘要LLM失败: {e}", exc_info=True)
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
            bulk_db_upd = []
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
                    bulk_db_upd.append(
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
                    and bulk_db_upd
                    and self.db_instance is not None
                ):
                    try:
                        self.db_instance.graph_data_nodes.bulk_write(
                            bulk_db_upd, ordered=False
                        )
                        bulk_db_upd = []
                    except Exception as e:
                        logger.error(f"批量更新概念嵌入DB出错:{e}")
                await asyncio.sleep(delay)
            if bulk_db_upd and self.db_instance is not None:
                try:
                    self.db_instance.graph_data_nodes.bulk_write(
                        bulk_db_upd, ordered=False
                    )
                except Exception as e:
                    logger.error(f"最后批量更新概念嵌入DB出错:{e}")
            logger.info(f"概念嵌入更新完成。成功:{upd_c},失败:{fail_c}。")
        items_to_proc = dict(self._items_needing_summary_embedding_update)
        self._items_needing_summary_embedding_update.clear()
        if items_to_proc:
            total_sum_upd = sum(len(v) for v in items_to_proc.values())
            logger.info(
                f"开始处理{len(items_to_proc)}节点中缺失的{total_sum_upd}个摘要嵌入..."
            )
            upd_s, fail_s = 0, 0
            for concept, indices in items_to_proc.items():
                if not indices or concept not in self.memory_graph.G:
                    continue
                node_t = self.memory_graph.get_dot(concept)
                if not node_t:
                    continue
                curr_items = list(node_t[1].get("memory_items", []))
                node_mod = False
                for idx in indices:
                    if 0 <= idx < len(curr_items):
                        txt, emb = curr_items[idx]
                        if emb is None:
                            sum_emb_vec = await self.get_embedding_async(
                                txt, request_type="memory_summary_embed_补"
                            )
                            if sum_emb_vec:
                                curr_items[idx] = (txt, sum_emb_vec)
                                upd_s += 1
                                node_mod = True
                            else:
                                fail_s += 1
                            await asyncio.sleep(delay)
                if node_mod and self.db_instance is not None:
                    self.memory_graph.G.nodes[concept]["memory_items"] = curr_items
                    self.memory_graph.G.nodes[concept][
                        "last_modified"
                    ] = datetime.now().timestamp()
                    try:
                        self.db_instance.graph_data_nodes.update_one(
                            {"concept": concept},
                            {
                                "$set": {
                                    "memory_items": curr_items,
                                    "last_modified": self.memory_graph.G.nodes[concept][
                                        "last_modified"
                                    ],
                                }
                            },
                        )
                    except Exception as e:
                        logger.error(f"更新节点'{concept}'摘要嵌入DB出错:{e}")
            logger.info(f"摘要嵌入更新完成。成功:{upd_s},失败:{fail_s}。")
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

    def calculate_node_hash(self, c: str, s: List[str]) -> int:
        return hash(f"c:{c}|s:{'|'.join(sorted(list(set(s))))}")

    def calculate_edge_hash(self, s: str, t: str) -> int:
        return hash(f"e:{'-'.join(sorted([s,t]))}")

    def _create_topic_specific_summary_prompt(self, t: str, ti: str, to: str) -> str:
        return self.prompt_builder.build_topic_specific_summary_prompt(
            text_to_summarize=t, time_info=ti, topic=to
        )

    async def generate_topic_specific_summary(
        self, txt: str, time_inf: str, topic: str
    ) -> Optional[str]:
        if not self.llm_summary_by_topic:
            logger.warning("摘要LLM未初始化。")
            return None
        if not txt or not txt.strip() or not topic:
            return None
        prompt = self._create_topic_specific_summary_prompt(txt, time_inf, topic)
        try:
            resp: Optional[GeminiSDKResponse] = (
                await self.llm_summary_by_topic.generate_response_async(
                    prompt, request_type="memory_gen_summary"
                )
            )
            if resp and resp.content and resp.content.strip():
                return resp.content.strip()
            logger.warning(
                f"LLM为主题'{topic}'返回摘要为空或无效。响应:{resp.to_dict() if resp else 'None'}"
            )
            return None
        except Exception as e:
            logger.error(f"为主题'{topic}'生成摘要出错:{e}", exc_info=True)
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
            if not extracted or (
                len(extracted) == 1 and extracted[0].strip().lower() == "none"
            ):
                logger.info(f"LLM未提取到有效主题或返回<none>: '{resp.content}'")
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
        self, kw: str, depth: int = 2
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
        for node, attr in self.memory_graph.G.nodes(data=True):
            node_emb = attr.get("embedding")
            if node_emb and isinstance(node_emb, list):
                try:
                    sim = cosine_similarity(kw_emb, node_emb)
                    if sim >= self.config.keyword_retrieval_node_similarity_threshold:
                        node_t = self.memory_graph.get_dot(node)
                        if node_t:
                            summaries = [
                                item[0]
                                for item in node_t[1].get("memory_items", [])
                                if item and item[0]
                            ]
                            if summaries:
                                similar_nodes.append((node, summaries, sim))
                except Exception as e:
                    logger.error(f"计算关键词'{kw}'与节点'{node}'相似度出错:{e}")
        similar_nodes.sort(key=lambda x: x[2], reverse=True)
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
        sorted_act_nodes = sorted(act_map.items(), key=lambda x: x[1], reverse=True)
        if not self.llm_embedding_topic:
            logger.warning("嵌入模型不可用，无法相似度排序。")
            return []
        txt_emb = await self.get_embedding_async(
            txt, request_type="memory_input_embed_retrieve"
        )
        if not txt_emb:
            logger.warning("无法获取输入文本嵌入。")
            return []
        candidates_rerank: List[Tuple[str, str, float]] = []
        top_n_scan = self.config.retrieval_top_activated_nodes_to_scan
        min_sum_sim = self.config.retrieval_min_summary_similarity
        for node, act_score in sorted_act_nodes[:top_n_scan]:
            node_t = self.memory_graph.get_dot(node)
            if not node_t:
                continue
            for s_txt, s_emb in node_t[1].get("memory_items", []):
                if s_emb and isinstance(s_emb, list):
                    try:
                        sim = cosine_similarity(txt_emb, s_emb)
                        if sim >= min_sum_sim:
                            candidates_rerank.append((node, s_txt, sim))
                    except Exception:
                        pass
        if not candidates_rerank:
            logger.info("无摘要通过相似度阈值。")
            return []
        candidates_rerank.sort(key=lambda x: x[2], reverse=True)
        to_llm_rerank = candidates_rerank[
            : self.config.retrieval_max_candidates_for_llm_rerank
        ]
        final_mems: List[Tuple[str, str]] = []
        if self.llm_re_rank and to_llm_rerank:
            logger.info(
                f"LLM({self.llm_re_rank.model_name})重排{len(to_llm_rerank)}条候选记忆..."
            )
            prompt = self._create_bulk_relevance_check_prompt(txt, to_llm_rerank)
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
                        temp_llm_sorted = [
                            to_llm_rerank[idx]
                            for idx in indices
                            if 0 <= idx < len(to_llm_rerank)
                        ]
                        final_mems = [(t, s) for t, s, _ in temp_llm_sorted]
                        logger.info(f"LLM成功重排{len(final_mems)}条记忆。")
                    except Exception as e:
                        logger.error(f"解析LLM重排索引失败:{e}.回退.")
                        final_mems = [(t, s) for t, s, _ in to_llm_rerank]
                else:
                    logger.warning("LLM重排响应为空,回退.")
                    final_mems = [(t, s) for t, s, _ in to_llm_rerank]
            except Exception as e:
                logger.error(f"LLM重排调用失败:{e}", exc_info=True)
                final_mems = [(t, s) for t, s, _ in to_llm_rerank]
        else:
            if not self.llm_re_rank:
                logger.debug("重排LLM未配置，使用初步筛选结果。")
            final_mems = [(t, s) for t, s, _ in to_llm_rerank]
        seen_sum, unique_final = set(), []
        for t, s_txt in final_mems:
            if s_txt not in seen_sum:
                seen_sum.add(s_txt)
                unique_final.append((t, s_txt))
            if len(unique_final) >= max_mem:
                break
        logger.info(
            f"记忆检索完成,找到{len(unique_final)}条。耗时:{time.time()-start_t:.3f}s"
        )
        return unique_final

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
