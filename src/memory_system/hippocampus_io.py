import time
from datetime import datetime
from pymongo import UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from pymongo.database import Database
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
import uuid
from .memory_config import MemoryConfig
from .hippocampus_graph import MemoryGraph
import logging

logger = logging.getLogger("memory_system.io")
if TYPE_CHECKING:
    from .hippocampus_core_logic import Hippocampus


class EntorhinalCortex:
    def __init__(self, hippocampus: "Hippocampus"):
        self.hippocampus = hippocampus
        self.memory_graph: MemoryGraph = hippocampus.memory_graph
        self.config: MemoryConfig = hippocampus.config
        self.db: Optional[Database] = hippocampus.db_instance

    def _get_chat_history_for_memory_sample(self) -> List[Dict[str, Any]]:
        """
        获取一个基于时间连续性的聊天记录片段作为记忆样本。
        会从头扫描聊天记录，寻找符合条件的连续对话片段。
        """
        if self.db is None or not hasattr(
            self.db, self.hippocampus.chat_collection_name
        ):
            logger.error("数据库或聊天集合未配置，无法获取记忆样本。")
            return []
        time_gap_threshold_seconds = self.config.sample_time_gap_threshold_seconds
        min_snippet_messages = self.config.sample_min_snippet_messages
        max_snippet_messages = self.config.sample_max_snippet_messages
        max_memorized_time = self.config.max_memorized_time_per_msg
        bot_name_filter = self.hippocampus.bot_name_for_history_filter
        try:
            query_filter = {"role_play_character": bot_name_filter}
            all_messages_cursor = (
                self.db[self.hippocampus.chat_collection_name]
                .find(query_filter)
                .sort("timestamp", ASCENDING)
            )
            all_messages = list(all_messages_cursor)
            if not all_messages:
                logger.info("数据库中没有符合条件的消息，无法采样。")
                return []
            if len(all_messages) < min_snippet_messages:
                logger.info(
                    f"数据库中符合条件的消息总数 ({len(all_messages)}) 少于设定的最小片段消息数 ({min_snippet_messages})，无法采样。"
                )
                return []
            logger.info(f"开始全表扫描 {len(all_messages)} 条消息以构建记忆样本...")
            current_snippet_start_index = 0
            while current_snippet_start_index < len(all_messages):
                potential_snippet: List[Dict[str, Any]] = []
                last_timestamp_in_chain = -1
                for i in range(current_snippet_start_index, len(all_messages)):
                    current_message = all_messages[i]
                    if not potential_snippet:
                        potential_snippet.append(current_message)
                        last_timestamp_in_chain = current_message["timestamp"]
                    else:
                        if (
                            current_message["timestamp"] - last_timestamp_in_chain
                        ) < time_gap_threshold_seconds:
                            potential_snippet.append(current_message)
                            last_timestamp_in_chain = current_message["timestamp"]
                        else:
                            break
                    if len(potential_snippet) >= max_snippet_messages:
                        break
                logger.debug(
                    f"形成潜在片段，起始索引: {current_snippet_start_index}, 长度: {len(potential_snippet)}."
                    f" (首消息ID: {potential_snippet[0].get('_id') if potential_snippet else 'N/A'}, "
                    f"尾消息ID: {potential_snippet[-1].get('_id') if potential_snippet else 'N/A'})"
                )
                if len(potential_snippet) < min_snippet_messages:
                    logger.debug(
                        f"潜在片段长度 ({len(potential_snippet)}) 小于最小要求 ({min_snippet_messages})。"
                    )
                    if not potential_snippet:
                        current_snippet_start_index += 1
                    else:
                        current_snippet_start_index += 1
                    continue
                last_message_in_potential_snippet = potential_snippet[-1]
                last_message_timestamp_val = last_message_in_potential_snippet.get(
                    "timestamp"
                )
                if last_message_timestamp_val is not None:
                    current_system_timestamp = datetime.now().timestamp()
                    time_since_last_message = (
                        current_system_timestamp - last_message_timestamp_val
                    )
                    if (
                        time_since_last_message
                        < self.config.sample_time_gap_threshold_seconds
                    ):
                        logger.info(
                            f"潜在片段 (首消息ID: {potential_snippet[0].get('_id')}) 因最后消息太新而被跳过。"
                            f"最后消息时间距现在: {time_since_last_message:.0f}s (阈值: {self.config.sample_time_gap_threshold_seconds}s)."
                        )
                        current_snippet_start_index += 1
                        continue
                else:
                    logger.warning(
                        f"潜在片段 (首消息ID: {potential_snippet[0].get('_id')}) 最后一条消息无时间戳，跳过新近度检查。"
                    )
                    current_snippet_start_index += 1
                    continue
                all_messages_in_snippet_pass_memorized_times_check = True
                for msg_in_snippet in potential_snippet:
                    if msg_in_snippet.get("memorized_times", 0) >= max_memorized_time:
                        all_messages_in_snippet_pass_memorized_times_check = False
                        logger.debug(
                            f"潜在片段因消息 ID '{msg_in_snippet.get('_id')}' (memorized_times: {msg_in_snippet.get('memorized_times', 0)}) 不合格。"
                        )
                        break
                if not all_messages_in_snippet_pass_memorized_times_check:
                    current_snippet_start_index += 1
                    continue
                ids_to_update = [
                    msg["_id"] for msg in potential_snippet if "_id" in msg
                ]
                if ids_to_update:
                    update_result = self.db[
                        self.hippocampus.chat_collection_name
                    ].update_many(
                        {"_id": {"$in": ids_to_update}},
                        {"$inc": {"memorized_times": 1}},
                    )
                    logger.info(
                        f"通过顺序扫描采样到合格连续对话片段 (长度: {len(potential_snippet)}，"
                        f"首消息ID: {potential_snippet[0].get('_id')}, 尾消息ID: {potential_snippet[-1].get('_id')})。"
                        f"更新了 {update_result.modified_count} 条记录的 memorized_times。"
                    )
                else:
                    logger.info(
                        f"通过顺序扫描采样到合格连续对话片段 (长度: {len(potential_snippet)}，"
                        f"首消息ID: {potential_snippet[0].get('_id')}, 尾消息ID: {potential_snippet[-1].get('_id')})。"
                        f"无需更新 memorized_times (无 _id?)。"
                    )
                return potential_snippet
            logger.info("全表扫描完成，未能找到符合所有条件的连续对话片段。")
            return []
        except Exception as e:
            logger.error(f"顺序扫描获取连续对话样本时发生意外错误: {e}", exc_info=True)
            return []

    def get_memory_sample(self) -> List[List[Dict[str, Any]]]:
        chat_samples = []
        num_samples_to_build = getattr(self.config, "build_memory_sample_num", 1)
        for _ in range(num_samples_to_build):
            messages = self._get_chat_history_for_memory_sample()
            if messages:
                chat_samples.append(messages)
            else:
                logger.info("本次未能获取到一个有效的记忆样本片段。")
        if chat_samples:
            logger.info(f"共获取到 {len(chat_samples)} 个记忆样本片段。")
        else:
            logger.info("未能获取到任何记忆样本片段。")
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
            f"从DB同步记忆完成。加载 {len(self.memory_graph.G.nodes())} 节点, {len(self.memory_graph.G.edges())} 边。耗时: {time.time() - start_time:.3f}s"
        )
        total_summary_levels_to_update = sum(
            len(v)
            for v in self.hippocampus._items_needing_summary_embedding_update.values()
        )
        logger.info(
            f"需补概念嵌入: {len(self.hippocampus._nodes_needing_embedding_update)} 个节点。需补层级摘要嵌入: {total_summary_levels_to_update} 个层级。"
        )
