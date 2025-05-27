import random
from datetime import datetime
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any
import uuid
import logging

logger = logging.getLogger("memory_system.graph")
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
