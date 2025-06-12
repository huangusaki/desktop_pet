import logging
import math
from typing import Optional, Dict, Any, Tuple, List
from ..database.mongo_handler import MongoHandler

logger = logging.getLogger("RelationshipManager")
RELATIONSHIP_LEVELS_CONFIG: List[Tuple[int, int, int, str, str]] = [
    (
        -1000,
        -900,
        -4,
        "Mortal Enemy",
        "Irreconcilable, expresses extreme hatred and abhorrence. Speech is aggressive and contemptuous; may actively initiate hostile actions.",
    ),
    (
        -900,
        -700,
        -3,
        "Hostile",
        "Full of animosity and distrust. Speech is often cutting, sarcastic, or belittling; frequently questions or mocks the other; generally has a resistant or antagonistic attitude.",
    ),
    (
        -700,
        -450,
        -2,
        "Distant/Alienated",
        "Indifferent and deliberately maintains distance. Responses are brief and perfunctory; unwilling to engage in much interaction and may actively avoid the other person.",
    ),
    (
        -450,
        -150,
        -1,
        "Cold/Indifferent",
        "Exhibits a cold attitude and lacks enthusiasm. Conversations are typically business-like, brief, or purely functional; rarely initiates interaction; shows minimal emotional investment.",
    ),
    (
        -150,
        150,
        0,
        "Neutral/Ordinary",
        "Calm and objective, showing no obvious preference or aversion. Interactions are standard, without particular closeness or distance; maintains a polite social distance.",
    ),
    (
        150,
        450,
        1,
        "Friendly",
        "Friendly and polite attitude. Open to light, everyday conversation and shows basic concern, but emotional connection is still superficial.",
    ),
    (
        450,
        700,
        2,
        "Close",
        "Harmonious relationship. Willing to share personal thoughts and feelings, proactively shows care, and can engage in deeper, more meaningful conversations. A good level of mutual trust.",
    ),
    (
        700,
        900,
        3,
        "Trusted",
        "Deep level of trust. Willing to rely on the other's judgment and support in important matters. Strong emotional connection; views the other as a valuable friend or partner.",
    ),
    (
        900,
        1001,
        4,
        "Dependent",
        "Extreme trust and strong emotional dependence. Views the other as an indispensable pillar in their life, as close and intimate as family; willing to make significant sacrifices for them.",
    ),
]
MAX_SCORE = 1000
MIN_SCORE = -1000
HIGH_POSITIVE_THRESHOLD = 700
LOW_NEGATIVE_THRESHOLD = -700
EXTREME_VALUE_SCALER_MULTIPLIER = 0.4
MAX_BASE_CHANGE_PER_INTERACTION = 20
MIN_BASE_CHANGE_PER_INTERACTION = -20


class RelationshipManager:
    """
    管理宠物和用户之间的关系（好感度）。
    引入了动态调整机制，使得好感度变化更真实、非线性。
    """

    def __init__(self, mongo_handler: MongoHandler, user_name: str, pet_name: str):
        if not mongo_handler or not mongo_handler.is_connected():
            raise ValueError("MongoHandler is not connected or initialized.")
        self.mongo_handler = mongo_handler
        self.user_name = user_name
        self.pet_name = pet_name
        self.current_score: Optional[int] = None
        logger.info(f"RelationshipManager for {pet_name}-{user_name} initialized.")

    def _get_current_score(self) -> int:
        """从数据库获取当前好感度分数，如果不存在则返回中性值0。优先使用缓存。"""
        if self.current_score is None:
            score = self.mongo_handler.get_favorability_score(
                self.user_name, self.pet_name
            )
            self.current_score = score if score is not None else 0
            logger.info(
                f"Fetched initial favorability score from DB: {self.current_score}"
            )
        return self.current_score

    def _get_context_scaler(
        self, current_score: int, is_positive_change: bool
    ) -> float:
        """
        核心动态调整逻辑：根据当前好感度分数，计算一个变化缩放因子。
        """
        normalized_score = current_score / MAX_SCORE
        scaler = 1.0
        if is_positive_change:
            if current_score >= 0:
                scaler = 1.0 - 0.8 * (normalized_score**2)
                if current_score > HIGH_POSITIVE_THRESHOLD:
                    scaler *= EXTREME_VALUE_SCALER_MULTIPLIER
            else:
                scaler = 1.0 + 0.8 * (normalized_score**2)
        else:
            if current_score > 0:
                scaler = 1.0 + 0.8 * (normalized_score**2)
            else:
                scaler = 1.0 - 0.8 * (normalized_score**2)
                if current_score < LOW_NEGATIVE_THRESHOLD:
                    scaler *= EXTREME_VALUE_SCALER_MULTIPLIER
        return max(0.1, scaler)

    async def update_favorability(self, base_change: int):
        """
        更新好感度分数。
        'base_change' 是 LLM 建议的基础变化值。
        此方法会应用动态缩放和边界限制。
        """
        if base_change == 0:
            logger.info("Base change is 0, no favorability update needed.")
            return
        current_score = self._get_current_score()
        base_change = max(
            MIN_BASE_CHANGE_PER_INTERACTION,
            min(base_change, MAX_BASE_CHANGE_PER_INTERACTION),
        )
        is_positive = base_change > 0
        scaler = self._get_context_scaler(current_score, is_positive)
        actual_change = base_change * scaler
        final_change = round(actual_change)
        if base_change != 0 and final_change == 0:
            final_change = 1 if base_change > 0 else -1
        new_score = current_score + final_change
        new_score = max(MIN_SCORE, min(new_score, MAX_SCORE))
        logger.info(
            f"Favorability update for {self.user_name}-{self.pet_name}: "
            f"CurrentScore={current_score}, LLMBaseChange={base_change}, "
            f"ContextScaler={scaler:.2f}, ActualChange={actual_change:.2f}, "
            f"FinalChange={final_change}, NewScore={new_score}"
        )
        updated = await self.mongo_handler.update_favorability_score(
            self.user_name, self.pet_name, new_score
        )
        if updated:
            self.current_score = new_score
            logger.info(
                f"Favorability score successfully updated in DB to {self.current_score}."
            )
        else:
            logger.error("Failed to update favorability score in the database.")

    def get_current_relationship_level(self) -> Optional[Dict[str, Any]]:
        """
        根据当前分数确定关系等级。
        """
        score = self._get_current_score()
        for min_score, max_score, level_id, name, desc in RELATIONSHIP_LEVELS_CONFIG:
            if min_score <= score < max_score:
                level_info = {
                    "level_id": level_id,
                    "name": name,
                    "description": desc,
                    "score": score,
                }
                logger.debug(f"Current score {score} falls into level '{name}'.")
                return level_info
        logger.warning(
            f"Score {score} is outside all defined relationship level ranges."
        )
        if score >= RELATIONSHIP_LEVELS_CONFIG[-1][0]:
            _, _, level_id, name, desc = RELATIONSHIP_LEVELS_CONFIG[-1]
            return {
                "level_id": level_id,
                "name": name,
                "description": desc,
                "score": score,
            }
        return None
