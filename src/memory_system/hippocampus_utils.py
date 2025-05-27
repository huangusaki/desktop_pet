import math
from collections import Counter
from typing import List
import numpy as np


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
