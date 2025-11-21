"""
LLM 工具函数模块
包含图片压缩、文本提取等共享工具函数
"""
import base64
import re
from PIL import Image
import io
import logging
from typing import Tuple

logger = logging.getLogger("llm_utils")


def compress_base64_image_by_scale(
    base64_data: str, target_size: int = int(0.8 * 1024 * 1024)
) -> str:
    """
    压缩 base64 编码的图片到目标大小
    
    Args:
        base64_data: base64 编码的图片数据
        target_size: 目标大小(字节),默认 0.8MB
        
    Returns:
        压缩后的 base64 编码图片数据
    """
    try:
        image_data = base64.b64decode(base64_data)
        if len(image_data) <= target_size:
            return base64_data
            
        img = Image.open(io.BytesIO(image_data))
        original_width, original_height = img.size
        current_quality = 85
        
        # 计算初始缩放比例
        scale = (
            (target_size / len(image_data)) ** 0.5
            if len(image_data) > target_size
            else 1.0
        )
        scale = min(scale, 0.95)
        
        compressed_data_to_return_if_loop_fails = image_data
        
        # 最多尝试5次压缩
        for attempt in range(5):
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            if new_width <= 0 or new_height <= 0:
                logger.warning(
                    f"计算出的新尺寸无效 ({new_width}x{new_height})，源尺寸 {original_width}x{original_height}, 缩放比例 {scale:.2f}。将使用原始图片。"
                )
                return base64_data
                
            output_buffer = io.BytesIO()
            img_format_to_save = (
                img.format
                if img.format and img.format.upper() in ["PNG", "WEBP", "GIF"]
                else "JPEG"
            )
            
            temp_img_copy = img.copy()
            
            # 处理 GIF 动画
            if (
                img_format_to_save == "GIF"
                and getattr(temp_img_copy, "is_animated", False)
                and temp_img_copy.n_frames > 1
            ):
                frames = []
                durations = []
                loop = temp_img_copy.info.get("loop", 0)
                
                try:
                    for frame_idx in range(temp_img_copy.n_frames):
                        temp_img_copy.seek(frame_idx)
                        current_duration = temp_img_copy.info.get("duration", 100)
                        durations.append(current_duration)
                        frame_rgba = temp_img_copy.convert("RGBA")
                        resized_frame = frame_rgba.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        frames.append(resized_frame)
                except EOFError:
                    pass
                    
                if frames:
                    frames[0].save(
                        output_buffer,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        optimize=True,
                        duration=durations,
                        loop=loop,
                        disposal=2,
                    )
                else:
                    img_format_to_save = "JPEG"
                    
            # 处理静态图片
            if img_format_to_save != "GIF":
                resized_img = temp_img_copy.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                
                if img_format_to_save == "PNG":
                    if resized_img.mode == "RGBA" or "A" in resized_img.mode:
                        resized_img.save(output_buffer, format="PNG", optimize=True)
                    else:
                        resized_img.convert("RGB").save(
                            output_buffer,
                            format="JPEG",
                            quality=current_quality,
                            optimize=True,
                        )
                        img_format_to_save = "JPEG"
                elif img_format_to_save == "WEBP":
                    resized_img.save(
                        output_buffer,
                        format="WEBP",
                        quality=current_quality,
                        lossless=False,
                    )
                else:
                    img_format_to_save = "JPEG"
                    if resized_img.mode in ("RGBA", "LA", "P"):
                        resized_img = resized_img.convert("RGB")
                    resized_img.save(
                        output_buffer,
                        format="JPEG",
                        quality=current_quality,
                        optimize=True,
                    )
                    
            compressed_data_loop = output_buffer.getvalue()
            compressed_data_to_return_if_loop_fails = compressed_data_loop
            
            # 检查是否达到目标大小
            if len(compressed_data_loop) <= target_size:
                final_format_check = Image.open(io.BytesIO(compressed_data_loop)).format
                logger.info(
                    f"压缩图片 (尝试 {attempt + 1}): {original_width}x{original_height} ({img.format or 'N/A'} -> {final_format_check or img_format_to_save}). "
                    f"大小: {len(image_data) / 1024:.1f}KB -> {len(compressed_data_loop) / 1024:.1f}KB (目标: {target_size / 1024:.1f}KB)"
                )
                return base64.b64encode(compressed_data_loop).decode("utf-8")
                
            # 调整压缩参数
            if img_format_to_save in ["JPEG", "WEBP"] and current_quality > 60:
                current_quality -= 10
            else:
                scale *= 0.85
                
            logger.info(
                f"压缩后仍然过大 (尝试 {attempt + 1}, {len(compressed_data_loop) / 1024:.1f}KB). 下次 scale={scale:.2f}, quality={current_quality}"
            )
            
        logger.warning(
            f"多次压缩后大小 {len(compressed_data_to_return_if_loop_fails) / 1024:.1f}KB 仍大于目标 {target_size / 1024:.1f}KB. 返回当前最佳压缩结果。"
        )
        return base64.b64encode(compressed_data_to_return_if_loop_fails).decode("utf-8")
        
    except Exception as e:
        logger.error(f"压缩图片失败: {str(e)}", exc_info=True)
        return base64_data


def extract_reasoning(content: str) -> Tuple[str, str]:
    """
    从内容中提取思考过程
    
    Args:
        content: 包含思考标签的文本内容
        
    Returns:
        (清理后的内容, 思考过程) 元组
    """
    if not isinstance(content, str):
        return "", ""
        
    match = re.search(
        r"<(?:think|thought)>(.*?)</(?:think|thought)>",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    
    if match:
        reasoning = match.group(1).strip()
        cleaned_content = content.replace(match.group(0), "", 1).strip()
        return cleaned_content, reasoning
        
    return content.strip(), ""
