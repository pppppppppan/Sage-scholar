"""Materialize images only at the provider boundary, never in conversation/history."""
import base64
from copy import deepcopy
from pathlib import Path


def image_labels(images):
    return "\n".join(f"附图 {index} [{item['evidence_id']}]，PDF 第 {item['page']} 页。" for index, item in enumerate(images, 1))


def attach_images(messages, images):
    result = deepcopy(messages)
    # prepare() sees only a small textual manifest; binary pixels are added afterwards.
    manifest = image_labels(images)
    blocks = [{"type": "text", "text": manifest}]
    for item in images:
        data = Path(item["path"]).read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 20 * 1024 * 1024:
            raise ValueError("视觉输入必须是大小不超过 20MB 的已渲染 PNG")
        blocks.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}})
    for message in reversed(result):
        if message.get("role") == "user" and message.get("content") == manifest:
            message["content"] = blocks
            return result
    raise ValueError("Context 未保留图像来源清单，拒绝发送无法绑定来源的图片")
