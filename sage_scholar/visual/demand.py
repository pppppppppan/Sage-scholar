"""Cheap visual demand decision; never adds a classifier/model call to ordinary QA."""
import re

_LOCATOR = re.compile(r'(?:figure|fig\.?|table|图|表)\s*\d+|第\s*\d+\s*页|\bpage\s+\d+\b', re.I)
_VISUAL = re.compile(r'架构图|流程图|示意图|图像|图片|曲线|柱状图|散点图|热力图|图形|图表|\b(?:diagram|image|plot|chart|curve|figure)\b', re.I)


def needs_visual(query, *, mode='auto', text_available=True):
    if mode not in {'off', 'auto', 'on'}:
        raise ValueError('visual_mode 必须是 off / auto / on')
    if mode == 'off':
        return False
    return mode == 'on' or bool(_LOCATOR.search(query) or _VISUAL.search(query)) or not text_available
