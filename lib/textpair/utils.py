"""Various utilities for textpair"""

import gc
from html import unescape as unescape_html
from xml.sax.saxutils import unescape as unescape_xml

import regex as re
import torch

TAGS = re.compile(r"<[^>]+>")
PHILO_TEXT_OBJECT_LEVELS = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}


def clean_text(text: str) -> str:
    """Cleaning text function which removes tags and converts entities"""
    text = TAGS.sub("", text)
    text = unescape_xml(text)
    text = unescape_html(text)
    text = text.replace("\n", " ")
    text = text.strip()
    return text


def get_text(start_byte: int, end_byte: int, filename: str, length: int = 300) -> str:
    """Grab all texts"""
    if start_byte < 0:
        start_byte = 0
    length = end_byte - start_byte
    with open(filename, "rb") as text_file:
        text_file.seek(start_byte)
        text: str = text_file.read(length).decode("utf8", "ignore")

    # Remove leading and closing tags
    if text.startswith("<"):
        text = re.sub(r"^<[^>]+>", "", text, count=1).strip()
    if text.endswith(">"):
        text = re.sub(r"<[^>]+>$", "", text, count=1).strip()
    # Remove unclosed tags at the end
    text = re.sub(r"<[^>]+$", "", text).strip()
    return clean_text(text)


def text_object_upper_bound(config) -> str:
    """Find the text object level above the one specified in the config"""
    object_type_to_level = {v: k for k, v in PHILO_TEXT_OBJECT_LEVELS.items()}
    text_object_level = PHILO_TEXT_OBJECT_LEVELS[config["text_object_type"]]
    if text_object_level == 1:
        return "doc"
    return object_type_to_level[text_object_level - 1]


def clear_device_cache():
    """Release cached memory across all backends (CPU, CUDA, MPS, XPU)"""
    gc.collect()
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "memory") and hasattr(torch.cpu.memory, "empty_cache"):
        torch.cpu.memory.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
