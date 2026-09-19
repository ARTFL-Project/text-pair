"""Various utilities for textpair"""

import ctypes
import gc
from html import unescape as unescape_html
from xml.sax.saxutils import unescape as unescape_xml

import regex as re

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
    # Imported here: torch is heavy and the alignment path never calls this.
    import torch

    gc.collect()
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "memory") and hasattr(torch.cpu.memory, "empty_cache"):
        torch.cpu.memory.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


# glibc serves an allocation this large with mmap and hands the pages straight
# back on free, so a worker faults in and zeroes a fresh buffer for every
# document it decompresses. Below the threshold the same heap is reused. 256MB
# covers all but a handful of documents, and those still go through mmap rather
# than being held for the rest of the run.
MMAP_THRESHOLD = 256 * 1024 * 1024
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3
_TUNED = False


def tune_allocator() -> None:
    """Stop glibc returning every large buffer to the kernel between documents.

    Worth about 7M page faults and 10% of the n-gram stage's wall clock on a
    3,630-document corpus. A no-op without glibc, so on macOS nothing happens.
    """
    global _TUNED
    if _TUNED:
        return
    _TUNED = True
    try:
        libc = ctypes.CDLL("libc.so.6")
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        return
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    try:
        mallopt(_M_MMAP_THRESHOLD, MMAP_THRESHOLD)
        mallopt(_M_TRIM_THRESHOLD, MMAP_THRESHOLD)
    except OSError:
        pass
