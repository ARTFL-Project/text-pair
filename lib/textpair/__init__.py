"""Global imports for main textpair function"""
from .sequence_alignment import merge_alignments, banality_auto_detect, phrase_matcher, Ngrams
from .parse_config import get_config
from .text_parser import parse_files
from .utils import get_text
from .vector_space_alignment import run_vsa
from .web_loader import create_web_app
