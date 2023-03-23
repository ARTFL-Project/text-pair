"""Global imports for main textpair function"""
from .generate_ngrams import Ngrams
from .text_parser import parse_files
from .web_loader import create_web_app
from .parse_config import get_config
from .vector_space_aligner import run_vsa
from .banality_finder import banality_auto_detect, phrase_matcher
from .utils import get_text
from .alignment_merger import merge_alignments
