"""Global imports for main textpair function"""
from .alignment_merger import merge_alignments
from .banality_finder import banality_auto_detect, phrase_matcher
from .generate_ngrams import Ngrams
from .parse_config import get_config
from .text_parser import parse_files
from .utils import get_text
from .vector_space_aligner import run_vsa
from .web_loader import create_web_app
