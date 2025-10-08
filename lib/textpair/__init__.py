"""Global imports for main textpair function"""
from .parse_config import get_config
from .passage_classifier import classify_passages
from .sequence_alignment import Ngrams, banality_auto_detect, merge_alignments, phrase_matcher
from .text_parser import parse_files
from .utils import get_text
from .vector_space_alignment import run_vsa
from .web_loader import create_web_app
