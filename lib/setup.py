#!/usr/bin/env python3
"""Python package install script"""

import os
import platform

from setuptools import setup

arch = platform.uname().machine
if arch == "x86_64":
    os.system("rm /usr/local/bin/compareNgrams")
    os.system("cp core/binary/x86_64/compareNgrams /usr/local/bin/")
    os.system("chmod +x /usr/local/bin/compareNgrams")
elif arch == "aarch64":
    os.system("rm /usr/local/bin/compareNgrams")
    os.system("cp core/binary/aarch64/compareNgrams /usr/local/bin/")
    os.system("chmod +x /usr/local/bin/compareNgrams")
else:
    print("Only x86_64 and ARM are supported at this time.")
    exit()


dependencies = [
    "multiprocess",
    "mmh3",
    "unidecode",
    "tqdm",
    "cython",
    "pystemmer",
    "lxml",
    "namedlist",  # to remove
    "recordclass",
    "sentence-transformers",
    "sacremoses",  # required by some models for tokenization
    "lz4",
    "orjson",
    "text_preprocessing @ git+https://github.com/ARTFL-Project/text-preprocessing@v1.0.2#egg=text_preprocessing",
    "fastapi",
    "psycopg2",
    "gunicorn",
    "uvicorn",
    "uvloop",
    "httptools",
    "philologic>=4.7.4.4",
    "regex",
    "ahocorasick-rs"
]


setup(
    name="textpair",
    version="2.1",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    python_requires=">=3.10",
    packages=["textpair"],
    scripts=["scripts/textpair"],
    install_requires=dependencies,
)
