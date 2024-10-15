#!/usr/bin/env python3
"""Python package install script"""

from setuptools import setup

dependencies = [
    "multiprocess",
    "mmh3",
    "unidecode",
    "tqdm",
    "cython",
    "pystemmer",
    "lxml",
    "namedlist",
    "sentence-transformers~=3.2",
    "lz4",
    "orjson",
    "text_preprocessing @ git+https://github.com/ARTFL-Project/text-preprocessing@v1.1.0.1#egg=text_preprocessing",
    "fastapi==0.110.3",
    "psycopg2",
    "gunicorn",
    "uvicorn",
    "uvloop",
    "httptools",
    "philologic>=4.7.5.0",
    "regex",
    "ahocorasick-rs",
    "msgspec"
]


setup(
    name="textpair",
    version="2.2",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    python_requires=">=3.10",
    packages=["textpair"],
    install_requires=dependencies,
)
