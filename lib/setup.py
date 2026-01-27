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
    "lz4",
    "orjson",
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl ; platform_machine == 'x86_64'",
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl ; platform_machine == 'aarch64'",
    "text_preprocessing @ git+https://github.com/ARTFL-Project/text-preprocessing@v1.1.0.1#egg=text_preprocessing",
    "fastapi==0.110.3",
    "psycopg2",
    "gunicorn",
    "uvicorn",
    "uvloop",
    "httptools",
    "philologic>=4.7.5.0,<5",
    "regex",
    "ahocorasick-rs",
    "msgspec",
]


setup(
    name="textpair",
    version="2.3.0",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    python_requires=">=3.10",
    packages=["textpair"],
    install_requires=dependencies,
)
