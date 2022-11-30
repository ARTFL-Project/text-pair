#!/usr/bin/env python3
"""Python package install script"""

import os
from sys import platform

from setuptools import setup

if platform == "linux":
    os.system("rm /usr/local/bin/compareNgrams")
    os.system("cp core/binary/linux_x64/compareNgrams /usr/local/bin/")
    os.system("chmod +x /usr/local/bin/compareNgrams")
elif platform == "darwin":
    os.system("rm /usr/local/bin/compareNgrams")
    os.system("cp core/binary/darwin/compareNgrams /usr/local/bin/")
    os.system("chmod +x /usr/local/bin/compareNgrams")
else:
    print("Only 64 bit linux and MacOS are supported at this time.")
    exit()


dependencies = [
    "multiprocess",
    "mmh3",
    "unidecode",
    "tqdm",
    "cython",
    "pystemmer",
    "lxml",
    "gensim",
    "namedlist",  # to remove
    "recordclass",
    "sentence-transformers",
    "lz4",
    "orjson",
    "text_preprocessing @ git+https://github.com/ARTFL-Project/text-preprocessing@v1.0-rc3#egg=text_preprocessing",
    "fastapi",
    "psycopg2",
    "gunicorn",
    "uvicorn",
    "uvloop",
    "httptools",
    "philologic>=4.7.3.2",
]


setup(
    name="textpair",
    version="2.0-beta.9",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    python_requires=">=3.10",
    packages=["textpair"],
    scripts=["scripts/textpair"],
    install_requires=dependencies,
)
