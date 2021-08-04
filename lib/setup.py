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

setup(
    name="textpair",
    version="2.0-beta.4",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    packages=["textpair"],
    scripts=["scripts/textpair"],
    install_requires=[
        "multiprocess",
        "mmh3",
        "unidecode",
        "tqdm",
        "cython",
        "pystemmer",
        "lxml",
        "gensim",
        "namedlist",
        "text_preprocessing @ git+https://github.com/ARTFL-Project/text-preprocessing@v0.8.3#egg=text_preprocessing",
    ],
    extras_require={"web": ["fastapi", "psycopg2", "gunicorn", "uvicorn", "uvloop", "httptools",]},
)
