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

setup(name="textalign",
      version="1.0alpha",
      author="The ARTFL Project and OBVIL",
      author_email="clovisgladstone@gmail.com",
      packages=["textalign"],
      scripts=["scripts/textalign"],
      install_requires=[
          "multiprocess", "mmh3", "unidecode", "tqdm",
          "pystemmer", "flask", "flask-cors", "mod_wsgi", "psycopg2-binary"
          ]
     )
