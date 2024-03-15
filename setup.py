#!/usr/bin/env python
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

exts = [
    Extension(
        name="match_ttsg.utils.monotonic_align.core",
        sources=["match_ttsg/utils/monotonic_align/core.pyx"],
    )
]

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "match_ttsg", "VERSION")) as fin:
    version = fin.read().strip()

setup(
    name="match_ttsg",
    version=version,
    description="🍵 MATCH-TTSG: UNIFIED SPEECH AND GESTURE SYNTHESIS USING FLOW MATCHING",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/MATCH-TTSG",
    install_requires=[str(r) for r in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))],
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "match_ttsg-data-stats=match_ttsg.utils.generate_data_statistics:main",
            "match_ttsg-tts=match_ttsg.cli:cli",
            "match_ttsg-tts-app=match_ttsg.app:main",
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
    python_requires=">=3.9.0",
)
