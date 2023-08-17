#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 cubicibo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from distutils.util import convert_path
from typing import Any, Dict
import sys

from setuptools import setup

NAME = 'piliq'

meta: Dict[str, Any] = {}
with open(convert_path(f"{NAME}/__metadata__.py"), encoding='utf-8') as f:
    exec(f.read(), meta)

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=meta['__version__'],
    author=meta['__author__'],
    description='Basic PIL-libimagequant inteface to palettize RGBA images.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=[NAME,],
    url='https://github.com/cubicibo/piliq',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={NAME: ['*.dll']} if sys.platform == 'win32' else {},
)
