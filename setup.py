#!/usr/bin/env python3

import os
import sys
import setuptools
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Get the long description from the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Define the extension module with all required source files
ext_modules = [
    Pybind11Extension(
        "deepquote_simulator",
        [
            "src/bindings/python_bindings.cpp",
            "src/core/order.cpp",
            "src/core/types.cpp",
            "src/market/trader.cpp",
            "src/market/rl_trader.cpp",
            "src/market/market_simulator.cpp",
            "src/market/order_book.cpp",
            "src/market/matching_engine.cpp",
        ],
        include_dirs=["include"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="deepquote-simulator",
    version="1.0.0",
    author="DeepQuote Team",
    author_email="team@deepquote.com",
    description="A C++ reinforcement learning market maker system with realistic limit order book simulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DeepQuote",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    include_package_data=True,
) 