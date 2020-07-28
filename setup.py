#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup


def get_install_requirements(path):
    content = open(os.path.join(os.path.dirname(__file__), path)).read()
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


def setup_package():
    setup(
        name="kartothek",
        author="Blue Yonder GmbH",
        install_requires=get_install_requirements("requirements.txt"),
        tests_require=get_install_requirements("test-requirements.txt"),
        packages=find_packages(exclude=["tests*"]),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        use_scm_version=True,
        long_description=open("README.md", "r").read(),
        long_description_content_type="text/markdown",
        python_requires=">=3.6",
        entry_points={"console_scripts": ["kartothek_cube=kartothek.cli:cli"]},
    )


if __name__ == "__main__":
    setup_package()
