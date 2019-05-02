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
        install_requires=get_install_requirements("requirements.in"),
        tests_require=get_install_requirements("test-requirements.in"),
        packages=find_packages(exclude=["tests"]),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Programming Language :: Python",
        ],
        use_scm_version=True,
    )


if __name__ == "__main__":
    setup_package()
