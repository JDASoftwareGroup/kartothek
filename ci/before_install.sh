#!/bin/bash

set -xueo pipefail

# https://github.com/JDASoftwareGroup/kartothek/issues/94
pip install --upgrade pip==19.1.*
pip install pip-tools
./ci/compile_requirements.sh
python setup.py bdist_wheel
