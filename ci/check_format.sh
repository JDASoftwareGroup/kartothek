#!/bin/bash -ex
pip install \
    black==18.9b0 \
    isort==4.3.4 \
    flake8==3.6.0

pip install flake8==3.6.0 flake8-mutable==1.2.0

flake8 \
    --output-file flake8_report.txt \
    --count \
    --tee

black --check .

isort \
    --multi-line=3 \
    --trailing-comma \
    --line-width=88 \
    -p kartothek \
    -sd THIRDPARTY \
    --check-only \
    --skip .eggs \
    --recursive \
    .