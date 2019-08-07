#!/bin/bash
set -xeo pipefail

pip-compile \
    -v \
    $PIP_COMPILE_ARGS \
    -o requirements-pinned.txt \
    requirements.txt

pip-compile \
    -v \
    $PIP_COMPILE_ARGS \
    -o test-requirements-pinned.txt \
    requirements-pinned.txt \
    test-requirements.txt