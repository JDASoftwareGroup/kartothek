#!/bin/bash
set -xeo pipefail

pip-compile \
    $PIP_COMPILE_ARGS \
    -o requirements-pinned.txt \
    requirements.txt

pip-compile \
    $PIP_COMPILE_ARGS \
    -o test-requirements-pinned.txt \
    requirements-pinned.txt \
    test-requirements.txt