#!/bin/bash
set -xueo pipefail

if [ -z ${PIP_COMPILE_ARGS+x} ]; then
    PIP_COMPILE_ARGS=""
fi

if [[ "${ARROW_NIGHTLY+1}" == 1 ]]; then
    ARROW_URL="$(./ci/get_pyarrow_nightly.py)"
    ARROW_FILE="$(basename $ARROW_URL)"
    ARROW_VERSION="$(echo "$ARROW_FILE" | awk '{split($1,a,"-"); print a[2]}')"

    # we need to download the package to make it usable for pip
    PACKAGE_PATH="$(mktemp -d)"
    ARROW_PATH="$PACKAGE_PATH/$ARROW_FILE"
    wget -O "$ARROW_PATH" "$ARROW_URL"

    PIP_COMPILE_ARGS="$PIP_COMPILE_ARGS -P pyarrow==$ARROW_VERSION -f file://$PACKAGE_PATH"
fi

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
