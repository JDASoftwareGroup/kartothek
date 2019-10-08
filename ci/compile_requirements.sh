#!/bin/bash
set -xueo pipefail

if [ -z ${PIP_COMPILE_ARGS+x} ]; then
    PIP_COMPILE_ARGS=""
fi

if [[ "${ARROW_NIGHTLY-0}" == 1 ]]; then
    ARROW_URL="$(./ci/get_pyarrow_nightly.py)"
    ARROW_FILE="$(basename $ARROW_URL)"
    ARROW_VERSION="$(echo "$ARROW_FILE" | awk '{split($1,a,"-"); print a[2]}')"

    # we need to download the package to make it usable for pip
    PACKAGE_PATH="$(mktemp -d)"
    ARROW_PATH="$PACKAGE_PATH/$ARROW_FILE"
    wget -O "$ARROW_PATH" "$ARROW_URL"

    PIP_COMPILE_ARGS="$PIP_COMPILE_ARGS -P pyarrow==$ARROW_VERSION -f file://$PACKAGE_PATH"

    # we limit the upper version of pyarrow by default, so we need to remove this
    grep --invert-match pyarrow requirements.txt > requirements.txt.tmp
    mv requirements.txt.tmp requirements.txt
fi

if [[ "${NUMFOCUS_NIGHTLY-0}" == 1 ]]; then
    # NumFOCUS nightly wheels, contains numpy and pandas
    PIP_COMPILE_ARGS="$PIP_COMPILE_ARGS --pre -f https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
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
