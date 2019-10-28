#!/bin/bash
set -xueo pipefail

if [[ "${ARROW_NIGHTLY-0}" == 1 ]]; then
    ARROW_URL="$(./ci/get_pyarrow_nightly.py)"
    ARROW_FILE="$(basename $ARROW_URL)"

    # we need to download the package to make it usable for pip
    PACKAGE_PATH="$(mktemp -d)"
    ARROW_PATH="$PACKAGE_PATH/$ARROW_FILE"
    wget -O "$ARROW_PATH" "$ARROW_URL"

    pip install $ARROW_PATH
fi

if [[ "${NUMFOCUS_NIGHTLY-0}" == 1 ]]; then
    # NumFOCUS nightly wheels, contains numpy and pandas
   PRE_WHEELS="https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
   pip install --pre --upgrade --timeout=60 -f $PRE_WHEELS pandas numpy
fi
