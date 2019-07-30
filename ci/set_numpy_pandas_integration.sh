#!/bin/bash

set -xeo pipefail

pip-compile ./ci/numpy_pandas_requirements.txt -o ./ci/numpy_pandas_req_pinned.txt

pip install -r ./ci/numpy_pandas_req_pinned.txt

echo "Upgrading to  Nightly build of numpy pandas"

PRE_WHEELS="https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"

pip install --pre --no-deps --upgrade --timeout=180 --no-cache-dir -f $PRE_WHEELS numpy pandas

pip freeze

rm -f ./ci/numpy_pandas_req_pinned.txt