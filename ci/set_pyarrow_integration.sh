#!/bin/bash

set -xeo pipefail

pip-compile ./ci/pyarrow_requirements.txt -o ./ci/pyarrow_req_pinned.txt

pip install -r ./ci/pyarrow_req_pinned.txt

echo "Upgrading to Nightly build of pyarrow"

pip install --pre --no-deps --upgrade --timeout=180 --no-cache-dir -f "https://github.com/ursa-labs/crossbow/releases/download/latest" pyarrow

pip freeze

rm -f ./ci/pyarrow_req_pinned.txt


