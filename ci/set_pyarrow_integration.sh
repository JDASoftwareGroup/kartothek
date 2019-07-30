#!/bin/bash

set -xeo pipefail

pip-compile ./ci/pyarrow_requirements.txt

pip install -r ./ci/integration_requirements.txt

echo "Upgrading to Nightly build of pyarrow"

pip install --pre --no-deps --upgrade --timeout=180 --no-cache-dir -f "https://github.com/ursa-labs/crossbow/releases/download/latest" pyarrow

pip freeze
