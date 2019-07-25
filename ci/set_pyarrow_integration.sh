#!/bin/bash

set -xeo pipefail

echo "Upgrading to Nightly build of pyarrow"

pip install --pre --no-deps --upgrade --timeout=180 --no-cache-dir -f "https://github.com/ursa-labs/crossbow/releases/download/latest" pyarrow

pip freeze
