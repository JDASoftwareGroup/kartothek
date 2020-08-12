#!/usr/bin/env bash

# Note: this assumes you have kartothek installed in your current environment and you are using conda

PYARROW_VERSIONS="0.14.1 0.15.0 0.16.0 1.0.0"

for pyarrow_version in $PYARROW_VERSIONS; do
    echo $pyarrow_version 
    conda install -y pyarrow==$pyarrow_version
    ./generate_reference.py || (echo "Failed for version $pyarrow_version"; exit 1)
done
