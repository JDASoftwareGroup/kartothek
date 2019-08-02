#!/bin/bash

echo "getting pyarrow latest build URL"

PYARROW_NIGHTLY=`python get_pyarrow_nightly.py`

echo " $PYARROW_NIGHTLY"

if [ ! -z ${PYARROW_NIGHTLY} ] && [! ${PYARROW_NIGHTLY} = "NULL" ];
then
    echo "Upgrading to Nightly build of pyarrow"

    pip install --pre --no-deps --upgrade --timeout=180 --no-cache-dir -f "$PYARROW_NIGHTLY" pyarrow


else
    echo "Nightly Build is not present. Failing the Travis Job"
    exit 1
fi
