#!/bin/bash
set -xeo pipefail

productionIndex=platform
developmentIndex=platform_dev

if [ ! -z ${KARTOTHEK_ARROW_VERSION} ];
then
    echo pyarrow==$KARTOTHEK_ARROW_VERSION > kartothek_env_reqs.txt
    trap 'rm -f kartothek_env_reqs.txt' EXIT
    pip-compile \
        --upgrade \
        --no-index \
        -o requirements-pinned.txt \
        kartothek_env_reqs.txt \
        requirements.txt
else
    pip-compile \
        --upgrade \
        --no-index \
        -o requirements-pinned.txt \
        requirements.txt
fi

pip-compile \
    --upgrade \
    --no-index \
    -o test-requirements-pinned.txt \
    requirements-pinned.txt \
    test-requirements.txt
