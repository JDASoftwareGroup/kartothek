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
        --index-url https://software.blue-yonder.org/${productionIndex}/${nodeName:-Debian_9}/+simple \
        -o requirements-pinned.txt \
        kartothek_env_reqs.txt \
        requirements.in
else
    pip-compile \
        --upgrade \
        --no-index \
        --index-url https://software.blue-yonder.org/${productionIndex}/${nodeName:-Debian_9}/+simple \
        -o requirements-pinned.txt \
        requirements.in
fi

pip-compile \
    --upgrade \
    --no-index \
    --index-url https://software.blue-yonder.org/${developmentIndex}/${nodeName:-Debian_9}/+simple \
    -o test-requirements-pinned.txt \
    requirements-pinned.txt \
    test-requirements.in
