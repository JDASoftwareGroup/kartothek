#!/bin/bash
set -xeo pipefail

if [ -z ${KARTOTHEK_NP_PD_NIGHTLY} ]; then
  KARTOTHEK_NP_PD_NIGHTLY=0
fi

if [ "${KARTOTHEK_NP_PD_NIGHTLY}" == 1 ];
then
  echo " KARTOTHEK_NP_PD_NIGHTLY Value--->  $KARTOTHEK_NP_PD_NIGHTLY"
   PIP_COMPILE_OPTIONS=(
    "--pre"
    "-f https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
    )

    PIP_COMPILE_ARGS=""
    for opt in "${PIP_COMPILE_OPTIONS[@]}";
    do
        PIP_COMPILE_ARGS=$PIP_COMPILE_ARGS" "$opt
    done
    trap 'rm -f test-requirements-pinned.txt' EXIT
    pip-compile\
        $PIP_COMPILE_ARGS \
        -o requirements-pinned.txt \
        requirements.txt

elif [ ${KARTOTHEK_ARROW_VERSION} == "NIGHTLY" ] ;
then
    echo " KARTOTHEK_ARROW_VERSION Value--->  $KARTOTHEK_ARROW_VERSION"
    pyarrow_url=$(python ci/get_pyarrow_nightly.py)
    echo "pyarrow URl --> $pyarrow_url"
    KARTOTHEK_ARROW_VERSION=$(python -c "print('$pyarrow_url'.split('/')[-1].split('-')[1])")
    echo pyarrow=="$KARTOTHEK_ARROW_VERSION" > kartothek_env_reqs.txt
    trap 'rm -f kartothek_env_reqs.txt requirements-pinned.txt' EXIT
    pip-compile\
        --pre \
        -f "$pyarrow_url" \
        -o requirements-pinned.txt \
        kartothek_env_reqs.txt \
        requirements.txt\

elif [ ! "${KARTOTHEK_ARROW_VERSION}" = "NIGHTLY" ];
then
    echo " KARTOTHEK_ARROW_VERSION Value--->  $KARTOTHEK_ARROW_VERSION"
    echo pyarrow=="$KARTOTHEK_ARROW_VERSION" > kartothek_env_reqs.txt
    trap 'rm -f kartothek_env_reqs.txt requirements-pinned.txt' EXIT
    pip-compile \
        --upgrade \
        --no-index \
        -o requirements-pinned.txt \
        kartothek_env_reqs.txt \
        requirements.txt

else
    echo " KARTOTHEK_ARROW_VERSION Value--->  $KARTOTHEK_ARROW_VERSION"
    trap 'rm -f requirements-pinned.txt' EXIT
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