#!/bin/bash
set -xeo pipefail

productionIndex=platform
developmentIndex=platform_dev

PIP_COMPILE_ARGS=""
KARTOTHEK_NP_PD_NIGHTLY=1
KARTOTHEK_ARROW_VERSION=0.12.1

if [ ! -z "${KARTOTHEK_NP_PD_NIGHTLY}" ]  && [ "${KARTOTHEK_NP_PD_NIGHTLY}" == 1 ];
then
    PIP_COMPILE_OPTIONS=(
    "--pre"
    "-f https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
     )
    for opt in "${PIP_COMPILE_OPTIONS[@]}";
    do
        PIP_COMPILE_ARGS=$PIP_COMPILE_ARGS" "$opt
    done
fi

if [ ! -z ${KARTOTHEK_ARROW_VERSION} ] && [ ! ${KARTOTHEK_ARROW_VERSION} = "NIGHTLY" ];
then
      echo pyarrow==$KARTOTHEK_ARROW_VERSION > kartothek_env_reqs.txt
      #trap 'rm -f kartothek_env_reqs.txt' EXIT
      pip-compile \
        --upgrade \
        --no-index \
          -o requirements-pinned.txt \
          kartothek_env_reqs.txt \
          requirements.txt
      if [ ! -z "${KARTOTHEK_NP_PD_NIGHTLY}" ]  && [ "${KARTOTHEK_NP_PD_NIGHTLY}" == 1 ];
      then      pip-compile "$PIP_COMPILE_ARGS"
      fi
elif [ ! -z "${KARTOTHEK_ARROW_VERSION}" ] && [ ${KARTOTHEK_ARROW_VERSION} == "NIGHTLY" ];
then
    pyarrow_url=`python get_pyarrow_nightly.py`
    KARTOTHEK_ARROW_VERSION=$(python -c "print('$pyarrow_url'.split('/')[-1].split('-')[1])")
    echo pyarrow==$KARTOTHEK_ARROW_VERSION > kartothek_env_reqs.txt
    trap 'rm -f kartothek_env_reqs.txt' EXIT
    pip-compile \
        --upgrade \
        --no-index \
        -o requirements-pinned.txt \
        --pre \
        -f $pyarrow_url \
        kartothek_env_reqs.txt \
        requirements.txt\
else
    pip-compile \
        --upgrade \
        --no-index \
        -o requirements-pinned.txt \
        requirements.txt
fi

