#!/usr/bin/env bash

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT}/environment.yml"
ENV_NAME="visor-cu118"

echo "ViSOR setup starting"

if command -v micromamba >/dev/null 2>&1; then
  TOOL="micromamba"
elif command -v conda >/dev/null 2>&1; then
  TOOL="conda"
else
  echo "No supported environment manager found."
  echo "Install micromamba or conda first, then rerun this script."
  exit 1
fi

if ! "${TOOL}" env list | awk '{print $1}' | tr -d '*' | grep -qx "${ENV_NAME}"; then
  echo "Creating ${ENV_NAME} from ${ENV_FILE}"
  "${TOOL}" env create -f "${ENV_FILE}"
else
  echo "Environment ${ENV_NAME} already exists"
fi

set +u
eval "$("${TOOL}" shell hook -s bash)"
"${TOOL}" activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
cd "${ROOT}"
pip install -e .

cat <<EOF

ViSOR is ready.

Later sessions:

  ${TOOL} activate ${ENV_NAME}
  export PYTHONNOUSERSITE=1

Common commands:

  bash ./run_train.sh
  bash ./watch_train.sh
  bash ./view_latest.sh

Optional native tiny-cuda-nn install:

  pip install ninja cmake
  pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.7#subdirectory=bindings/torch"
EOF
