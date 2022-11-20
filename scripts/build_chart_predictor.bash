#bin/bash

set -e

export HELM_CHART_DIRECTORY=../charts/predictor
mkdir -p bin && cd bin && helm lint --strict ${HELM_CHART_DIRECTORY} && helm package ${HELM_CHART_DIRECTORY}
