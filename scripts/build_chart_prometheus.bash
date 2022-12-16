#bin/bash

set -e

export HELM_CHART_DIRECTORY=../charts/prometheus

mkdir -p bin && cd bin && helm dependency update ${HELM_CHART_DIRECTORY} && helm dependency build ${HELM_CHART_DIRECTORY} && helm lint --strict ${HELM_CHART_DIRECTORY} && helm package ${HELM_CHART_DIRECTORY}
