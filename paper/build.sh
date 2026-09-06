#!/usr/bin/env bash

set -euo pipefail

export SOURCE_DATE_EPOCH=1788048000
export TZ=UTC

cd "$(dirname "$0")"
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
