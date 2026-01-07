#!/usr/bin/env bash
set -euo pipefail

TEX="geometric_table_snowflakes_preprint.tex"

lualatex -interaction=nonstopmode "$TEX"
lualatex -interaction=nonstopmode "$TEX"
