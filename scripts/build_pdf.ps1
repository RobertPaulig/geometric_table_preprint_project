$ErrorActionPreference = "Stop"

$tex = "geometric_table_snowflakes_preprint.tex"

lualatex -interaction=nonstopmode $tex
lualatex -interaction=nonstopmode $tex
