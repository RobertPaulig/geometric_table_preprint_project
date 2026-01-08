param(
  [switch]$PdfOnly
)

$ErrorActionPreference = "Stop"

if (-not $PdfOnly) {
  $env:PYTHONPATH = "code"
  python code\scripts\wave_atlas_generate.py
  python code\scripts\wave_metrics.py
}

Push-Location docs
lualatex wave_atlas.tex
lualatex wave_atlas.tex
Pop-Location

$pdfPath = "docs\wave_atlas.pdf"
Write-Host "PDF: $pdfPath"

$hash = Get-FileHash -Algorithm SHA256 $pdfPath
Write-Host "$($hash.Hash)  $pdfPath"
