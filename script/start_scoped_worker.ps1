param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot 'jobs.json'))) {
    throw 'DatasetRoot must contain the prepared shared jobs.json.'
}
$env:OMP_NUM_THREADS = '4'
python -m script.dedust --dataset-root $DatasetRoot jobs-ls --scope symmetry_filter_20261007 --all
if ($LASTEXITCODE -ne 0) { throw 'Queue inspection failed.' }
python -m script.dedust --dataset-root $DatasetRoot worker --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120
if ($LASTEXITCODE -ne 0) { throw 'Worker stopped with an error; preserve its results and claims for review.' }
