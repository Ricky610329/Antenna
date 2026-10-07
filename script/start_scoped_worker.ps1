param(
    [string]$DatasetRoot = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $DatasetRoot) {
    $taskRootJson = python -c "import json; from antenna.utils import ROOTDIR; print(json.dumps(str(ROOTDIR.joinpath('experiments', 'r80_symmetry_20261007', 'dataset'))))"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the personal Antenna workspace.' }
    $DatasetRoot = $taskRootJson | ConvertFrom-Json
}
if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot 'jobs.json'))) {
    throw 'DatasetRoot must contain the prepared shared jobs.json.'
}
# This launcher is restricted to the current single-port symmetry phase.
$taskJobs = Get-Content -LiteralPath (Join-Path $DatasetRoot 'jobs.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($taskJob in $taskJobs) {
    if ($taskJob.scope -eq 'symmetry_filter_20261007') {
        $taskMeasurementPath = Join-Path (Join-Path $DatasetRoot $taskJob.input) 'measurement.json'
        $taskMeasurement = Get-Content -LiteralPath $taskMeasurementPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskMeasurement.port -ne 'single') { throw 'R80 launcher refuses filter jobs; filter work is deferred.' }
    }
}
$env:OMP_NUM_THREADS = '4'
python -m script.dedust --dataset-root $DatasetRoot jobs-ls --scope symmetry_filter_20261007 --all
if ($LASTEXITCODE -ne 0) { throw 'Queue inspection failed.' }
if ($CheckOnly) { return }
python -m script.dedust --dataset-root $DatasetRoot worker --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120
if ($LASTEXITCODE -ne 0) { throw 'Worker stopped with an error; preserve its results and claims for review.' }
