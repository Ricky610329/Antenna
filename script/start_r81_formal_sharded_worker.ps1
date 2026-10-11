param(
    [Parameter(Mandatory = $true)] [string]$Release,
    [Parameter(Mandatory = $true)] [ValidatePattern('^[0-9a-f]{64}$')] [string]$ReleaseSha256,
    [switch]$CheckOnly,
    [string]$Report = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$entry = @('-m', 'script.r81_formal_sharded_entry', '--release', $Release,
           '--release-sha256', $ReleaseSha256)
if ($Report) { $entry += @('--report', $Report) }
python @entry
if ($LASTEXITCODE -ne 0) { throw 'R81 formal-sharded immutable release preflight failed.' }
if ($CheckOnly) { return }

$releaseDoc = Get-Content -LiteralPath $Release -Raw -Encoding UTF8 | ConvertFrom-Json
$datasetRoot = [string]$releaseDoc.dataset_root
while ($true) {
    if ((Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP')) -or
        (Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP_symmetry_filter_20261007'))) {
        return
    }
    python -m script.r81_formal_sharded_entry --release $Release `
        --release-sha256 $ReleaseSha256 --queue-status
    $queueStatus = $LASTEXITCODE
    if ($queueStatus -eq 0) { return }
    if ($queueStatus -eq 4) {
        throw 'R81 formal-sharded queue exhausted the exact authorized roster; preserve every marker and store.'
    }
    if ($queueStatus -ne 3) { throw 'R81 formal-sharded release or semantic queue-state validation failed.' }
    python -m script.dedust --dataset-root $datasetRoot worker `
        --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120 --once
    if ($LASTEXITCODE -ne 0) {
        throw 'R81 formal-sharded worker stopped with an error; preserve results and claims for review.'
    }
    Start-Sleep -Seconds 60
}
