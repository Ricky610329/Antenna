param(
    [Parameter(Mandatory = $true)] [string]$Release,
    [Parameter(Mandatory = $true)] [ValidatePattern('^[0-9a-f]{64}$')] [string]$ReleaseSha256,
    [Parameter(Mandatory = $true)]
    [ValidateSet('140.123.106.216', '140.123.106.218', '140.123.106.37')]
    [string]$ExpectedMachine,
    [switch]$CheckOnly,
    [string]$Report = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$entry = @('-m', 'script.r81_provisional_filter_dispatch', '--release', $Release,
           '--release-sha256', $ReleaseSha256)
if ($Report) { $entry += @('--report', $Report) }
python @entry
if ($LASTEXITCODE -ne 0) { throw 'R81 provisional S21 immutable release preflight failed.' }

$releaseDoc = Get-Content -LiteralPath $Release -Raw -Encoding UTF8 | ConvertFrom-Json
$job = @($releaseDoc.queue.jobs | Where-Object { $_.job.machine -eq $ExpectedMachine })
if ($job.Count -ne 1) { throw 'ExpectedMachine must own exactly one released provisional shard.' }

# A native worker identifies itself by its local IP and honors the job.machine
# field.  Check the same identity here before any worker or HFSS process starts.
$localIp = python -c "from antenna.utils.web import get_local_ip; print(get_local_ip())"
if ($LASTEXITCODE -ne 0 -or $localIp.Trim() -ne $ExpectedMachine) {
    throw "This host is not the released worker $ExpectedMachine."
}
if ($CheckOnly) { return }

# Do not overlap the original Engineering, SweepProbe, or another provisional
# wrapper/worker on this host.  This is a read-only process check and never
# stops any process.  A later operator must end the old wrapper normally first.
$conflicts = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and (
        ($_.CommandLine -match 'script[.]dedust.+\bworker\b') -or
        ($_.CommandLine -match 'start_r81_worker[.]ps1') -or
        ($_.CommandLine -match 'start_r81_sweep_probe_worker[.]ps1') -or
        ($_.CommandLine -match 'start_r81_formal_sharded_worker[.]ps1') -or
        ($_.CommandLine -match 'start_r81_provisional_filter_worker[.]ps1')
    )
})
if ($conflicts.Count -ne 0) {
    throw 'Another R81/native dedust worker or wrapper is active on this host; preserve it and do not double-dispatch.'
}

$datasetRoot = [string]$releaseDoc.dataset_root
while ($true) {
    if ((Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP')) -or
        (Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP_symmetry_filter_20261007'))) {
        return
    }
    python -m script.r81_provisional_filter_dispatch --release $Release `
        --release-sha256 $ReleaseSha256 --queue-status
    $queueStatus = $LASTEXITCODE
    if ($queueStatus -eq 0) { return }
    if ($queueStatus -eq 4) {
        throw 'R81 provisional S21 queue reached a terminal failure; preserve every marker and store.'
    }
    if ($queueStatus -ne 3) { throw 'R81 provisional release or queue-state validation failed.' }
    python -m script.dedust --dataset-root $datasetRoot worker `
        --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120 --once
    if ($LASTEXITCODE -ne 0) {
        throw 'R81 provisional worker stopped with an error; preserve results and claims for review.'
    }
    Start-Sleep -Seconds 60
}
