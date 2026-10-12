param(
    [Parameter(Mandatory = $true)] [string]$Bootstrap,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')] [string]$BootstrapSha256,
    [Parameter(Mandatory = $true)]
    [ValidateSet('140.123.106.216', '140.123.106.218', '140.123.106.37')]
    [string]$ExpectedMachine,
    [switch]$CheckOnly,
    [ValidateRange(15, 3600)] [int]$PollSeconds = 60,
    [string]$Report = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Assert-WorkerIdentity {
    $localIp = python -c "from antenna.utils.web import get_local_ip; print(get_local_ip())"
    if ($LASTEXITCODE -ne 0 -or $localIp.Trim() -ne $ExpectedMachine) {
        throw "This host is not the released worker $ExpectedMachine."
    }
}

function Assert-NoR81Conflict {
    # Read-only guard. Never stop, suspend, or signal a process owned by anyone.
    $conflicts = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and (
            ($_.CommandLine -match 'script[.]dedust.+\bworker\b') -or
            ($_.CommandLine -match 'start_r81_(worker|sweep_probe_worker|formal_sharded_worker|provisional_filter_worker|s21_live_worker)[.]ps1')
        )
    })
    if ($conflicts.Count -ne 0) {
        throw 'Another R81/native dedust worker or wrapper is active on this host; preserve it and do not double-dispatch.'
    }
}

function Read-LiveStatus {
    $arguments = @('-m', 'script.r81_s21_live_entry', '--bootstrap', $Bootstrap,
                   '--bootstrap-sha256', $BootstrapSha256, '--machine', $ExpectedMachine)
    if ($Report) { $arguments += @('--report', $Report) }
    $text = python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'R81 S21 live bootstrap/registry/release preflight failed.' }
    return ($text | Out-String | ConvertFrom-Json)
}

Assert-WorkerIdentity
Assert-NoR81Conflict
$status = Read-LiveStatus
if ($CheckOnly) { return }

while ($true) {
    if ([string]$status.status -eq 'stopped') { return }
    if ([string]$status.status -eq 'budget_complete') { return }
    if ([string]$status.status -eq 'terminal_failure') {
        throw "R81 S21 batch $($status.sequence) reached a same-host terminal failure; preserve every marker and store."
    }
    if ([string]$status.status -eq 'waiting_for_publication') {
        Start-Sleep -Seconds $PollSeconds
        $status = Read-LiveStatus
        continue
    }
    if ([string]$status.status -ne 'ready') {
        throw "Unknown R81 S21 live worker status: $($status.status)"
    }
    $datasetRoot = [string]$status.dataset_root
    if ((Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP')) -or
        (Test-Path -LiteralPath (Join-Path $datasetRoot 'jobs_state\STOP_symmetry_filter_20261007'))) {
        return
    }
    # Recheck immediately before each native invocation. This loop launches at
    # most one --once child, then authenticates the registry and own marker again.
    Assert-NoR81Conflict
    python -m script.dedust --dataset-root $datasetRoot worker `
        --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120 --once
    if ($LASTEXITCODE -ne 0) {
        throw 'R81 S21 live worker stopped with an error; preserve results and claims for review.'
    }
    Start-Sleep -Seconds $PollSeconds
    $status = Read-LiveStatus
}
