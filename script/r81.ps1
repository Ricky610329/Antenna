param(
    [switch]$Probe,
    [switch]$CheckOnly,
    [string]$Report = ''
)
$ErrorActionPreference = 'Stop'
$taskDefaultsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\r81_worker_current.json'
$taskDefaults = Get-Content -LiteralPath $taskDefaultsPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskExpectedFields = @('release', 'release_sha256', 'stage')
$taskFields = @($taskDefaults.PSObject.Properties.Name)
if ($taskFields.Count -ne 3 -or @(Compare-Object $taskExpectedFields $taskFields).Count -ne 0) {
    throw 'R81 worker preset fields are invalid.'
}
if ($Probe) {
    $taskProbePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\r81_sweep_probe_current.json'
    $taskProbe = Get-Content -LiteralPath $taskProbePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskProbeFields = @($taskProbe.PSObject.Properties.Name)
    if ($taskProbeFields.Count -ne 2 -or
        @(Compare-Object @('release_name', 'release_sha256') $taskProbeFields).Count -ne 0 -or
        [string]$taskProbe.release_name -notmatch '^r81_[A-Za-z0-9_.-]+\.json$') {
        throw 'R81 sweep probe preset fields are invalid.'
    }
    $taskSourceRoot = Split-Path -Parent (Split-Path -Parent ([string]$taskDefaults.release))
    $taskProbeRoot = Join-Path (Split-Path -Parent $taskSourceRoot) 'sweep_probe_v001\release'
    $taskProbeArguments = @{
        Release = Join-Path $taskProbeRoot ([string]$taskProbe.release_name)
        ReleaseSha256 = [string]$taskProbe.release_sha256
        CheckOnly = $CheckOnly
        Report = $Report
    }
    & (Join-Path $PSScriptRoot 'start_r81_sweep_probe_worker.ps1') @taskProbeArguments
    return
}
$taskArguments = @{
    Release = [string]$taskDefaults.release
    ReleaseSha256 = [string]$taskDefaults.release_sha256
    Stage = [string]$taskDefaults.stage
    CheckOnly = $CheckOnly
    Report = $Report
}
if ($taskArguments.Stage -eq 'FormalSharded') {
    $taskArguments.Remove('Stage')
    & (Join-Path $PSScriptRoot 'start_r81_formal_sharded_worker.ps1') @taskArguments
} elseif ($taskArguments.Stage -in @('EngineeringOnly', 'Formal')) {
    & (Join-Path $PSScriptRoot 'start_r81_worker.ps1') @taskArguments
} else {
    throw 'R81 worker preset stage is invalid.'
}
