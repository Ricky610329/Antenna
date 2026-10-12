param(
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
