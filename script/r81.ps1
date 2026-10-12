param(
    [switch]$Probe,
    [switch]$Survey,
    [switch]$Loop,
    [switch]$CheckOnly,
    [string]$Report = ''
)
$ErrorActionPreference = 'Stop'
if (@(@($Probe, $Survey, $Loop) | Where-Object { $_ }).Count -gt 1) {
    throw 'Choose one of Probe, Survey or Loop.'
}
$taskDefaultsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\r81_worker_current.json'
$taskDefaults = Get-Content -LiteralPath $taskDefaultsPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskExpectedFields = @('release', 'release_sha256', 'stage')
$taskFields = @($taskDefaults.PSObject.Properties.Name)
if ($taskFields.Count -ne 3 -or @(Compare-Object $taskExpectedFields $taskFields).Count -ne 0) {
    throw 'R81 worker preset fields are invalid.'
}
if ($Loop) {
    $taskLoopPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\r81_s21_live_current.json'
    $taskLoop = Get-Content -LiteralPath $taskLoopPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskLoopFields = @($taskLoop.PSObject.Properties.Name)
    if ($taskLoopFields.Count -ne 2 -or
        @(Compare-Object @('bootstrap_name', 'bootstrap_sha256') $taskLoopFields).Count -ne 0 -or
        [string]$taskLoop.bootstrap_name -notmatch '^r81_[A-Za-z0-9_.-]+\.json$' -or
        [string]$taskLoop.bootstrap_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'R81 S21 live-loop preset fields are invalid.'
    }
    $taskSourceRoot = Split-Path -Parent (Split-Path -Parent ([string]$taskDefaults.release))
    $taskLoopRoot = Join-Path (Split-Path -Parent $taskSourceRoot) 'survey_loop_v001\release'
    $taskMachine = python -c "from antenna.utils.web import get_local_ip; print(get_local_ip())"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify this worker.' }
    $taskLoopArguments = @{
        Bootstrap = Join-Path $taskLoopRoot ([string]$taskLoop.bootstrap_name)
        BootstrapSha256 = [string]$taskLoop.bootstrap_sha256
        ExpectedMachine = $taskMachine.Trim()
        CheckOnly = $CheckOnly
        Report = $Report
    }
    & (Join-Path $PSScriptRoot 'start_r81_s21_live_worker.ps1') @taskLoopArguments
    return
}
if ($Survey) {
    $taskSurveyPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\r81_s21_survey_current.json'
    $taskSurvey = Get-Content -LiteralPath $taskSurveyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskSurveyFields = @($taskSurvey.PSObject.Properties.Name)
    if ($taskSurveyFields.Count -ne 2 -or
        @(Compare-Object @('release_name', 'release_sha256') $taskSurveyFields).Count -ne 0 -or
        [string]$taskSurvey.release_name -notmatch '^r81_[A-Za-z0-9_.-]+\.json$') {
        throw 'R81 S21 survey preset fields are invalid.'
    }
    $taskSourceRoot = Split-Path -Parent (Split-Path -Parent ([string]$taskDefaults.release))
    $taskSurveyRoot = Join-Path (Split-Path -Parent $taskSourceRoot) 'survey_v001\release'
    $taskMachine = python -c "from antenna.utils.web import get_local_ip; print(get_local_ip())"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify this worker.' }
    $taskSurveyArguments = @{
        Release = Join-Path $taskSurveyRoot ([string]$taskSurvey.release_name)
        ReleaseSha256 = [string]$taskSurvey.release_sha256
        ExpectedMachine = $taskMachine.Trim()
        CheckOnly = $CheckOnly
        Report = $Report
    }
    & (Join-Path $PSScriptRoot 'start_r81_provisional_filter_worker.ps1') @taskSurveyArguments
    return
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
