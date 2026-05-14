param(
    [string]$CondaEnvName = "STDR",
    [string]$PythonExe = "python",
    [string]$RepoRoot = "",
    [string]$CampaignRoot = "",
    [string]$DataDir = "data",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $CampaignRoot = Join-Path $RepoRoot ("runs\local_4090_dc\autopilot_sessions\DC_autopilot_fast_from_speed_" + $stamp)
}

$runArgs = @{
    RepoRoot = $RepoRoot
    CampaignRoot = $CampaignRoot
    CondaEnvName = $CondaEnvName
    PythonExe = $PythonExe
    DataDir = $DataDir
    BatchSize = 32
    Device = $Device
    Precision = $Precision
    NumWorkers = $NumWorkers
    PredHorizon = 4
    ReportHorizonsMinutes = "15,30,60"
    SplitPolicy = "project_monthly"
    SplitAlignment = "none"
    TopKs = "16,20"
    HistLens = "12,14"
    LearningRates = "7e-4,1e-3"
    WeightDecays = "1e-5"
    HiddenDims = "32"
    NumStBlocks = "2"
    TimeFeatureModes = "baseline"
    FinalSeeds = "19,42"
    SmokeEpochs = 2
    Stage1Epochs = 8
    RefineEpochs = 12
    FinalEpochs = 20
    Stage1Keep = 2
    Stage2Keep = 2
    Stage3Keep = 2
    Stage4Keep = 2
    FinalConfigCount = 2
    Stage1ValInterval = 2
    RefineValInterval = 1
    FinalValInterval = 1
}

if ($ResumeExisting.IsPresent) {
    $runArgs.ResumeExisting = $true
}

& (Join-Path $RepoRoot "local_4090\run_dc_autopilot.ps1") @runArgs
