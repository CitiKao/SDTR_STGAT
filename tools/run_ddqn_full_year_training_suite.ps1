$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TERM = "xterm-256color"

$Repo = "D:\STDR"
$PythonExe = "C:\Users\WMNL1691\Anaconda3\envs\stable_diffusion\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo "runs\ddqn_full_year_training_suite_$Stamp"
$Log = Join-Path $RunDir "terminal.log"
$Summary = Join-Path $RunDir "full_year_training_suite_summary.json"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Content -Path (Join-Path $Repo "runs\latest_ddqn_full_year_training_suite_run.txt") -Value $RunDir

Write-Host "======================================================================================================================" -ForegroundColor Cyan
Write-Host "Full-year supervised reranker + pure DDQN 200k + warm-start DDQN 200k started" -ForegroundColor Cyan
Write-Host "RunDir:  $RunDir" -ForegroundColor Yellow
Write-Host "Log:     $Log" -ForegroundColor Yellow
Write-Host "Summary: $Summary" -ForegroundColor Yellow
Write-Host "======================================================================================================================" -ForegroundColor Cyan

Set-Location $Repo

$PythonArgs = @(
  "-u",
  "D:\STDR\ddqn_full_year_training_suite.py",
  "--train-config", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\config.json",
  "--run-dir", $RunDir,
  "--device", "cuda",
  "--seed", "231",
  "--profile-batch-size", "24",
  "--record-log-interval", "5000",
  "--min-unique-routes", "1",
  "--min-pred-hops", "0",
  "--min-pred-distance-km", "0",
  "--benefit-delta", "0.005",
  "--bad-delta", "0.05",
  "--minor-bad-delta", "0.01",
  "--ddqn-bad-penalty", "2.5",
  "--ddqn-minor-bad-penalty", "0.25",
  "--sample-weight-mode", "sqrt_count",
  "--supervised-epochs", "20",
  "--supervised-batch-size", "1024",
  "--supervised-log-interval", "100",
  "--supervised-positive-weight", "8.0",
  "--supervised-reward-loss-weight", "0.15",
  "--pure-ddqn-episodes", "200000",
  "--warm-ddqn-episodes", "200000",
  "--ddqn-batch-size", "256",
  "--ddqn-log-interval", "5000",
  "--epsilon-start", "1.0",
  "--epsilon-end", "0.05",
  "--epsilon-decay", "80000",
  "--target-update", "5000",
  "--eval-batch-size", "8192"
)

& $PythonExe @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
  Write-Host "======================================================================================================================" -ForegroundColor Green
  Write-Host "Full-year training suite finished successfully." -ForegroundColor Green
  Write-Host "Summary: $Summary" -ForegroundColor Green
  Write-Host "======================================================================================================================" -ForegroundColor Green
} else {
  Write-Host "======================================================================================================================" -ForegroundColor Red
  Write-Host "Full-year training suite failed with exit code $ExitCode." -ForegroundColor Red
  Write-Host "Check log: $Log" -ForegroundColor Red
  Write-Host "======================================================================================================================" -ForegroundColor Red
}
