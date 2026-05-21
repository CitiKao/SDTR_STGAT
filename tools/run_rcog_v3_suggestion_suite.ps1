$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TERM = "xterm-256color"

$Repo = "D:\STDR"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo "runs\rcog_v3_suggestion_suite_$Stamp"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Log = Join-Path $RunDir "terminal.log"
$Summary = Join-Path $RunDir "rcog_v3_suggestion_suite_summary.json"

Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host "RCOG v3 suggestion suite started" -ForegroundColor Cyan
Write-Host "RunDir: $RunDir" -ForegroundColor Yellow
Write-Host "Log:    $Log" -ForegroundColor Yellow
Write-Host "Summary will be: $Summary" -ForegroundColor Yellow
Write-Host "========================================================================================================" -ForegroundColor Cyan

Set-Location $Repo
$PythonExe = "C:\Users\WMNL1691\Anaconda3\envs\stable_diffusion\python.exe"

$PythonArgs = @(
  "D:\STDR\rcog_v3_suggestion_suite.py",
  "--checkpoint", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\ddqn_reranker_final.pt",
  "--train-config", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\config.json",
  "--run-dir", $RunDir,
  "--train-pool-size", "2000",
  "--small-train-size", "1000",
  "--eval-pool-size", "500",
  "--candidate-build-log-interval", "100",
  "--candidate-pool-attempt-multiplier", "1800",
  "--ddqn-episodes", "15000",
  "--ddqn-log-interval", "1500",
  "--model-iter", "160",
  "--perturb-sims", "12",
  "--device", "cuda",
  "--no-require-candidate-oracle-diff"
)

& $PythonExe @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
  Write-Host "========================================================================================================" -ForegroundColor Green
  Write-Host "RCOG v3 suggestion suite finished successfully." -ForegroundColor Green
  Write-Host "Summary: $Summary" -ForegroundColor Green
  Write-Host "========================================================================================================" -ForegroundColor Green
} else {
  Write-Host "========================================================================================================" -ForegroundColor Red
  Write-Host "RCOG v3 suggestion suite failed with exit code $ExitCode." -ForegroundColor Red
  Write-Host "Check log: $Log" -ForegroundColor Red
  Write-Host "========================================================================================================" -ForegroundColor Red
}
