$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TERM = "xterm-256color"

$Repo = "D:\STDR"
$PythonExe = "C:\Users\WMNL1691\Anaconda3\envs\stable_diffusion\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo "runs\rcog_safety_narrow_$Stamp"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Log = Join-Path $RunDir "terminal.log"
$Summary = Join-Path $RunDir "rcog_safety_refined_summary.json"
Set-Content -Path (Join-Path $Repo "runs\latest_rcog_safety_narrow_run.txt") -Value $RunDir

Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host "RCOG narrow safety/recall search started" -ForegroundColor Cyan
Write-Host "RunDir:  $RunDir" -ForegroundColor Yellow
Write-Host "Log:     $Log" -ForegroundColor Yellow
Write-Host "Summary: $Summary" -ForegroundColor Yellow
Write-Host "========================================================================================================" -ForegroundColor Cyan

Set-Location $Repo

$PythonArgs = @(
  "D:\STDR\rcog_safety_refined_search.py",
  "--checkpoint", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\ddqn_reranker_final.pt",
  "--train-config", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\config.json",
  "--cache-run-dir", "D:\STDR\runs\rcog_v3_focused_fullscale_20260520_211152",
  "--records-cache-run-dir", "D:\STDR\runs\rcog_stability_validation_20260520_232340",
  "--run-dir", $RunDir,
  "--seeds", "121,122,123,124,125",
  "--train-resample-mode", "bootstrap",
  "--caps", "0.32,0.34,0.36",
  "--bad-weights", "1.2,1.4,1.6,1.8",
  "--max-activations", "9.5,10,10.5",
  "--min-precisions", "62,63,64",
  "--min-benefit-recalls", "59,60,61,62",
  "--min-opportunity-recalls", "50,51,52",
  "--objectives", "ratio,balanced",
  "--bootstrap-top-k", "30",
  "--bootstrap-iters", "1000",
  "--device", "cuda",
  "--no-require-candidate-oracle-diff"
)

& $PythonExe @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
  Write-Host "========================================================================================================" -ForegroundColor Green
  Write-Host "RCOG narrow safety/recall search finished successfully." -ForegroundColor Green
  Write-Host "Summary: $Summary" -ForegroundColor Green
  Write-Host "========================================================================================================" -ForegroundColor Green
} else {
  Write-Host "========================================================================================================" -ForegroundColor Red
  Write-Host "RCOG narrow safety/recall search failed with exit code $ExitCode." -ForegroundColor Red
  Write-Host "Check log: $Log" -ForegroundColor Red
  Write-Host "========================================================================================================" -ForegroundColor Red
}
