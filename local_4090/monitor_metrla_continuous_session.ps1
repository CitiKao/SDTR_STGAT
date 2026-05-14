param(
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [string]$SessionTag,
    [int]$PollSeconds = 2
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SessionTag)) {
    throw "SessionTag is required."
}

try {
    $Host.UI.RawUI.WindowTitle = "METR-LA Live Monitor $SessionTag"
}
catch {
}

$statusPath = Join-Path $RootDir ("runs\external_speed\METR-LA_untilbeat_session_{0}.status.json" -f $SessionTag)
$summaryPath = Join-Path $RootDir ("runs\external_speed\METR-LA_untilbeat_session_{0}.jsonl" -f $SessionTag)
$currentLog = $null
$lastLineCount = 0
$lastSummaryCount = 0

Write-Host ("Monitoring METR-LA continuous session {0}" -f $SessionTag) -ForegroundColor Cyan
Write-Host ("status:  {0}" -f $statusPath)
Write-Host ("summary: {0}" -f $summaryPath)

while ($true) {
    if (Test-Path -LiteralPath $summaryPath) {
        try {
            $summaryLines = Get-Content -LiteralPath $summaryPath
            if ($summaryLines.Count -gt $lastSummaryCount) {
                for ($i = $lastSummaryCount; $i -lt $summaryLines.Count; $i++) {
                    Write-Host ""
                    Write-Host ("[SUMMARY] {0}" -f $summaryLines[$i]) -ForegroundColor Yellow
                }
                $lastSummaryCount = $summaryLines.Count
            }
        }
        catch {
        }
    }

    if (Test-Path -LiteralPath $statusPath) {
        try {
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            $activeSeed = $status.active_seed
            if (-not [string]::IsNullOrWhiteSpace([string]$activeSeed)) {
                $logPath = Join-Path $RootDir ("runs\external_speed\METR-LA_untilbeat_seed{0}_{1}\run.stdout.log" -f [int]$activeSeed, $SessionTag)
                if ($logPath -ne $currentLog) {
                    $currentLog = $logPath
                    $lastLineCount = 0
                    Write-Host ""
                    Write-Host ("==> Active seed: {0} | completed_runs={1} | baseline={2}" -f $activeSeed, $status.completed_runs, $status.historical_best_baseline) -ForegroundColor Green
                    Write-Host ("    log: {0}" -f $currentLog)
                }
                if (Test-Path -LiteralPath $currentLog) {
                    $lines = Get-Content -LiteralPath $currentLog
                    if ($lines.Count -gt $lastLineCount) {
                        for ($i = $lastLineCount; $i -lt $lines.Count; $i++) {
                            Write-Host $lines[$i]
                        }
                        $lastLineCount = $lines.Count
                    }
                }
            }
        }
        catch {
        }
    }

    Start-Sleep -Seconds $PollSeconds
}
