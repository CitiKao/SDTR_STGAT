param(
    [string]$CampaignRoot = "",
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

$breakthroughPath = Join-Path $CampaignRoot "campaign_breakthroughs.jsonl"
$batchIndexPath = Join-Path $CampaignRoot "batch_index.tsv"
$autopilotLogPath = Join-Path $CampaignRoot "autopilot.log"

while ($true) {
    Clear-Host
    Write-Host "AUTOPILOT BREAKTHROUGHS"
    Write-Host "CampaignRoot: $CampaignRoot"
    Write-Host ""
    if (Test-Path $batchIndexPath) {
        Write-Host "batch_index tail:"
        Get-Content -Path $batchIndexPath -Tail 10
        Write-Host ""
    }
    if (Test-Path $breakthroughPath) {
        Write-Host "campaign_breakthroughs tail:"
        Get-Content -Path $breakthroughPath -Tail 20
        Write-Host ""
    } else {
        Write-Host "waiting for campaign_breakthroughs.jsonl ..."
        Write-Host ""
    }
    if (Test-Path $autopilotLogPath) {
        Write-Host "autopilot.log tail:"
        Get-Content -Path $autopilotLogPath -Tail 20
    }
    Start-Sleep -Seconds $IntervalSeconds
}
