# Refresh raw data in dependency order:
#   1. Download all configured broker activity.
#   2. Rebuild data/stocks/stock_ids.json from the union of broker stock IDs.
#   3. Download/update shared stock OHLCV data for that list.

param(
    [string]$StartDate = "2021-06-30",
    [string]$EndDate = (Get-Date -Format "yyyy-MM-dd")
)

$ErrorActionPreference = "Stop"

Write-Host "=== Phase 1/2: broker activity ==="
& "$PSScriptRoot\run_broker_activity.ps1" -StartDate $StartDate -EndDate $EndDate
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Phase 2/2: stock ID union and stock prices ==="
& "$PSScriptRoot\run_stock_info.ps1" -StartDate $StartDate -EndDate $EndDate
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Data pipeline completed successfully."
