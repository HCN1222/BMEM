# run_portfolio_test.ps1
# Batch test: run daily update for broker 8440 from 2026-06-04 to 2026-07-03

$start  = [datetime]"2026-06-05"
$end    = [datetime]"2026-07-03"
$broker = "8440"

$current = $start
while ($current -le $end) {
    $dateStr = $current.ToString("yyyy-MM-dd")
    Write-Host "=== $dateStr ===" -ForegroundColor Cyan
    & .\script\run_daily_update.ps1 -BrokerId $broker -Date $dateStr
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed on $dateStr (exit $LASTEXITCODE), stopping."
        exit $LASTEXITCODE
    }
    $current = $current.AddDays(1)
}

Write-Host "Batch test complete." -ForegroundColor Green
