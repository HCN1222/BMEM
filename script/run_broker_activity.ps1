# run_broker_activity.ps1
# Loops through the configured broker IDs and downloads broker activity.

param(
    [string]$StartDate = "2021-06-30",
    [string]$EndDate = (Get-Date -Format "yyyy-MM-dd"),
    [ValidateSet("parquet", "csv")]
    [string]$Format = "parquet"
)

$ErrorActionPreference = "Stop"

# Make console output UTF-8 friendly (does not change file parsing encoding)
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

# Broker list (ASCII-only to avoid script encoding/parser issues on Windows PowerShell 5.1)
# Mapping reference:
# 1440 Merrill Lynch (美林)
# 1470 Morgan Stanley Taiwan (台灣摩根士丹利)
# 1480 Goldman Sachs (美商高盛)
# 8440 J.P. Morgan (摩根大通)
# 1650 UBS (新商瑞銀)
# 1360 Macquarie (港商麥格理)
# 8960 HSBC (香港上海匯豐)
# 7030 CHI-HO (致和)
# 1660 CLSA (港商聯昌) 沒在做了
# 1400 Schroders (港商蘇皇) FinMind 無分點資料
# 1560 Nomura (港商野村)
$brokers = @(
    @{ Name = "MerrillLynch";  TraderId = 1440 },
    @{ Name = "MorganStanley"; TraderId = 1470 },
    @{ Name = "GoldmanSachs";  TraderId = 1480 },
    @{ Name = "JPMorgan";      TraderId = 8440 },
    @{ Name = "UBS";           TraderId = 1650 },
    @{ Name = "Macquarie";     TraderId = 1360 },
    @{ Name = "HSBC";          TraderId = 8960 },
    @{ Name = "ChiHo";         TraderId = 7030 },
    # @{ Name = "CLSA";          TraderId = 1660 },
    # @{ Name = "Schroders";     TraderId = 1400 },  # FinMind 無分點資料
    @{ Name = "Nomura";        TraderId = 1560 }
)

$failedBrokers = @()
foreach ($b in $brokers) {
    Write-Host "=== Running: $($b.Name) (broker-id=$($b.TraderId)) ==="

    & python -m src.experiments.download_broker_activity `
        --start $StartDate `
        --end $EndDate `
        --broker-id $($b.TraderId) `
        --format $Format

    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("Command failed for {0} (broker-id={1}), exit code: {2}" -f $b.Name, $b.TraderId, $LASTEXITCODE)
        $failedBrokers += $b.TraderId
    }
}

if ($failedBrokers.Count -gt 0) {
    Write-Error "Broker download failed for: $($failedBrokers -join ', ')"
    exit 1
}
