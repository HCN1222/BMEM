# run_daily_update.ps1
#
# Run the BMEM daily update pipeline for Merrill Lynch (broker 1440).
# Fetches today's broker activity, computes HMM + XGBoost signals,
# and writes outputs/daily/signals_YYYY-MM-DD.csv.
#
# Usage (from repo root):
#   .\script\run_daily_update.ps1                        # target = today
#   .\script\run_daily_update.ps1 -Date "2026-04-16"     # specific date
#
# Prerequisites:
#   - FINMIND_API_KEY set in .env (or as an environment variable)
#   - Trained HMM params  : outputs/models/HMM/trained_hmm_params.npz
#   - Trained XGBoost     : outputs/models/XGBoost/long/xgb_trading_model.json
#                           outputs/models/XGBoost/short/xgb_trading_model.json
#   - Historical data     : data/brokers/1440/*.parquet
#                           data/stocks/*.parquet

param(
    [string]$Date     = (Get-Date -Format "yyyy-MM-dd"),
    [string]$BrokerId = "1440",
    [string]$OutDir   = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\hcn12\OneDrive\Desktop\NING\Github\BMEM"
$Script   = "$RepoRoot\src\daily_update.py"
if (-not $OutDir) { $OutDir = "$RepoRoot\outputs\daily" }

Write-Host "=== BMEM Daily Update ==="
Write-Host "  Target date : $Date"
Write-Host "  Broker ID   : $BrokerId"
Write-Host "  Output dir  : $OutDir"
Write-Host ""

conda run -n BMEM --no-capture-output python $Script `
    --date      $Date `
    --broker-id $BrokerId `
    --outdir    $OutDir

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Daily update failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done. Signals saved to: $OutDir\signals_$Date.csv"
