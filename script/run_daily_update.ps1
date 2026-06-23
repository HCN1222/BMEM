# run_daily_update.ps1
#
# Run the BMEM daily update pipeline for one broker.
# Fetches today's broker activity, computes HMM + XGBoost signals,
# and writes outputs/{broker_id}/daily/signals_YYYY-MM-DD.csv.
#
# Usage (from repo root):
#   .\script\run_daily_update.ps1 -BrokerId "1440"
#   .\script\run_daily_update.ps1 -BrokerId "1470" -Date "2026-04-16"
#
# Prerequisites:
#   - FINMIND_API_KEY set in .env (or as an environment variable)
#   - Trained HMM params  : outputs/{broker_id}/models/hmm/trained_hmm_params.npz
#   - Trained XGBoost     : outputs/{broker_id}/models/xgboost/{long,short}/xgb_trading_model.json
#   - Historical data     : data/brokers/{broker_id}/*.parquet
#                           data/stocks/*.parquet

param(
    [string]$Date     = (Get-Date -Format "yyyy-MM-dd"),
    [Parameter(Mandatory = $true)]
    [string]$BrokerId,
    [string]$OutDir   = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Script   = "$RepoRoot\src\daily_update.py"
if (-not $OutDir) { $OutDir = "$RepoRoot\outputs\$BrokerId\daily" }

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
