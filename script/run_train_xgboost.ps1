$BrokerId = "1440"
$Sides = @("long", "short")

foreach ($Side in $Sides) {
    Write-Host "=== Training XGBoost: broker=$BrokerId side=$Side ==="
    python -m src.experiments.train_xgboost `
        --broker-id $BrokerId `
        --side $Side

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
