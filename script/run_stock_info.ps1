# run_stock_info.ps1
param(
    [string]$StartDate = "2021-06-30",
    [string]$EndDate = (Get-Date -Format "yyyy-MM-dd")
)

python -m src.experiments.download_stock_info `
    --mode brokers `
    --start $StartDate `
    --end $EndDate

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
