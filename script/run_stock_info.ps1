# run_stock_info.ps1
python ./src/experiments/download_stock_info.py `
    --mode list --stock-ids-json ./data/stocks/stock_ids.json `
    --start 2021-06-30 --end 2026-02-11 `
    --outdir ./data/stocks