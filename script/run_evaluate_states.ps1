$BrokerId = "1440"

python -m src.experiments.evaluate_states `
    --broker-id $BrokerId `
    --iterations 200 `
    --covariance_type "full" `
    --min_states 10 `
    --max_states 12
