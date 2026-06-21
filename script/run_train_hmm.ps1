$BrokerId = "1440"

python -m src.experiments.train_hmm `
    --broker-id $BrokerId `
    --iterations 200 `
    --n_states 3 `
    --covariance_type "full" `
    --tol 1e-3 `
    --random_seed 25
