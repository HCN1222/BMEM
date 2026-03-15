python ./src/train.py `
    --input_file "./data/preprocessed_data/hmm_data_train.npz" `
    --outdir "./outputs" `
    --iterations 1000 `
    --n_states 5 `
    --covariance_type "diag" `
    --tol 1e-3 `
    --random_seed 12