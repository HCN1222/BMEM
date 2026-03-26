python ./src/train.py `
    --input_file "./data/preprocessed_data/hmm_data_train.npz" `
    --outdir "./outputs" `
    --iterations 200 `
    --n_states 3 `
    --covariance_type "full" `
    --tol 1e-3 `
    --random_seed 25