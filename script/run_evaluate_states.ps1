python ./src/evaluate_states.py `
    --input_file "./data/preprocessed_data/hmm_data_train.npz" `
    --outdir "./outputs" `
    --iterations 200 `
    --covariance_type "full" `
    --min_states 2 `
    --max_states 6