# Run the BMEM pipeline test bench.
#
# Tests included:
#   Test 1   - Feature array consistency (parquet vs npz)
#   Test 1b  - compute_observation_features end-to-end (raw data -> features vs parquet)
#   Test 2   - HMM rolling inference replication  [skipped by default; set RUN_INFERENCE=True to enable]
#   Test 3   - Long XGBoost signal quality
#   Test 4   - Short XGBoost signal quality
#
# Usage (from repo root):
#   .\script\run_test_pipeline.ps1 -BrokerId "1440"

param(
    [Parameter(Mandatory = $true)]
    [string]$BrokerId
)

conda run -n BMEM --no-capture-output python ./src/test_pipeline.py --broker-id $BrokerId
