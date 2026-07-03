# BMEM — Claude Instructions

## Startup

At the start of every new conversation, read `README.md` before answering any question. This gives you full context on the project's pipeline, data layout, model architecture, and broker ID conventions.

## Environment

- Always use the conda environment `BMEM` when running Python commands.
- Run scripts from the repository root.

## Project Context (summary — README is authoritative)

- **Stack:** HMM (hmmlearn) → XGBoost two-stage classifier for Taiwan stock signals.
- **Broker namespace:** Every artifact (data, models, outputs, signals) lives under `{broker_id}/`. Always require `--broker-id` when invoking any pipeline script.
- **Production entry point:** `src/daily_update.py --broker-id <id>`
- **Research workflow:** Steps 1–8 in README; scripts under `script/`.
- **Key paths:** `data/brokers/`, `data/stocks/`, `data/preprocessed_data/`, `outputs/{broker_id}/`
- **Signal thresholds:** long `≥ 0.6`, short `≥ 0.8`
- **No lookahead:** Rolling 120-day HMM window only; never use future data in inference.
