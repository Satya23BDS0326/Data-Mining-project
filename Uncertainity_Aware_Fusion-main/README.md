[README.md](https://github.com/user-attachments/files/26611368/README.md)
# UAF TruthfulQA Pipeline

Uncertainty-Aware Ensemble Framework (UAF) for TruthfulQA MC1. This project runs multiple LLMs, estimates uncertainty with multiple methods, performs uncertainty-aware fusion, and exports evaluation tables.

## Project files

- `main.py`: Full pipeline orchestration (validation, selector, test, fusion, metrics)
- `config.py`: Runtime configuration (models, device, cache version, paths)
- `model_runner.py`: Model loading, inference, compatibility helpers
- `data_loader.py`: TruthfulQA loading and MC1 extraction
- `uncertainty.py`: Perplexity, margin, semantic entropy, Haloscope
- `selector.py`: Model scoring/ranking and K selection
- `fuser.py`: Ensemble fusion and baseline majority voting
- `metrics.py`: Result tables and report exports
- `UAF_TruthfulQA_Pipeline.ipynb`: Colab notebook workflow

## Environment setup

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install torch transformers datasets accelerate bitsandbytes scikit-learn pandas matplotlib tqdm huggingface_hub sentencepiece sentence-transformers protobuf
```

## Run locally

Run full pipeline:

```bash
python main.py
```

Run single model for sanity check:

```bash
RUN_ONLY_MODEL=phi3.5-mini python main.py
```

## Run on Google Colab

1. Open `UAF_TruthfulQA_Pipeline.ipynb`.
2. Set runtime to GPU.
3. Upload project files or mount Drive.
4. Install dependencies in notebook cells.
5. If using gated models, run Hugging Face login.
6. Execute pipeline cells in order.

## Important environment variables

- `UAF_DEVICE`: `cuda`, `mps`, or `cpu`
- `UAF_USE_4BIT`: `true`/`false`
- `UAF_MODEL_PROFILE`: `default` or `colab_safe`
- `UAF_CACHE_VERSION`: Cache namespace to force clean recompute
- `UAF_CACHE_DIR`: Cache folder path
- `UAF_RESULTS_DIR`: Results folder path
- `RUN_ONLY_MODEL`: Restrict run to one or more model aliases

## Output folders

- `cache/`: Cached validation/test artifacts
- `results/`: CSV results, predictions, plots

## Troubleshooting

- If a model shows unrealistic accuracy (for example 100%), clear cache and rerun:

```bash
rm -rf cache/* results/*
```

- If all inferences fail for a model, start with one model first and inspect first failure logs.
- For gated models, confirm Hugging Face auth is completed in your runtime.

## Notes

This project is intended for research and experimentation on uncertainty-aware model fusion.
