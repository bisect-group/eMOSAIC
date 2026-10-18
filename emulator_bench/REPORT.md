# eMOSAIC-T wrapper report

## Method (paper: Badkul et al., Nature Machine Intelligence 7, 1985-1995, 2025)
- eMOSAIC adds an uncertainty module on top of a binding-affinity model. This wrapper trains only the base regressor (TrustAffinityNet), as the Leak-CURBER paper does for "eMOSAIC-T" on Ki.
- Protein: ESMFold structure-module embedding, padded to 700 residues, 5-stage 2D ResNet.
- Ligand: 2D graph, 2-layer GIN, sum pooling.
- PLI module: attentive pooling, then MLP (1024, 512, 128, 1), MSE loss.
- The best validation checkpoint is used for evaluation. The anomaly-detection (Mahalanobis) module is not used.

## Original default settings
| Setting | Value | Source |
|---|---|---|
| epochs | 50 | code/BindingAffinityModule/main.py:113 (`--num_epochs`) |
| batch size | 256 | main.py:112 |
| optimiser | AdamW, amsgrad=True, lr 1e-4, wd 1e-4 | main.py:55, 115-116 |
| schedule | CosineAnnealingWarmRestarts T_0=10, stepped per epoch | main.py:56, 89 |
| loss | MSE | main.py:54 |
| dropout | 0.2 | main.py:117 |
| gradient clipping | none | main.py / trainer.py |
| checkpoint | lowest validation MSE, strict `<` | main.py:90-93 |
| early stopping | none | main.py |
| architecture | 2 GIN layers, emb 300, ResNet [16,32,64,32,16] x2, combined_dim 1004 (300+704), max_length 700 | code/config.yaml, model.py |
| precision | fp32 | main.py (no autocast) |
| protein features | ESMFold embeddings, sequences > 700 dropped | paper Methods p.1992; dataset.py (precomputed files) |
| seed | 42 | main.py:114 |

Paper vs code: the paper gives hyperparameters only in Supplementary Tables 5 and 6, which are not in the PDF. I used the code defaults. Paper and code agree on 700 residues, 5-layer ResNet and 2D GIN. The paper text says "5-layer GIN"; the code config uses `num_layers: 2`. I follow the code (2).
The paper text says protein sequences longer than 700 are removed. The wrapper truncates to 700 residues instead (Leak-CURBER test rows must not be dropped).

## What the wrapper does now
- `run_split_benchmarks.py` finds split folders in `--base_dir` (default `data/processed/datasets/kinetic_dataset/ki`): `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`. Thresholds come from disk (now 0.05 and 0.545).
- Caches ESMFold and ligand features, then trains each split x seed (default seeds 666 777 888) with the original settings above. Optional `--gpus` runs jobs in parallel.
- Writes results to `results/<task>/...` in the submodule (gitignored), not into the dataset folders.
- `aggregate_results.py` computes mean and variance over seeds.

## Files
Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `default_hparams_original.json` (argparse defaults now equal the original settings).
Changed: `common.py`, `run_split_benchmarks.py`, `train_single_target_tvt.py`, `cache_embeddings.py`, `aggregate_results.py`, `README.md`, `commands.txt`, `.gitignore` (now `__pycache__/` and `results/` on separate lines), `feature_pipeline.py`, `predict_single_target.py`.
Unchanged: `dataset.py`, `modeling.py`. No original (non-emulator_bench) code was edited.

## Bugs and mismatches fixed
- Default `--clip_grad 1.0` was not in the original. Gradient clipping is removed.
- Default split groups had no random splits, and listed `enzyme_structure_splits` and `conformer_cosine_splits`, which do not exist. Defaults now match the folders on disk. Old `random_splits` alias and `group_shuffle_splits` code removed.
- Default paths pointed to `/home/mallikarjun/...` and a missing `data/processed/baselines` folder. Now repo-relative.
- Results were written inside the dataset split folders. Now written to `--output_dir`.
- Default seed 42 changed to 666 777 888.
- Aggregator had a hardcoded `~/github/...` path. Now `--results_dir`.
- `run_split_benchmarks.py` exits non-zero if any parallel job failed. It writes the runs CSV and prints the failed jobs first.
- `commands.txt`: the redundant step 1b is removed.
- `logfile.csv` is reset at the start of a run (a rerun no longer appends to an old log).
- `train_single_target_tvt.py` now needs explicit `--train_path/--val_path/--test_path/--out_dir` (the base_dir mode was removed).

## Deviations from the original protocol
- Default precision is fp32, like the original (the Leak-CURBER appendix bf16 claim does not override the code). `--amp` is an opt-in flag in training, caching, prediction and the launcher. It turns on bf16 autocast (fp16 with GradScaler on pre-Ampere), TF32 and cudnn benchmark. Protein embeddings are cast to fp32 on the device.
- Protein embeddings stored as fp16 in the cache.
- Sequences truncated to 700 instead of dropped (see above).
- Validation ranking uses RMSE (monotone with MSE, same checkpoint).
- Embeddings come from HF `transformers` `facebook/esmfold_v1` (`s_s`). The original code loads precomputed embeddings from the authors' private disk, so the exact extractor cannot be checked. The ResNet output size (704) matches `combined_dim 1004`.

## Commands (from `baselines/eMOSAIC`, env mldb)
```
python -u emulator_bench/run_split_benchmarks.py --seeds 666 777 888 --gpus 0 1 --pin_memory --preload_proteins
python emulator_bench/aggregate_results.py --results_dir results/ki
```
Only Ki applies to this model (`kinetic_dataset/ki`). For another task folder pass `--base_dir` and `--output_dir`.

## Smoke test
Env: `/home/adhil/.conda/envs/mldb`, GPU 1 (A10), fp32 default. Tiny data (64/32/32 rows) from `ki` random_splits_grouped_sequence and enzyme_sequence_splits/threshold_0.05, copied to the scratchpad. ESMFold weights are not cached locally, so protein embeddings were random 384-dim fakes written with the wrapper's cache writer. Ligand caching used the real `cache_embeddings.py --skip_proteins`.
`run_split_benchmarks.py --skip_cache --epochs 1 --seeds 666` trained both splits, wrote all outputs, and `aggregate_results.py` ran. Nothing was written under EMULaToR/data.
Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/eMOSAIC`.
Re-run after the fixes (default fp32; `run_summary.json` shows `precision_mode: fp32`): training, test evaluation, aggregation and `predict_single_target.py` ran. A forced job failure (`--gpus 1` with a missing embeddings dir) exited with code 1. Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/eMOSAIC/out2`.
Not tested: real ESMFold embedding (`facebook/esmfold_v1` download, ~8 GB), multi-GPU `--gpus` path, full-size runs.

## Open issues
- Supplementary Tables 5 and 6 (paper hyperparameters) were not available. Code defaults used.
- Original embedding extractor is not reproducible from the repo. The HF ESMFold `s_s` embedding is an assumption.
- Protein embeddings are cached as fp16 (the extractor is unverified anyway).
- Time-split ki on disk: `uniprot_time_splits` has 9512 val and 8728 test rows (as in manifest).
- `results/` for the full Ki run is large (checkpoints per seed x split). The ESMFold cache for 183k Ki rows needs a long first run.
