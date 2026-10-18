# emulator_bench

Multi-seed evaluation of the eMOSAIC-T regressor (TrustAffinityNet, binding affinity part only)
on the default EMULaToR splits. The model uses its original published settings. There is no
hyperparameter search.

## Data

Default data: `data/processed/datasets/kinetic_dataset/ki` of the EMULaToR superproject
(found from this repo's location). Use `--base_dir` for another task folder.
Never use the `unified` datasets.

Each split folder holds `train.parquet`, `val.parquet` and `test.parquet` with the columns
`sequence`, `smiles` and `log10_value`. The scripts find these split groups under `--base_dir`:

- `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`
- `enzyme_sequence_splits/threshold_<t>`, `substrate_splits/threshold_<t>`

The threshold folders come from disk. Use `--thresholds threshold_0.05` to select some.

## Outputs

- Results: `<EMULaToR>/outputs/baselines/eMOSAIC/<task>/<split_group>/<split_name>/seed_<n>/` (`--output_dir`).
  `<task>` is the name of the `--base_dir` folder (`ki` by default). `<split_name>` is `all` for flat split groups.
  Each folder has `bestmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv` and `run_summary.json`.
  `checkpoint_last.pt` holds the resumable training state.
- Summaries: `emosaic_summary_runs.csv` and `emosaic_summary_splits.csv` in `--output_dir`.
- Embedding cache: `<EMULaToR>/outputs/baselines/eMOSAIC/embeddings` (`--embeddings_dir`). It is shared by all tasks and seeds.
  It has `proteins_v3/` and `ligands_v3/`. Protein keys cover the model, dtype and `max_seq_len`.
  Change `CACHE_VERSION` in `common.py` to force a rebuild.
- The scripts never write to the dataset folders.

## Model and settings

- Protein: ESMFold-v1 structure-module single representation (384 per residue, from
  `facebook/esmfold_v1` through `transformers`). Sequences are cut to 700 residues and padded.
  A 5-stage 2D ResNet encodes them.
- Ligand: RDKit 2D graph and a 2-layer GIN (sum pooling). `GNN(pretrained=True)` only skips the Xavier init of the atom embeddings (PyTorch default init is used). No pretrained GIN weights are loaded.
- Attentive pooling, then a 3-layer MLP head (1024, 512, 128, 1). Dropout 0.2.
- Loss MSE. AdamW (lr 1e-4, weight decay 1e-4, amsgrad), cosine warm restarts with T_0 = 10,
  batch size 256, 50 epochs, no gradient clipping.
- The checkpoint with the lowest validation RMSE (same ranking as MSE) is used for test.
- Seeds: 666, 777, 888.

## Scripts

- `cache_embeddings.py`: builds the protein (ESMFold) and ligand caches once per unique input.
- `train_single_target_tvt.py`: trains one split (explicit file paths and `--out_dir`).
- `run_split_benchmarks.py`: caches features, then runs every split group x threshold x seed.
  `--gpus 0 1 --runs_per_gpu 2` runs jobs in parallel, one process per GPU slot.
- `aggregate_results.py`: mean and variance over seeds for each split.
- `predict_single_target.py`: predicts for a parquet or CSV file from a saved checkpoint.

See `commands.txt` for the commands. Run them from `baselines/eMOSAIC`.

## Speed-ups (same model and settings)

- Feature cache for proteins and ligands.
- Training and caching run in fp32 like the original. `--amp` opts in to mixed precision (bf16 on Ampere or newer, fp16 with a grad scaler on older GPUs, plus TF32, cudnn benchmark and `channels_last` for the protein ResNet).
- Data loader workers, optional `--preload_proteins`.
- GPU-resident protein embedding table (default; `--no_gpu_protein_table` switches to per-sample loading with pinned memory).
- Optional `--torch_compile`.
