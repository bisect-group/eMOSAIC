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

- Results: `results/<task>/<split_group>[/<threshold>]/seed_<n>/` in this repo (`--output_dir`).
  Each folder has `bestmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv` and `run_summary.json`.
- Summaries: `emosaic_summary_runs.csv` and `emosaic_summary_splits.csv` in `--output_dir`.
- Embedding cache: `results/embeddings` (`--embeddings_dir`). It is shared by all tasks and seeds.
- The scripts never write to the dataset folders.

## Model and settings

- Protein: ESMFold-v1 structure-module single representation (384 per residue, from
  `facebook/esmfold_v1` through `transformers`). Sequences are cut to 700 residues and padded.
  A 5-stage 2D ResNet encodes them.
- Ligand: RDKit 2D graph and a 2-layer GIN (pretrained init, sum pooling).
- Attentive pooling, then a 3-layer MLP head (1024, 512, 128, 1). Dropout 0.2.
- Loss MSE. AdamW (lr 1e-4, weight decay 1e-4, amsgrad), cosine warm restarts with T_0 = 10,
  batch size 256, 50 epochs, no gradient clipping.
- The checkpoint with the lowest validation RMSE (same ranking as MSE) is used for test.
- Seeds: 666, 777, 888.

## Scripts

- `cache_embeddings.py`: builds the protein (ESMFold) and ligand caches once per unique input.
- `train_single_target_tvt.py`: trains one split (explicit file paths and `--out_dir`).
- `run_split_benchmarks.py`: caches features, then runs every split group x threshold x seed.
  `--gpus 0 1` runs jobs in parallel, one process per GPU slot.
- `aggregate_results.py`: mean and variance over seeds for each split.
- `predict_single_target.py`: predicts for a parquet or CSV file from a saved checkpoint.

See `commands.txt` for the commands.

## Speed-ups (same model and settings)

- Feature cache for proteins and ligands.
- Training and caching run in fp32 like the original. `--amp` opts in to mixed precision (bf16 on Ampere or newer, fp16 with a grad scaler on older GPUs, plus TF32 and cudnn benchmark).
- Data loader workers, pinned memory, optional `--preload_proteins`.
- Optional `--torch_compile`.
