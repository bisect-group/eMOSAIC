# emulator_bench

Multi-seed evaluation of the eMOSAIC-T regressor (TrustAffinityNet, binding affinity part only)
on the default EMULaToR splits. The model uses its original published settings. There is no
hyperparameter search.

## Data

Default data: `data/processed/datasets/kinetic_dataset/ki` of the EMULaToR superproject
(found from this repo's location). eMOSAIC runs on Ki only. Use `--base_dir` to override the root.
Never use the `unified` datasets.

Each split folder holds `train.parquet`, `val.parquet` and `test.parquet` with the columns
`sequence`, `smiles` and `log10_value`. The scripts find these split groups under `--base_dir`:

- `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`
- `enzyme_sequence_splits/threshold_<t>`, `enzyme_structure_splits/threshold_<t>`,
  `substrate_splits/threshold_<t>`, `conformer_cosine_splits/threshold_<t>`

The threshold folders come from disk. Use `--thresholds threshold_0.05` to select some.

## Outputs

- Results: `<EMULaToR>/outputs/baselines/eMOSAIC/ki/<split_group>/<split_name>/seed_<n>/` (`--output_dir`).
  `<split_name>` is `all` for flat split groups.
  Each folder has `bestmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv` and `run_summary.json`.
  `checkpoint_last.pt` holds the resumable training state.
- Summaries: `emosaic_summary_runs.csv` and `emosaic_summary_splits.csv` in `--output_dir`.
- Embedding cache: `<EMULaToR>/outputs/baselines/eMOSAIC/embeddings` (`--embeddings_dir`). It is shared by all seeds.
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
- Protein cache (`proteins_v4`): the final ESMFold structure-module single state (`states[-1]`, 384 per residue). It is not the 1024-d folding-trunk `s_s` of the older `proteins_v3` cache, which is never read. The ESM-2 language model runs in fp16 with bf16 autocast, stored as fp16. fp32 weights need about 15 GB and run out of memory at 700 residues on a 22 GB A10. `cache_embeddings.py --fp32` runs the full model in fp32 instead. The manifest records `protein_precision`. Training stops with an error if the cache width differs from `--prot_input_dim`.
- Training runs in fp32 like the original. `--amp` opts in to mixed precision (bf16 on Ampere or newer, fp16 with a grad scaler on older GPUs, plus TF32, cudnn benchmark and `channels_last` for the protein ResNet).
- Data loader workers, optional `--preload_proteins`.
- GPU-resident protein embedding table (default; `--no_gpu_protein_table` switches to per-sample loading with pinned memory).
- Optional `--torch_compile`.

## Verify equivalence

```
emulator_bench/verify_equivalence.sh          # CPU, fp32/fp64, about 20 s
emulator_bench/verify_equivalence.sh --gpu    # also the bf16 test; needs one CUDA device
```

Expected env: `mldb` (torch, torch_geometric, rdkit, scikit-learn, pytest). The script does not create it. It fails if any CPU test is skipped. It needs no private data or downloads: it uses random protein embeddings and a tiny random ESMFold.

What `tests/` compares:
- Model: outputs, loss and gradients against the original `BindingModel2` in fp64 (atol 1e-10), in eval mode and in train mode with seeded dropout.
- Batches: wrapper batches (with and without the protein table) against the original collate, exactly.
- Training: one optimizer step of the wrapper `main()` against the original `trainer.train` (atol 1e-6, dropout removed on both sides). The learning-rate schedule matches (rtol 1e-12). Neither side clips gradients.
- Features: the structure-module feature extractor and the 384-d cache width check.
- `--gpu`: three epochs of `main()` with `--amp` (bf16, `channels_last`) against fp32 on the same GPU and seed. Dropout is removed from all runs, because dropout masks differ between dtypes. The toy has 96 rows, batch 48, lr 1e-4, 3 epochs. The noise control is the production run with initial weights perturbed by one rounding of the production dtype (2**-8, bf16). Per step, the gap |production - fp32| must be at most 10 times the floor |perturbed production - production|. The same holds for the final test RMSE. Step 0 is judged like every other step: its gap must be at most 10 times its floor. Each test prints `max gap`, `max floor` and `ratio`; run with `-s` to see the line.

Negative control: `emulator_bench/verify_equivalence.sh --negative-control` multiplies each initial weight of the compared (production) run by 1 + 0.20 * randn (fixed seed; size set by `EQUIV_NEGATIVE_CONTROL_SIZE`) and runs only the GPU tests. It exits 0 only if every GPU test fails, so a passing GPU test shows a bound too loose to catch a 20% random weight error. The eMOSAIC bf16 test detects a random per-weight error of 20% and larger, not 5%, because its bf16 loss is weakly sensitive to weights; the fp32 CPU tests check exact equivalence. A uniform x1.05 scale is not used because BatchNorm makes it nearly a no-op.

Not exact: `--amp` training differs from fp32 by rounding. The protein cache is stored in fp16 and the GPU table is fp16 in both modes. The original metric code needs an older scikit-learn (`squared=`), so the test replaces it with plain MSE.
