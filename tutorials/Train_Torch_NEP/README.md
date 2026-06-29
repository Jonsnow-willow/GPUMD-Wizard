# TorchNEP Training Commands

This tutorial collects the command-line operations used for TorchNEP runs. It is especially useful when a training directory has been copied from a remote machine and the final `nep.txt` was not exported before the job stopped.

## Environment

Run commands from a checkout of `GPUMD-Wizard`, or add the checkout to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/GPUMD-Wizard:$PYTHONPATH
```

On the Infini AI mounted storage used in current large runs, the equivalent is:

```bash
export PYTHONPATH=/mnt/project_materials/jiahui/GPUMD-Wizard:/mnt/project_materials/jiahui/python_libs:$PYTHONPATH
```

In the examples below, `RUN_DIR` means a training directory containing `nep.in`, for example:

```bash
RUN_DIR=/mnt/project_materials/jiahui/runs/torchnep-mptrj-nep4-ann80-e100-resume-20260626
```

## Train

Start a run directory that already contains `nep.in`, `train.xyz`, and optional `test.xyz`:

```bash
python -m wizard.torchNEP train RUN_DIR
```

Override runtime options from the command line:

```bash
python -m wizard.torchNEP train RUN_DIR --device cuda --epochs 100
```

`--epochs` is the target total epoch count, not the number of extra epochs.

## Resume

Resume from the last saved checkpoint:

```bash
python -m wizard.torchNEP train RUN_DIR --resume checkpoints/last.pt --epochs 100 --device cuda
```

If training stops in the middle of an epoch, `checkpoints/last.pt` only contains the last fully saved epoch. Use `inspect` to check what is actually saved.

For Infini AI jobs, prefer the mounted launch script and resume mode when submitting a new web task:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore:Grad strides do not match bucket view strides:UserWarning" TORCHNEP_MODE=resume TORCHNEP_RESUME_FROM=/mnt/project_materials/jiahui/runs/OLD_RUN TORCHNEP_RUN_DIR=/mnt/project_materials/jiahui/runs/NEW_RUN TORCHNEP_BATCH_SIZE=16 TORCHNEP_MAX_TRAIN_FRAMES=all TORCHNEP_EPOCHS=100 TORCHNEP_COMPARE=0 bash /mnt/project_materials/jiahui/run_torchnep_train.sh
```

## Inspect Checkpoints

Inspect the latest saved checkpoint:

```bash
python -m wizard.torchNEP inspect RUN_DIR --artifact checkpoints/last.pt
```

Inspect the best checkpoint:

```bash
python -m wizard.torchNEP inspect RUN_DIR --artifact checkpoints/best.pt
```

Important artifacts:

- `checkpoints/last.pt`: last saved training state, including model, optimizer, scheduler, and RNG state.
- `checkpoints/best.pt`: checkpoint with the best tracked training metric.
- `logs/metrics.csv`: epoch-level loss, component losses, learning rate, and timing.
- `exports/nep.txt`: static NEP artifact for GPUMD or CPU NEP workflows.

## Export `nep.txt`

If the job was interrupted before final export, export from `last.pt`:

```bash
python -m wizard.torchNEP export RUN_DIR --artifact checkpoints/last.pt --output exports/nep_last.txt --device cpu
```

Export the best checkpoint:

```bash
python -m wizard.torchNEP export RUN_DIR --artifact checkpoints/best.pt --output exports/nep_best.txt --device cpu
```

Exporting does not need a GPU. `--device cpu` is recommended when converting a copied run directory on a login node or local machine.

## Evaluate

Evaluate a checkpoint or exported artifact on the configured train/test data:

```bash
python -m wizard.torchNEP eval RUN_DIR --artifact checkpoints/best.pt --device cpu
```

Choose a split explicitly:

```bash
python -m wizard.torchNEP eval RUN_DIR --artifact exports/nep_best.txt --split test --device cpu
```

## Compare Exported Text Against Checkpoint

After exporting, compare the static `nep.txt` artifact against the checkpoint it came from:

```bash
python -m wizard.torchNEP compare RUN_DIR --left exports/nep_last.txt --right checkpoints/last.pt --split train --device cpu
```

For a best-checkpoint export:

```bash
python -m wizard.torchNEP compare RUN_DIR --left exports/nep_best.txt --right checkpoints/best.pt --split train --device cpu
```

This is the quick check that the exported NEP text artifact reproduces the Torch checkpoint predictions on the same structures.
