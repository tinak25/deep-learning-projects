# Regularisation, Capacity, and Generalisation

A controlled experimental study of how model capacity and L2 regularisation affect generalisation in deep neural networks, using the Breast Cancer Wisconsin (Diagnostic) dataset in PyTorch.

## Central question

**How does controlling model complexity influence generalisation behaviour?**

This project investigates that question through two controlled experiments, isolating one variable at a time so that observed differences in performance can be attributed to a single cause rather than confounded tuning.

## Part 1 — Capacity and Generalisation

Two fully connected MLPs of different capacity were trained under identical conditions (same data split, optimiser, learning rate, epochs, and random seed), with **no regularisation applied**, to isolate the effect of model size alone:

- **Small model:** 1 hidden layer (64 units)
- **Large model:** 3 hidden layers (256 → 128 → 64 units)

For each model, training/validation loss and accuracy were tracked across 60 epochs, and the **generalisation gap** (`train_acc − val_acc`) was measured to quantify overfitting.

## Part 2 — L2 Regularisation Study

Using the higher-capacity model from Part 1, three runs were trained with different L2 regularisation strengths (`λ = 0, 1e-4, 1e-2`) via PyTorch's `weight_decay`, holding everything else constant. For each run, the model's global weight norm, final accuracy across train/validation/test splits, and generalisation gap were recorded and compared.

**Key finding:** increasing λ substantially reduced the model's weight norm (from ~14.08 at λ=0 down to ~3.77 at λ=1e-2), which in turn reduced overfitting — directly demonstrating the mechanism by which L2 regularisation constrains model complexity to improve generalisation.

## Methodology

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), scikit-learn — 30 features, binary classification
- **Split:** 70% train / 15% validation / 15% test, stratified, fixed random seed
- **Preprocessing:** Features standardised via `StandardScaler`, fit on training data only (no data leakage into validation/test)
- **Models:** Fully connected MLPs, ReLU activations, cross-entropy loss, Adam optimiser
- **Reproducibility:** Fixed random seeds throughout for consistent, comparable results across runs
- **Verification:** All experiments include automated assertion checks confirming schema correctness, valid metric ranges, and that training actually converged

## Tech stack

- Python, PyTorch
- scikit-learn (data loading, splitting, preprocessing)
- Matplotlib (loss/accuracy visualisation)
- Pandas (results aggregation)

## File

- [`regularisation_capacity_generalisation.ipynb`](./regularisation_capacity_generalisation.ipynb) — full notebook with experiments, results tables, loss curves, and written analysis
