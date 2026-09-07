# CNN Image Classification, Augmentation, and Domain Generalisation

A controlled experimental study of how data augmentation and domain shift affect a CNN's real-world generalisation, using a ResNet-18 image classifier on a recycling waste-sorting problem in PyTorch.

## Central question

Does training-time data augmentation actually improve a model's robustness to real-world conditions it wasn't trained on — and how much does performance drop when a model meets a genuine domain shift?

This project investigates that question through four controlled experiments, isolating one variable at a time (capacity is held fixed throughout; only preprocessing, augmentation, and test domain change) so observed differences can be attributed to a single cause rather than confounded tuning.

## Part 1 — Baseline Model

A ResNet-18 CNN, pretrained on ImageNet with its final layer replaced for 3 classes, was trained to classify waste images into **plastic**, **paper**, and **metal**:

- Dataset: TrashNet subset, 1,486 images (482 plastic / 594 paper / 410 metal)
- Training: 5 epochs, batch size 32, Adam optimiser (lr = 0.001), cross-entropy loss, 224×224 input
- Result: **0.893 test accuracy, 0.891 test macro-F1**, ~11.1 s/epoch

## Part 2 — Data Augmentation & Equal-Budget Comparison

The same architecture was retrained with a training-only augmentation pipeline (random horizontal flip, random rotation, colour jitter), then compared against the baseline under an identical training budget (same epochs, same split, same evaluation metrics).

**Key finding:** under an equal training budget, augmentation *reduced* performance (0.893 → 0.737 test accuracy). The added variance made the 5-epoch training task harder to fit rather than improving generalisation — demonstrating that augmentation strength has to be matched to training budget, not applied by default.

## Part 3 — Error Analysis & Domain Generalisation

Misclassified baseline test images were inspected for patterns (metal was consistently the weakest class, confused with paper and plastic due to reflective, variable surfaces). A 25-image test set was then hand-collected from a genuinely different domain — real-world backgrounds, lighting, and object framing, unlike TrashNet's clean product shots.

**Key finding:** accuracy collapsed from 0.893 (in-domain test) to 0.560 (new-domain test) despite no change to the model — a direct measurement of the cost of domain shift versus in-distribution evaluation.

## Part 4 — Targeted Improvement

A second, more conservative augmentation pipeline (smaller rotation, lighter colour jitter) was applied specifically to target the failure pattern found in Part 3, and evaluated on both the original and new-domain test sets before/after.

**Key finding:** this targeted augmentation did not recover new-domain performance (0.560 → 0.400) and cost accuracy on the original test set (0.893 → 0.670) — evidence that augmentation tuning alone wasn't sufficient here, and that closing a genuine domain gap likely needs real-world training examples rather than synthetic variation.

## Bonus — Model Extraction Reproduction

As a standalone extension, part of Carlini et al. (2024), *Stealing Part of a Production Language Model*, was reproduced at small scale: a language model's hidden dimension was estimated purely from its output logits via SVD, on GPT-2 Small and DistilGPT-2. Both models' estimated dimensions landed within 6–7 of their true hidden dimension (768), reproducing the paper's core low-rank finding.

## Methodology

- Dataset: TrashNet (Kaggle) subset — plastic, paper, metal classes
- Split: 70% train / 15% validation / 15% test, fixed random seed, held out and never used for model selection
- Preprocessing: images resized to 224×224; augmentation (where used) applied to training data only
- Model: ResNet-18, ImageNet-pretrained weights, same architecture held constant across all experiments
- Reproducibility: fixed random seeds throughout for consistent, comparable results across runs
- Evaluation: accuracy and macro-F1 reported for every comparison, using the same split each time

## Tech stack

- Python, PyTorch, torchvision
- Hugging Face Transformers (bonus SVD reproduction)
- scikit-learn (metrics)
- Matplotlib (loss curves, confusion matrices)
- NumPy / Pandas (results aggregation)

## Files

- `Recycling-Image-Classification-Solution.ipynb` — full notebook with experiments, results tables, confusion matrices, and written analysis
- `Recycling-Image-Classification-Output.pdf` — exported PDF of the notebook
