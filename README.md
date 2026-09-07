# Deep Learning Projects

A portfolio of independent deep learning projects, progressing from implementing backpropagation by hand through to applied computer vision and large language model fine-tuning. Each project is self-contained in its own folder with its own README, notebook, and (where applicable) an exported PDF.

## Projects

### [01 — Backpropagation From Scratch](./01-backprop-from-scratch)

A full backward pass for a two-layer MLP (affine → ReLU → affine, trained with MSE loss) derived entirely by hand and implemented in NumPy, with no autograd. Covers the loss gradient, both affine layers' gradients (with respect to weights, biases, and inputs), the ReLU backward mask, full reverse-mode composition in topological order, and a reusable finite-difference gradient checker.

**Key finding:** every analytically derived gradient — for each layer individually and for the full end-to-end composition — matched numerical (finite-difference) gradients to a relative error under 2e-5, confirming the hand-derived chain rule was implemented correctly throughout.

### [02 — Regularisation, Capacity, and Generalisation](./02-regularisation-capacity-generalisation)

A controlled study of how model capacity and L2 regularisation affect generalisation, using two MLPs of different capacity on the Breast Cancer Wisconsin (Diagnostic) dataset in PyTorch.

**Key finding:** the higher-capacity model reached perfect training accuracy (1.00 vs. 0.99) but showed a larger generalisation gap (0.023 vs. 0.013) with no improvement in test accuracy (~0.965 for both) — capacity alone bought optimisation, not generalisation. Increasing L2 strength (λ = 0 → 1e-2) shrank the global weight norm from ~14.08 to ~3.77 and reduced the generalisation gap from ~0.0235 to ~0.0185, directly demonstrating the mechanism by which weight decay constrains complexity.

### [03 — CNN Image Classification, Augmentation, and Domain Generalisation](./03-recycling-image-cnn-classification)

A ResNet-18 image classifier sorting recycling waste into plastic/paper/metal, used to test whether training-time augmentation actually improves robustness to a genuine domain shift.

**Key finding:** under an equal training budget, augmentation *reduced* test accuracy (0.893 → 0.737) rather than improving it, and accuracy collapsed further on a hand-collected real-world test set (0.893 → 0.560) despite no change to the model — a direct measurement of the cost of domain shift that a targeted, more conservative augmentation pass did not recover.

### [04 — DistilGPT2 Domain Adaptation: Full Fine-Tuning vs. LoRA](./04-distilgpt2-lora-finetuning)

A comparison of full fine-tuning against LoRA for adapting DistilGPT2 to a specialised parliamentary-text domain, evaluated by perplexity on both an in-domain validation set and a held-out related-domain test set.

**Key finding:** full fine-tuning (81.9M parameters updated) reduced perplexity from 51.80 → 27.23 (in-domain) and 61.28 → 37.13 (related domain); LoRA recovered roughly half of that improvement (→ 40.85 / 48.32) while updating only 147K parameters — **0.18%** of the model — in about half the training time.

## Repository structure

```
deep-learning-projects/
├── README.md
├── 01-backprop-from-scratch/
├── 02-regularisation-capacity-generalisation/
├── 03-recycling-image-cnn-classification/
└── 04-distilgpt2-lora-finetuning/
```

## Tech stack

- Python, NumPy, PyTorch, torchvision
- Hugging Face Transformers, Datasets, PEFT (LoRA), Accelerate
- scikit-learn (data loading, splitting, preprocessing, metrics)
- pypdf (source document extraction)
- Matplotlib, Pandas
