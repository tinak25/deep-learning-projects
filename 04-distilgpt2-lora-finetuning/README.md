# DistilGPT2 Domain Adaptation: Full Fine-Tuning vs. LoRA

A controlled experimental comparison of full fine-tuning against LoRA (parameter-efficient fine-tuning) for adapting a small pretrained language model to a specialised text domain, using DistilGPT2 in PyTorch/Hugging Face.

## Central question

When adapting a small pretrained language model to a specialised domain, how much improvement comes from full fine-tuning, and how much of that improvement can LoRA recover at a fraction of the training cost?

This project investigates that question through three controlled stages, holding the base model, tokenizer, data files, and evaluation protocol fixed throughout so that differences in results can be attributed to the adaptation method alone.

## Part 1 — Data Preparation & Pretrained Baseline

Two public parliamentary reports (Joint Standing Committee on Aboriginal and Torres Strait Islander Affairs) were extracted to plain text and cleaned:

- **Fine-tuning domain:** *Inquiry into the application of the United Nations Declaration on the Rights of Indigenous Peoples in Australia* — split 80/20 into `undrip_train.txt` (~92k tokens) and `undrip_val.txt` (~22k tokens)
- **Held-out related domain:** *Inquiry into economic self-determination and opportunities for First Nations Australians* — used whole as `economic_test.txt` (~100k tokens), never seen during training

Pretrained DistilGPT2 was evaluated on both files before any adaptation:

| | PPL on UNDRIP validation | PPL on Economic test |
|---|---|---|
| Pretrained baseline | 51.80 | 61.28 |

## Part 2 — Full Fine-Tuning

DistilGPT2 was fully fine-tuned for 1 epoch on `undrip_train.txt` (all 81,912,576 parameters updated), then evaluated on both the in-domain validation set and the held-out related-domain test set.

**Result:** PPL dropped to **27.23** (UNDRIP validation) and **37.13** (Economic test) — a substantial improvement on both, showing the adaptation transferred beyond just the training text itself.

## Part 3 — LoRA Adaptation & Comparison

A fresh DistilGPT2 was adapted with LoRA instead (rank 8, applied to attention projections only — **147,456 trainable parameters, 0.18% of the model**), trained under the same protocol, and evaluated identically.

| Model state | Parameters updated | Training time (min) | PPL — UNDRIP val | PPL — Economic test |
|---|---|---|---|---|
| Pretrained | 0 | — | 51.80 | 61.28 |
| Full fine-tuned | 81,912,576 | 0.53 | 27.23 | 37.13 |
| LoRA-adapted | 147,456 | 0.25 | 40.85 | 48.32 |

**Key finding:** LoRA recovered roughly half of full fine-tuning's perplexity improvement while updating **~0.18% of the parameters** in about half the training time — a clear illustration of the efficiency/performance trade-off PEFT methods are designed around.

Qualitatively, both adapted models shifted generation toward domain-specific language ("Declaration on the Rights of Indigenous Peoples") compared to the generic pretrained output, though both showed repetition — a sign the small training set encouraged some overfitting to recurring parliamentary phrasing rather than deeper generalisation.

## Methodology

- Base model: `distilgpt2`, fixed tokenizer throughout (pad token set to EOS)
- Same three data files used for every evaluation: `undrip_train.txt`, `undrip_val.txt`, `economic_test.txt`
- Evaluation metric: perplexity via sliding-window negative log-likelihood (block size 128)
- Generation compared using one fixed prompt across all three model states, greedy decoding
- Reproducibility: fixed random seed throughout
- LoRA config: r=8, alpha=16, dropout=0.05, targeting attention projection layers only

## Tech stack

- Python, PyTorch
- Hugging Face Transformers, Datasets, PEFT (LoRA), Accelerate
- pypdf (source document extraction)
- Matplotlib (training-loss curves)
- NumPy / Pandas (results aggregation)

## Files

- `DistilGPT2-LoRA-Finetuning-Solution.ipynb` — full notebook: data preparation, pretrained baseline, full fine-tuning, LoRA adaptation, results tables, and analysis
- `DistilGPT2-LoRA-Finetuning-Output.pdf` — exported PDF of the notebook
