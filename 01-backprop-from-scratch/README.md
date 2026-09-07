# Backpropagation from Scratch — Reverse-Mode Autodiff with NumPy

A from-scratch implementation of backpropagation for a small feedforward neural network, built without any autograd library (no PyTorch, no TensorFlow — pure NumPy).

## What this project demonstrates

Rather than treating backpropagation as a black box, this project derives and implements every gradient by hand, framing backprop as **reverse composition of Jacobians**. Each layer's backward pass is derived from first principles using perturbation analysis and inner-product manipulation, then verified numerically against finite-difference gradients.

The network:

```
X → (affine: XW + b) → (ReLU) → (affine: Hv + c) → (MSE loss)
```

## What's implemented

- **Local gradient derivations** for each layer (loss, output affine map, ReLU activation, input affine map), each derived from perturbation reasoning rather than memorised calculus formulas
- **Fully vectorised backward pass functions** for every layer, each independently verified against finite-difference approximations
- **A complete forward + backward pass** (`full_forward_backward`) that composes all local gradients in strict reverse topological order
- **A reusable gradient-checking utility** comparing analytical gradients against central-difference numerical gradients
- **Written analysis** covering why reverse-mode differentiation is efficient for scalar-valued losses, why the computation order reverses, and the conceptual challenges of working through Jacobian transposes and batch gradient accumulation

## Why this approach

Every gradient in this project satisfies:

```
relative_error < 2e-5
```

against a finite-difference numerical check — meaning the hand-derived math and the hand-written code are verified correct, not just assumed correct.

## Tech stack

- Python
- NumPy (no autograd/deep learning frameworks)

## File

- [`backprop_from_scratch.ipynb`](./backprop_from_scratch.ipynb) — full notebook with derivations, implementation, and numerical verification
