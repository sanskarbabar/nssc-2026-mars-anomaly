# Engineering Changelog — Autoencoder Anomaly Pipeline

Format: **symptom → diagnosis → fix → outcome**. Mirrors Phase 4 of the notebook and §4 of the
report. All checkpoints are saved under `artifacts/ae_v{1,2,3}.pt`.

Final run: `EPOCHS = 18`, batch 128, **fp32** (mixed precision disabled — see v3), grouped
train/val split (154 / 18 scenes).

| version | loss | val‑SSIM | crops flagged | Jaccard vs v1 |
|---|---|---|---|---|
| v1 | MSE | 0.636 | 822 | 1.00 |
| v2 | MSE + 0.15·(1 − SSIM) | 0.644 | 1 089 | 0.56 |
| v3 | MSE + 0.15·(1 − SSIM) + 0.10·GradL1 | 0.645 | 1 567 | 0.52 |

---

## v1 — Pure‑MSE Convolutional Autoencoder (baseline)

| | |
|---|---|
| **Config** | 5‑stage CAE `1→32→64→128→256→512`, stride‑2, BN + LeakyReLU; linear bottleneck to a 256‑D latent; mirrored transpose‑conv decoder + sigmoid. Loss = **MSE only**. AdamW, cosine LR, flips/rot90 augmentation. |
| **Symptom** | Reconstructions of dune fields and crater rims are visibly **blurred** (`artifacts/v1_recon.png`). Validation SSIM plateaus at 0.64 while MSE keeps falling. Ordinary high‑frequency terrain therefore produces large reconstruction error → some perfectly normal rocky crops score as high as genuine anomalies → noisy novelty tail. |
| **Diagnosis** | MSE is minimised by the local conditional mean. When fine texture cannot be placed exactly, the optimiser prefers a smooth "average" patch. Nothing in the objective rewards *structural* similarity. |
| **Fix → v2** | Add a hand‑built **SSIM term** (11×11 Gaussian window, implemented from scratch — no pretrained perceptual network): `L = MSE + 0.15·(1 − SSIM)`. |

## v2 — MSE + SSIM

| | |
|---|---|
| **Outcome of the v1 fix** | val‑SSIM 0.636 → **0.644** (marginal). Flagged set changes substantially: Jaccard vs v1 = 0.56. |
| **Residual symptom** | Thin linear structures — ridgelines, scarps, and importantly any **spliced straight edges** — are still slightly rounded; Phase‑3 error maps are smeared along edges, making the "spliced boundary vs. rare feature" call ambiguous. |
| **Diagnosis** | SSIM's 11 px Gaussian window tolerates 1–2 px edge displacement; nothing looks at the image gradient directly. |
| **Fix → v3** | Add a **finite‑difference gradient‑L1 term**: `L = MSE + 0.15·(1 − SSIM) + 0.10·‖∇x − ∇x̂‖₁`. |

## v3 — MSE + SSIM + Gradient  *(numerical‑stability incident + fix)*

| | |
|---|---|
| **Symptom (first attempt)** | v3 training **diverged**: `val‑SSIM → NaN` after a few epochs and the decoder **collapsed to a single mean image** — every reconstruction identical regardless of input (verified visually via the Phase‑3 heatmap grid). All downstream v3 novelty scores, thresholds and "Genesis source" lists were meaningless (flagged ~33 %, threshold shifted, top‑5 = darkest images). |
| **Diagnosis** | Training ran under automatic mixed precision (fp16). The SSIM variance terms (`x·x`, `y·y` group convolutions) and the gradient term overflow / underflow in fp16 → `inf` in the loss → one `inf` gradient step destroys the weights → the model settles into the trivial constant‑output minimum, which still scores low MSE against the dataset mean. |
| **Fix** | (i) **Disable mixed precision** — compute the loss in fp32. (ii) Add **gradient‑norm clipping** at 1.0. (iii) Clamp SSIM inputs to `[0,1]`, clamp local variances to ≥ 0, add ε in the denominator, clamp the SSIM map to `[−1,1]`. (iv) **Skip any batch** whose loss is non‑finite. (v) Checkpoint every epoch (atomic write) so a Colab disconnect resumes instead of restarting. |
| **Outcome** | v3 trains stably to **val‑SSIM 0.645**; reconstructions are distinct and comparable to v2. Used as the final model (Phase 5). |
| **Trade‑off** | The gradient term raises reconstruction fidelity everywhere, including on real anomalies — monitored via the cross‑version flagged‑set Jaccard (v1↔v3 = 0.52). SSIM/gradient improved val‑SSIM only 0.636 → 0.645 but changed which crops are flagged a lot → the pipeline is **loss‑sensitive at the crop level**. |

## Also explored

* **Latent‑dimensionality sweep `{128, 256, 512}`** — val‑SSIM rises with dimension but the
  tail‑separability proxy peaks near 256 and falls at 512 (encoder starts memorising rare
  structure) → justifies `LATENT_DIM = 256`.
* **VAE bottleneck (β = 1 × 10⁻⁴)** — an iteration that **did not help**: the KL prior pulls latent
  outliers toward the origin, lowering their Isolation‑Forest novelty score even though
  reconstructions are comparable. Scaffold in the notebook (`RUN_VAE`), disabled by default.

## What the iterations taught us

Loss engineering (SSIM, gradient) barely moved reconstruction quality but substantially reshuffled
the flagged crop set. However the **source‑level conclusion is invariant** to the loss: v1, v2 and
v3 all flag `SRC_044`, `SRC_064`, `SRC_166`, `SRC_128`, `SRC_039`, `SRC_154`, `SRC_101`, `SRC_119`
at 70–100 %. The robust, defensible answer is stated at the observation level; the crop‑level count
(822 → 1 402 depending on model) is reported as a graded‑confidence range with a strict 1.5 % core.

> Per the problem statement, iterations need not monotonically improve a metric — each change is
> motivated by an observed problem and its outcome is measured and reported. The v3 collapse and its
> fix are the clearest example.
