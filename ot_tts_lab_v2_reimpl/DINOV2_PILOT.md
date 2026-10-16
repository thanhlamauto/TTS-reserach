# DINOv2 pilot amendment — user authorized 2026-10-16

The user explicitly requested DINOv2 in place of the gated DINOv3 to run the
experiment first. This is a separate pilot, not a DINOv3 reproduction.

Encoder: `facebook/dinov2-base`, revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`; global normalized CLS, ViT-B/14,
FP32. Checkpoint preprocessing: bicubic antialiased square resize256,
center-crop224, ImageNet normalization. The effective 224 crop is divisible
by patch14. Tensor-only preprocessing retains the VAE input gradient.

Everything else is inherited from DESIGN_CHOICES.md: SD1.5 pinned revision,
DDIM100 eta1 CFG7.5, 512px, source8/reference16, current prediction index50,
bound .025, factor4, two projected gradient steps, P1 lookahead4/capacity
penalty.1, P2 kappa1, P3 slices32 and fixed seeds, nine configurations with
their controls. No reward-based retuning is allowed in the smoke.

Run order: asset/environment checks -> new TRAIN bank on the original sampler
-> actual VAE/encoder gradient preflight -> trusted original endpoint replay
and duplicate replay -> nine paired methods -> terminal evaluation and integrity.
Only the first existing TRAIN prompt and bank seed20261016 are initially used.
No VALIDATION/TEST/external pool. A single prompt cannot nominate a winner.

The CPU worker is isolated from the frozen Torch2.4/XLA2.4 sampler. Only one
TPU chip and one CPU encoder worker are used initially. Technical wall cap
is3600 seconds for the neural smoke; whole allocated slice occupancy (four
chips) is reported separately from active-chip time. This is not an estimate
of actual Cloud billing. Environment installation has no accelerator access.

Abort on wrong model identity, nonfinite gradient/objective/state, failed
finite difference/replay, stale input/source hashes, or post-cast trust failure.
Implementation repairs are recorded and do not change scientific parameters.
