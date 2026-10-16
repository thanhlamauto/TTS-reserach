# ImageReward TPU correctness repair

The first actual gate (ORX38d0ec3a-6ca1-4cee-bba0-881acff99fe7,
commit7184f12) failed after34.04 worker/supervisor seconds. On8 saved images,
maximum absolute TPU-versus-CPU error was.0073502064, exceeding the fixed
atol=rtol=.001. All4 within-prompt argmax comparisons matched. No neural host
fallback was reported, and no scientific policy comparison followed this failure.

FP32 weights do not alone force full-precision TPU matrix products. The installed
PyTorch/XLA2.4 binding `_xla_set_use_full_mat_mul_precision(True)` maps to
`xla::PrecisionConfig::HIGHEST`; this was checked against release2.4 source
`torch_xla/csrc/init_python_bindings.cpp`, lines1907–1913 and the installed binding.
The repair explicitly requests this precision for the verifier. It does not change
weights, preprocessing, prompt/image inputs, scores, tolerance or reward objective.
The existing BF16 diffusion stack remains unchanged; later integration must scope
this setting around completed verifier operations, restoring the denoiser setting.

Retry uses the same persistent campaign clock and cost ledger, not a fresh budget.
Per-attempt outputs are separated by the immutable lease job ID. The failed
first gate remains archived under its original ORX run and campaign attempt folder.
Successful numerical parity is still required before any scientific experiment.
