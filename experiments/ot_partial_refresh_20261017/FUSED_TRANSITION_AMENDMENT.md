# Preserve the original sampler; charge the auxiliary preview explicitly

The native-BF16 prediction-cache attempt cdf70b4c failed the strict one-step
gate:10481/65536 BF16 state values differed, max absolute difference0.015625.
This is unacceptable as an exact replacement for the existing fused step.
The report does not claim to have repaired prediction reuse or to have proved
a particular compiler pass responsible. No method endpoint was sampled under
either failed cache implementation.

Use the original frozen `FrozenXlaAdapter.advance` unchanged after copying the
donor state. It computes the prediction of the ACTUAL copied donor state,
never the removed child's stale prediction. Proxy reward/feature at index40
uses the original `predict_clean`; its24 evaluations are auxiliary, not
silently counted as free. No model, sampler, DDIM coefficient, timestep, eta,
CFG, noise tape, allocation formula, gamma, survivor mask, donor pool or
rounding rule is retuned.

Each method's cold recipe is2400 sampling NFE +24 auxiliary NFE =2424 total,
with the same auxiliary feature/reward work for all four controls. The
original requested cached-prediction optimization remains FAILED/UNRESOLVED.
This run can verify the R1 allocator with that explicit compatibility overhead;
it cannot claim exactly2400 total UNet NFE or successful prediction reuse.

A new gate compares a4-particle original `Sampler.run` full100 trajectory and
its actual index40 checkpoint against the keyed prefix and no-op resumed
endpoint, bitwise. It costs400 additional gate NFE, recorded separately.
All18 unchanged survivors must still match across the four completed methods.
The frozen event remains after40 completed updates (next index40).

Completed E0 and the original24-particle index40 prefix are reused by hash.
No deadline is reset; no new backend, model or accelerator is introduced.
