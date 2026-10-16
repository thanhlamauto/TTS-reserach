# Cache the native UNet prediction before CFG

Run ead9e469-0d34-49dc-9f9c-bd8506d28fff completed E0, including exact
latent/pixel/reward replay, all10 missing endpoints and terminal features.
R1 then failed before any method suffix: materializing the FP32 CFG epsilon on
CPU and feeding it back to XLA did not reproduce the fused original DDIM step
bitwise. The combined/fused expression can have a different rounding boundary.
That failed attempt is not R1 quality evidence.

The targeted repair caches the two native BF16 UNet outputs (unconditional and
conditional), copies the donor's exact pair, and computes CFG inside the same
transition graph as the original DDIM expression. No sampler coefficient,
noise, state, model weight, CFG scale or comparison tolerance changes. The
no-op and donor-reorder gates must still pass bitwise, with max errors saved.

Reuse only hash-pinned completed E0 files and the failed R1 attempt's actual
index40 prefix. No E0 proposal is regenerated; the960NFE prefix is not rerun.
The event prediction itself is regenerated under the corrected cache contract
and charged. The original first-launch deadline remains unchanged.
