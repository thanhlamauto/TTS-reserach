# Restore the original CPU verifier batch contract for exact replay

Run09da2b9b-1c6d-4d37-9632-7abf777893c6 authenticated the old full schedule
hash. The14 replay latents matched old endpoints bitwise, but the combined
pixel/reward parity condition failed. That condition did not save separate
pixel and reward errors, so the failed run alone cannot isolate which caused
the failure. No missing proposal was sampled.

The original pilot scored8-image populations. The first follow-up gate had
packed8 no-ops plus6 selected edits into a14-image score call. The repair
restores BOTH original8-image populations and their original source order,
scored in two8-image calls. It includes2 duplicate no-op checks instead of
changing verification batch geometry. Gate denoising cost rises644->736 NFE;
the ten missing edits remain460 NFE. No tolerance is relaxed. The repair
also saves latent/pixel/reward parity separately before raising any error.

If exact parity still fails with the original call contract, stop before new
scientific outcomes. All failed attempts remain in the campaign namespace;
the first-launch3600-second wall clock is not reset.
