# Mean-gradient noise avoidance vs frozen BFS at 2400 NFE

Fixed TRAIN pilot: six existing prompt texts, two fresh paired seeds2026101714/15.
Primary: best terminal ImageReward in each charged pool. Secondary top8 mean IR,
CLIP/DINO diversity on that same selector. No reserved final prompts.

Mean recipe:12×40 prefix +18×60 scout +3×60 guided +11×60 independent=2400.
Scout outputs are retained. Root0 is fixed, never reward-selected. Use mean A/B
finite-difference gradient, radius.05, delta.05, scale.1, gamma1, rho.25, floor.2,
epsilon0. Radius selected from earlier development, not a confirmatory choice.
Uniform control has identical scout information/allocation and paired suffix
noise, but samples q0. All fit costs are charged to both recipes.

Frozen BFS graph:24 particles×100=2400, Max+SSP, indices40/60/90,
tau8/8/32. Keep upstream zero-initialized history and event indexing. No retune.
Same prompt/seed/master initial stream, with12 roots for mean and24 for BFS;
not an equal-root-count claim. Original sampler microbatch4 is unchanged and
all executed batches divide by4:zero padding and zero auxiliary UNet evaluations.

Runtime baseline smoke checks full100/resume/BFS continuation bitwise on4 roots;
880NFE is reported separately. Twelve complete units require67120 actual NFE,
including replay and the allocation-matched control, after honest execution reuse.
Finish rectangle, no search until a win. Persistent40-minute cap.

Run through the inherited fixed ORX command on SSH alias otlab-dinov2-tpu:
`orx exp run 1b48932c-d09a-48d8-8791-211e05fa67ea --backend ssh --host otlab-dinov2-tpu --no-container`.
Never use the configured paid Modal default for this campaign. Exact source is
committed before launch. Full protocol and run output directory are in protocol.json.
No semantic-success claim from reward/diversity alone; this is exploratory TRAIN.
