# Sinkhorn A mechanism probe on the existing SD1.5 TPU stack

Implements the supplied `TASK_README_VI.md` experiment; its original generic
`probe.py`/`summarize.py` helpers were not attached. No model/package upgrades.
Uses real cached DDIM100 eta1 checkpoints and the frozen epsilon0.1/0.3 A configs.

Stage0 reconstructs 48 original float32 OT operations, retaining unrefined mass
plans and normalized costs. Historical steered FP32 tensors must match exactly.
`X_after` is the BF16 sampler-entry tensor used directly by Stage1. Raw historical
mass plans were not saved, so reconstructed plans are explicitly labeled.

Stage1 uses the first12 existing manifest prompts, six methods, epsilon0.1,
checkpoint50, OT step0.3 and trust0.1. Two suffix replicates use the same bank and
prefix. All methods within each replicate use the same saved FP32 variance-noise
tape. Method is absent from its RNG key. The original custom BF16-model/FP32-DDIM
arithmetic remains unchanged. This is a new suffix-noise protocol, so none/A are
rerun; old suffix scores are never substituted.

Controls: none, A, mean_matched, minus_A, shuffle_matched, random_matched. Match
per-source RMS to A's actual post-cast delta. Fixed2% norm-fidelity threshold;
zero/nonzero mismatches stop rather than silently inserting a no-op control.
Shuffled control alone uses a float64 reference solver at1e-12 marginal tolerance.

Pool metrics use reference16 plus a single method's suffix8, separately for every
replicate. Ties prefer reference identities. Do not merge methods/replicates.
Average suffix replicates within each prompt before paired bootstrap20,000 times.

Fresh output namespace, `frozen_plan.json`, source/inherited SHA manifests, managed
job logs, real noise tapes, control latents, full metrics CSV and audit artifacts
are retained. Stage1 replicate1 is gated on successful completion/audit of all of
replicate0. A one-chip smoke includes exact latent/image/reward replay; its six
branch outputs are reused in the full diagnostic.

```bash
python -m unittest sd15_sinkhorn_probe.test_probe -v
python -m sd15_sinkhorn_probe.stage0 --output OUTPUT --prepare --readme TASK_README
python -m sd15_sinkhorn_probe.stage0 --output OUTPUT
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU python -m sd15_sinkhorn_probe.run --output OUTPUT --smoke
# Four chip-pinned workers; campaign coordinator opens replicate1 after rep0 audit.
python -m sd15_sinkhorn_probe.analyze --output OUTPUT
python -m sd15_sinkhorn_probe.audit --output OUTPUT
```

The full six-method/two-replicate campaign has57,600 fresh suffix NFE and1,152
final reward evaluations. Mandatory duplicate replay adds400 NFE/8 reward calls;
five runtime warmups add4,000 NFE. Cold inference cost of each A recipe is still
2,400 NFE; diagnostic cached-prefix costs are not deployed-method costs.
