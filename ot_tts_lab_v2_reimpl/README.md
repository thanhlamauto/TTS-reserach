# OT-TTS: independent reconstruction from the supplied v2 specification

This source was written on 2026-10-16 because the original `ot_tts_lab_v2`
package was unavailable. It is **not** that package or its claimed 41-test
release. Its version is `0.1.0-reimplementation`. No pretrained-quality result
is implied by unit tests or the synthetic demo.

Implemented: KL-regularized log-domain FP64 Sinkhorn and envelope gradient,
both self derivatives of Sinkhorn divergence, weighted unequal-mass/unequal-count
1D transport, fixed-projection sliced W2, deterministic grouping/positive
target, exact small capacitated assignment, P1 proposals and paired lookahead,
P2/P3 two-pass feature VJP, orthonormal block projection and post-BF16 trust
projection. The nine-configuration runner includes no-steering, P1 independent
and greedy, feature mean, and shuffled-target controls.

The file API rejects source terminal outcomes and stale clean predictions.
Noise is keyed only by source and step. Plans lock input hashes; caches lock
code, execution identity, checkpoint, noise and relevant configuration. Separate
method pools, bank-first ties and prompt-block bootstrap prevent selection and
analysis leakage. Cache savings are not cold-speedup evidence.

## Test and synthetic integration

Use a separate environment. Do not upgrade the working TPU environment.

```bash
cd /absolute/path/to/TTS-reserach/ot_tts_lab_v2_reimpl
export PYTHONPATH="$PWD"
python -m pytest -q
python -m otlab.cli demo --out /absolute/path/to/a/new/demo_directory
```

The demo runs 2 synthetic prompts x 2 paired suffix tapes x 9 configurations.
It includes duplicate replay and a separately driven trusted synthetic endpoint.
It does not load SD1.5 or DINOv3 and does not demonstrate image improvement.

## Real setup and order

1. Check access to the exact DINOv3 revision. `model_access` reads a token with
   terminal echo disabled; it does not save it or change account settings.

   ```bash
   python -m otlab.model_access --token-prompt --out /new/run/access.json
   ```

   If access is granted, an explicit `--download --cache-dir /separate/cache`
   downloads only the pinned encoder files. It never downloads another SD1.5.
   Access/license approval happens on the official model page, not in this code.

2. Install `requirements-worker.txt` into a **new** CPU worker venv. The existing
   XLA venv stays unchanged. Point `configs/worker.yaml` to the existing SD1.5
   snapshot and the exact encoder revision/cache. FP32 surrogate VAE is an
   intentional difference from the BF16 terminal VAE; identity/scaling match.
   Start with one worker and feature microbatch1.

3. Use the historical `Sampler` to create a **new TRAIN** bank16/source8 with
   DDIM100 eta1 CFG7.5 512px. Existing 30-step eta0 TRAIN banks are ineligible.
   Keep the complete historical `bank.json` and trajectories. `export_bank`
   checks the model/schedule/counts and evaluates clean predictions from the
   actual index50 states. It exports a closed worker checkpoint, the original
   source-specific FP32 suffix tape, and a separate controller-only trusted
   endpoint. It never relabels an intermediate state or supplies source
   terminal rewards/images to steering.

4. Supply an adapter YAML with `legacy_pythonpath`, exact `legacy_source_hashes`,
   `runtime_output`, `dcs_root`, and the frozen `sampler` configuration. Use
   `otlab.adapter:frozen_xla_factory`. This loads the original model/scheduler
   helpers and adds `advance`, `predict_clean`, and `score`; it does not replace
   the backend. It requires an actual neural replay before claims of equivalence.
   Use a separate XLA cache on tmpfs when disk is limited.

5. Run actual VAE/DINO gradient preflight from a closed checkpoint using the
   worker's `preflight` operation. Verify repeatability, finite nonzero gradient
   and directional finite difference. Inspect the decoded clean image before
   the neural smoke. The pretrained preflight is not yet passed.

6. Create a smoke plan, duplicate/trusted replay, then the nine methods:

   ```bash
   python -m otlab.cli plan --manifest /data/export/manifest.jsonl --out /new/plan
   python -m otlab.cli replay --plan /new/plan/plan.json \
     --adapter otlab.adapter:frozen_xla_factory --adapter-config /data/adapter.yaml \
     --out /new/replay.json
   python -m otlab.cli run --plan /new/plan/plan.json \
     --adapter otlab.adapter:frozen_xla_factory --adapter-config /data/adapter.yaml \
     --worker-python /new/worker/bin/python --worker-config configs/worker.yaml \
     --replay-proof /new/replay.json --out /new/results --cache /new/cache
   ```

   These commands are scientific entrypoints. For remotely tracked experiments,
   invoke the fixed committed campaign command through `orx exp run` on the
   explicit SSH backend, after model/preflight/budget gates. The default Modal
   backend must not be used accidentally. No remote experiment was launched by
   preparing this package.

## What remains and what is intentionally not claimed

- The original release's behavior cannot be recovered exactly from prose. New
  engineering choices are documented in `DESIGN_CHOICES.md`.
- Actual SD1.5/XLA model replay and VAE/DINO gradient preflight must pass. CPU
  mock equivalence tests are not neural-backend parity.
- Source DINO diversity is available in the demo. The real runner currently
  saves terminal tensors/RGB/rewards for a separate terminal encoder evaluation;
  it **does not** silently reuse intermediate bank features for terminal metrics.
- Existing CLIP metrics and independent DreamSim/PickScore/fidelity evaluation
  still need to be connected before a quality/diversity claim. No thresholds
  for `N_good_distinct` are invented. The runner does not report that metric.
- The requested 6-prompt development sweep, stage0 selector-only baseline,
  later geometry/timing/target sweeps and fresh confirmation are **not yet run**
  or automatically selected. This reconstruction implements the initial smoke,
  not every command of the missing release.
- One prompt is an engineering smoke, not a method winner. P1 must beat both
  independent and greedy on the same proposals; P2/P3 must beat feature mean
  and their shuffled control on the same encoder/trust/steps/target.
- Measured sample/call counters are recorded separately from nominal cold
  recipe NFE. Padding, compile, replay and CPU feature/VAE work are separate.
  When initial auxiliary NFE is unknown the exact cold count is null. Worker
  costs cannot be inferred from UNet NFE or cache-hit latency.

## Primary references checked

- [Official DINOv3 ViT-B/16 model and gated access](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [Transformers 4.56.2 DINOv3 features and preprocessing API](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/dinov3)
- [Official DINOv3 implementation](https://github.com/facebookresearch/dinov3)

The four user-supplied files are preserved with hashes under `specification/`.
