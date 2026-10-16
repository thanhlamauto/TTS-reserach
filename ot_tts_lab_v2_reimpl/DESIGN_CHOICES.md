# Reconstruction choices and preregistered initial scope

This is a new implementation from an incomplete, prose-only specification.
The missing release's numerical behavior and reported tests are not assumed.

Fixed first smoke: source8, reference16, SD1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, 512px, original DDIM100 eta1,
CFG7.5, BF16 state/UNet/terminal VAE, FP32 CFG/DDIM. Current-state clean
evaluation at completed-update index50. New TRAIN banks are required because
the located TRAIN m16_n8 banks use DDIM30 eta0. No VALIDATION, TEST or external
100-prompt pool is used by this package's preparation or synthetic demo.

DINOv3 ViT-B/16 global CLS features, L2 normalized, revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. Tensor-only bicubic antialiased
resize, center crop if required by the processor, checkpoint mean/std. Only
square RGB is supported initially. Pretrained preprocessing/gradient parity
requires the actual model assets and has not been verified.

Positive target: stable top ceil(M/2), selected rewards standardized within that
fixed bank subset, exp(beta*z) with beta1. Group-balanced mass is mixed 50/50
with quality mass. There is no negative repulsion. These details fill gaps in
the prose and are not tuned using terminal source outcomes.

Grouping: deterministic farthest-point initialization from atom0, up to4 groups,
20 normalized Lloyd updates, empty groups compressed. P1 clusters the actual
scored preview RGB. Grouping is a design choice, not an optimality guarantee.

P1: no-op, nearest positive reference, nearest positive reference in a different
group. Lowpass clean correction uses an orthonormal factor4 block basis.
Absolute actual x_t RMS bound .025. Active edits only, same source variance
noise, lookahead50->54. Preview reward below the same-source no-op is forbidden.
All feasible proxy gains use one common proxy-score standard deviation.
Capacity ceil(N/G), soft overflow penalty .1. No-op contributes to occupancy.
Exact DP and source-order greedy share proposals/previews/groups/quality guard.
Independent uses penalty0. Ties favor lower proposal index/no-op.

P2/P3: same clean optimizer, two normalized projected steps, each of size half
the clean trust radius in RMS units. No line search or promise of monotonic
loss. FP32 surrogate VAE and input gradients, encoder eval/frozen. Two-pass
feature VJP checks deterministic recomputation. Clean correction maps to x_t
with the actual signal scale, then post-BF16 RMS is enforced by bisection.
Feature mean shares target/trust/steps. Shuffle permutes target weights with
fixed seed20261017; no seed search. P2 epsilon is fixed once from initial
double-centered cross-cost standard deviation, kappa1, floor1e-6. Actual costs
are uncentered. P3 uses32 normalized Gaussian projections, seed20261016,
with exact overlap-quantile unequal-mass weighted1D transport.

Initial smoke: none; P1 OT/independent/greedy; P2/feature-mean/P2-shuffle;
P3/P3-shuffle. Do not promote a branch from a single prompt. A later 6-prompt
development campaign needs a locked manifest and all implementation gates;
there is no automatic reward-dependent expansion here.

Source terminal outcomes remain controller-only. No steering worker receives
them. Evaluations and bootstraps keep each method's own pool and average
bank/prefix/suffix replicates within prompt blocks. Bank wins reward ties.

Access gate on 2026-10-16: anonymous HEAD401; user-supplied token HEAD403,
both `GatedRepo`. The token was entered through a hidden prompt and not saved.
No model bypass, alternate encoder, paid GPU launch or TPU denoising was used.
