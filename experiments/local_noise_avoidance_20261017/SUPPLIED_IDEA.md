## Refined idea: local, uncertainty-aware noise avoidance

I would refine the proposal into **a local proposal filter that learns which noise interventions are repeatedly unhelpful, while limiting how much it can suppress exploration**.

The important change is to stop treating the learned geometry as a set of “forbidden directions.” Instead, treat it as an imperfect prediction that must earn the right to influence sampling.

The uploaded proposal already establishes the useful foundation: paired interventions, a small action space, terminal-reward evaluation, and soft avoidance rather than hard projection. The refinements below add curvature to the response model, an explicit abstention rule, protection against accidental step-size shrinkage, and a more realistic compute allocation. These are proposed changes, not results established on your sampler. Pasted text(20261017-113205) Pasted text(20261017-113205)

The resulting method is:

\[
\boxed{
\text{Paired probes}
\;\rightarrow\;
\text{local response model}
\;\rightarrow\;
\text{evidence of degradation}
\;\rightarrow\;
\text{bounded reweighting at fixed action size}.
}
\]

---

## 1. Narrow the research claim

The original intuition is that recognizing harmful changes may be easier than identifying the best change. That is a useful hypothesis, but it should not be assumed.

I would make the research question more precise:

> **Can a small number of paired diffusion rollouts identify interventions whose suppression increases the number of good, distinct terminal images produced by subsequent rollouts?**

This distinguishes three things that otherwise become mixed together:

| Question | What would answer it? |
|---|---|
| Does the local response model predict degradation? | Held-out actions and future-noise tapes |
| Does filtering improve the next sampling batch? | Comparison against unfiltered sampling from the same checkpoint |
| Is learning the filter worth its cost? | End-to-end comparison at equal total compute |

A filter can pass the first test and fail the third. For example, it might predict degradation accurately but require so many scouting rollouts that ordinary sampling remains more efficient.

The source already identifies the key outcome as the contribution of **new candidates beyond the scouting pool**. I would make that the central objective rather than treating it as a secondary metric. Pasted text(20261017-113205)

One additional limitation should be explicit: **avoiding degradation is not the same as repairing a bad starting point**. A filter can preserve a consistently incorrect composition. Independent roots and absolute semantic-quality evaluation therefore remain necessary.

---

## 2. Keep the intervention simple, but normalize its actual strength

I would retain the proposed starting configuration: SD1.5, the existing DDIM100 setup, a checkpoint after 40 completed updates, and interventions in the next five injected-noise steps. Keep the action space four-dimensional. These are continuity choices from the note, not established optimal settings. Pasted text(20261017-113205)

The intervention remains:

\[
\xi_j(z)=\xi_j^0+B_jz,
\qquad z\in\mathbb R^4.
\]

However, I would strengthen the basis normalization.

### Equal action norms should mean equal injected-control energy

Your note correctly distinguishes the coordinate norm \(\|z\|\) from the magnitude of the actual injected change:

\[
b_{\rm inj}(z)^2
=
\sum_j \operatorname{RMS}(s_jB_jz)^2.
\]

That distinction matters when comparing directions: one basis vector should not look more destructive simply because it injects a larger perturbation. Pasted text(20261017-113205)

A concrete refinement is to orthonormalize the basis under this weighted metric.

For raw basis matrices \(\widetilde B_j\), define:

\[
G=
\sum_j
\frac{s_j^2}{n_j}
\widetilde B_j^\top\widetilde B_j,
\]

where \(n_j\) is the number of scalar noise entries at step \(j\). When \(G\) is well-conditioned, use:

\[
B_j=\widetilde B_jG^{-1/2}.
\]

Then:

\[
b_{\rm inj}(z)^2=\|z\|^2.
\]

This is only a small matrix calculation. It does not require differentiating through the denoiser.

Near-degenerate basis directions should be discarded rather than amplified through a nearly singular inverse. Also, this normalization equalizes the **injected control**, not terminal-image change; actual state differences and terminal diversity still need measurement.

---

## 3. Use the curvature information already purchased by the probes

The original proposal measures both directional and symmetric effects, but its main avoidance distribution uses only a linear prediction. I would compare that linear model against a **diagonal quadratic response model**.

This is the most useful modeling extension because it uses the same rollout data.

For each future-noise tape \(s\), retain:

\[
Y_{0,s},\qquad
Y_{\ell,s}^{+},\qquad
Y_{\ell,s}^{-}.
\]

Estimate:

\[
\hat g_{\ell,s}
=
\frac{Y_{\ell,s}^{+}-Y_{\ell,s}^{-}}{2r},
\]

and:

\[
\hat h_{\ell,s}
=
\frac{
Y_{\ell,s}^{+}+Y_{\ell,s}^{-}-2Y_{0,s}
}{r^2}.
\]

The proposed response predictor is:

\[
\boxed{
\hat m_s(z)
=
\hat g_s^\top z
+
\frac12\sum_{\ell=1}^{4}\hat h_{\ell,s}z_\ell^2.
}
\]

Here, \(\hat m_s(z)\) predicts reward change relative to no-op under tape \(s\). The finite differences build directly on the directional and symmetric components in the note. Pasted text(20261017-113205)

### Why this improves the formulation

A linear model can represent:

> Positive movement along this axis is harmful; negative movement is helpful.

A diagonal quadratic model can also represent:

> Large movement along this axis is harmful in either direction.

For example:

\[
\Delta R(z)=-z_1^2
\]

has zero gradient at the origin. A purely linear avoidance rule sees no harmful direction. The quadratic model can instead favor directions with a smaller \(z_1\) component—without incorrectly choosing one sign over the other.

### But do not fit a full Hessian yet

Four-dimensional curvature includes interactions between coordinates. Axis probes do not identify those interactions.

Consider:

\[
\Delta R(z)=10z_1z_2.
\]

Every coordinate-axis probe returns zero, yet mixed directions can be strongly beneficial or harmful.

Repeating the same axis probes on more noise tapes does not fix that missing geometry.

Therefore, the diagonal quadratic model is a **candidate approximation**, not an automatic upgrade. Keep the linear model as an ablation, and require mixed-direction validation before relying on either.

---

## 4. Make uncertainty weaken avoidance

This is the most important decision-rule refinement.

A tempting approach is to penalize actions with a poor lower-confidence estimate:

\[
\hat\mu(z)-\text{uncertainty}(z).
\]

That is inappropriate for the intended exploration philosophy. It makes unfamiliar actions look harmful precisely because they are unfamiliar.

Instead:

> **Suppress an action only when even an optimistic interpretation of the available evidence predicts meaningful degradation.**

Let:

\[
U(z)=\hat\mu(z)+b(z),
\]

where \(\hat\mu(z)\) is predicted reward change and \(b(z)\ge0\) is an uncertainty/error allowance.

Evidence for suppression exists when:

\[
\boxed{
U(z)<-\delta_R.
}
\]

Here \(\delta_R\) is the predeclared tolerance for a meaningful reward decrease.

An illustrative example:

| Predicted change | Error allowance | Optimistic score \(U\) | With \(\delta_R=0.05\) |
|---:|---:|---:|---|
| −0.20 | 0.03 | −0.17 | Suppress |
| −0.20 | 0.25 | +0.05 | Abstain |
| +0.02 | 0.03 | +0.05 | Do not suppress |

The second action is not declared good. There simply is not enough evidence to suppress it.

### A concrete two-tape implementation

For the initial two-tape setting, an implementable screening rule is:

\[
\boxed{
U(z)
=
\max\{\hat m_A(z),\hat m_B(z)\}
+
\epsilon_{\rm model}.
}
\]

This requires both tape-specific models to predict degradation, with an additional allowance for model error.

The allowance \(\epsilon_{\rm model}\) should be calibrated from prediction errors on separate development actions and anchors, then frozen before evaluation. It must not be chosen by inspecting the final test outcomes.

**This is an empirical agreement rule—not a statistically certified upper bound.** Two tapes cannot establish a reliable tail probability for unseen continuations, and the source already warns against interpreting that sample size as a 95% certificate. Pasted text(20261017-113205)

There is also a target distinction:

\[
\mathbb E_\omega[\Delta R(z,\omega)]<0
\]

does not imply:

\[
P_\omega(\Delta R(z,\omega)<-\delta_R)
\]

is large. An action can have a negative mean but occasionally produce a valuable alternative. The first prototype should report both observed mean effects and sign stability, without claiming it has learned the full distribution of harm.

---

## 5. Preserve action size and cap suppression

The original soft-avoidance distribution is preferable to hard projection, but it still has two weaknesses.

First, reweighting actions throughout a ball can favor smaller radii. The method could appear better simply because it intervenes less.

Second, an exponential penalty can become extremely concentrated even though its mathematical support remains nonzero.

I would address both directly.

### Start with same-radius exploration

For the first comparison, draw:

\[
z=ru,
\qquad \|u\|=1,
\]

using a fixed, calibrated \(r\).

The method now chooses **which direction to explore**, not whether to quietly reduce the step size.

This turns the note’s recommendation to control for radius into a property of the main experiment. Pasted text(20261017-113205)

Later, multiple radii can be introduced while preserving the original radial distribution: draw the radius first, then filter directions conditional on that radius. Do not redraw the radius after each rejection, because that would reintroduce radial bias.

### Use a bounded penalty

Define the evidence-supported degradation score:

\[
a(z)
=
\left[
\frac{-\delta_R-U(z)}{s_R}
\right]_+,
\]

where \(s_R\) is a fixed reward scale.

Then use:

\[
\boxed{
w(z)
=
\max\left\{
w_{\min},
\exp[-\gamma a(z)^2]
\right\}.
}
\]

Construct:

\[
q_{\rm avoid}(z)
=
\frac{q_0(z)w(z)}
{\mathbb E_{q_0}[w(Z)]},
\]

and retain the exploration mixture:

\[
\boxed{
q_{\rm next}
=
\rho q_0+(1-\rho)q_{\rm avoid}.
}
\]

An initial ablation setting could use \(\rho=0.25\), as suggested in the source, and a new suppression floor such as \(w_{\min}=0.2\). These would be experimental defaults, not optimized values. Pasted text(20261017-113205)

### What the cap actually guarantees

Because:

\[
w_{\min}\le w(z)\le1,
\]

the density ratio satisfies:

\[
\rho+(1-\rho)w_{\min}
\le
\frac{q_{\rm next}(z)}{q_0(z)}
\le
\rho+\frac{1-\rho}{w_{\min}}.
\]

For the illustrative values above:

\[
\boxed{
0.4
\le
\frac{q_{\rm next}(z)}{q_0(z)}
\le
4.
}
\]

Thus the filter cannot almost erase a locally supported region or assign it arbitrarily large relative density in a single batch.

This is a limited mathematical guarantee about the **local proposal distribution**, not proof of terminal-image diversity or global mode coverage.

Also, if every direction receives the same penalty, normalization gives \(q_{\rm avoid}=q_0\). The method has learned nothing useful for directional allocation. That should trigger fallback, not a claim of progress.

---

## 6. Freeze the anchor and reuse the model for a batch

The cost of learning a local model only makes sense when it influences enough later samples.

For the first algorithm, I would therefore use:

> **One exact checkpoint, one fixed noise-action basis, one scouting phase, and one subsequent sampling batch.**

Do not move the search center to the best scout. Do not carry the model to another timestep. Do not share it with nearby particles.

The source already restricts the model to the prompt, checkpoint state, action-block noise, basis, and trust radius. I would make that restriction literal in the implementation. Pasted text(20261017-113205)

All subsequent candidates restart from that same checkpoint. They use the frozen proposal distribution and fresh future-noise tapes drawn from the same continuation distribution used during calibration.

This separation matters:

**Reusing a model at an unchanged anchor** amortizes its cost.

**Moving to a new anchor while retaining the model** assumes transferability that has not been demonstrated.

For version one, do not update the predictor using the guided batch. A frozen batch makes the initial experiment easier to interpret and avoids introducing adaptive-data complications before the basic mechanism works.

---

## 7. Refine the diagnostic before building the optimizer

I would retain the original 27-rollout diagnostic, but change the held-out actions to make the geometry test more informative.

### Training: unchanged

At one checkpoint:

\[
\{0,\pm re_1,\ldots,\pm re_4\}
\times
\{\omega_A,\omega_B\}.
\]

That is 18 suffix rollouts.

Fit the linear and diagonal quadratic models from those same observations.

### Validation: use paired mixed directions

Generate four fresh, dense unit directions \(u_1,\ldots,u_4\), chosen independently of the probe rewards.

Evaluate:

\[
\{0,\pm ru_1,\ldots,\pm ru_4\}
\]

under a new tape \(\omega_C\).

That remains nine validation rollouts, but it tests combinations of coordinates rather than simply repeating the training geometry.

The cost remains:

\[
40+(18+9)\times60
=
1660
\]

sampling-NFE per checkpoint, or 4,980 across the three proposed diagnostic prompts, under the note’s accounting convention. Additional replay, reward evaluation, and operational costs still need reporting. Pasted text(20261017-113205)

### Evaluate the filter’s decision, not only regression error

The most useful diagnostics are:

| Diagnostic | Interpretation |
|---|---|
| Degradation among strongly suppressed actions | Does the filter recognize actual failures? |
| Useful actions receiving low weights | How much upside does it suppress? |
| Weighted versus unweighted degradation rate | Would the proposed sampling distribution improve this batch? |
| Actual perturbation and image differences | Is it winning through negligible changes or repetition? |

Compare against **shuffled weights within the same checkpoint and radius**, not just an unconstrained random baseline. Shuffling preserves the amount of reweighting while breaking its relationship to the predicted outcomes.

This directly asks whether the learned information is useful, rather than whether reducing exploration happens to help.

The three-prompt experiment remains a screening test. It is not enough to calibrate strong uncertainty claims, establish broad generalization, or select many hyperparameters. Mixed-action and fresh-tape generalization are also tested jointly here; a later crossed experiment can separate them when diagnosing failures.

Keep the reserved final evaluation set unopened, as the source specifies. Pasted text(20261017-113205)

---

## 8. Use a compute allocation that can actually fit

Under the note’s 2,400-NFE budget and eight-root setup, learning a separate four-dimensional model at every root is not feasible. The original accounting permits only 34 full suffix rollouts across those roots. Pasted text(20261017-113205)

A feasible **post-calibration efficiency test** is:

| Allocation | Sampling-NFE |
|---|---:|
| Eight prefixes to checkpoint 40 | \(8\times40=320\) |
| Complete seven roots without local search | \(7\times60=420\) |
| Eighteen scouting suffixes at one selected root | \(18\times60=1080\) |
| Nine post-scouting suffixes at that root | \(9\times60=540\) |
| **Total** | **2,360** |

The searched root should initially be selected by a fixed rule independent of unseen outcomes. Reward-based root selection would introduce another algorithmic component that needs its own comparison.

This schedule preserves seven independent roots while testing whether local avoidance helps at one anchor.

But it also makes the economic problem explicit:

> **Eighteen scouting rollouts influence only nine additional rollouts.**

That is a demanding return-on-compute requirement.

This deployment schedule does **not** include an additional nine-rollout validation phase at every searched root. Reliability and hyperparameters must already have been assessed in separate development experiments, whose costs should be reported separately.

Importantly, scouting is not pure overhead: its terminal images can be useful outputs. Every scout must remain eligible for the method’s output pool. The fair comparison is between complete pools generated with equal compute, not between guided samples and a baseline denied access to its own exploratory outputs. Pasted text(20261017-113205)

---

## 9. Separate mechanism comparisons from end-to-end comparisons

I would refine the baseline design into two levels.

### Mechanism comparison: equal information

Give the methods the same scouts, checkpoint, action-size distribution, and number of subsequent rollouts.

Compare unfiltered sampling, the original linear avoidance rule, the refined quadratic rule, and a positive-reward sampler using the same response estimate:

\[
q_{\rm positive}(z)
\propto
q_0(z)\exp[\beta\hat\mu(z)].
\]

Match exploration and concentration limits where practical.

This asks:

> Does suppressing credible degradation preserve more useful alternatives than explicitly favoring predicted improvement?

It is a stronger comparison than testing only against a deterministic gradient step.

### End-to-end comparison: equal resources, freedom to allocate them

Give random search, best-of-\(N\), and an appropriate active CMA-ES baseline the same total compute, but do not force them to purchase your 18-probe design.

A baseline that does not need a local response model should be allowed to spend those queries on additional outputs.

Active CMA-ES is particularly important because it already uses unsuccessful samples to reduce search variance. Therefore, learning from failure is not itself a novel contribution. [CMA Evolution Strategy](https://cma-es.github.io/cmaes_sourcecode_page.html)

Similarly, NSGD already avoids recently unsuccessful exploration directions through null-space search, and Noise Trajectory Search already performs gradient-free search over diffusion noise trajectories. [arXiv](https://arxiv.org/abs/1805.07317?utm_source=chatgpt.com)

The defensible potential contribution is the **combination and demonstrated benefit** of paired local measurements, curvature-aware degradation prediction, abstention under uncertainty, and bounded same-radius sampling—not any one ingredient in isolation.

---

## 10. Keep reward preservation separate from semantic preservation

The first implementation should use one clearly named target:

> Predict and avoid meaningful decreases in the selected terminal reward.

It should not call those decreases “semantic damage” unless independent evaluation supports that interpretation.

The source explicitly separates ImageReward differences, reliable semantic vetoes, and blind evaluation, and treats uncertain semantic judgments as abstentions rather than failures. I would preserve that separation. Pasted text

In particular, I would **not** immediately combine ImageReward and several uncertain semantic checks into a weighted scalar. That would add another calibration problem and make failures harder to interpret.

Initially, use semantic judgments to audit the filter’s decisions. Only add a semantic veto to the actual sampler after validating its ability to identify that specific error.

Noise-space constraints also do not solve reward misalignment. DNO explicitly identifies out-of-distribution reward hacking during noise optimization; limiting intervention magnitude addresses only part of that broader problem. [arXiv](https://arxiv.org/abs/2405.18881?utm_source=chatgpt.com)

The final outcome should continue to report:

\[
Q=\text{number of good images},
\qquad
D=\text{number of good-distinct images},
\]

with \(D/Q\) reported when \(Q>0\). Retain the existing quality rubric and distinctness definition rather than retuning them to favor the method. Pasted text(20261017-113205)

Most importantly, measure:

\[
\boxed{
D(\text{scouts}\cup\text{new samples})
-
D(\text{scouts}).
}
\]

A filter that raises average reward but adds no new useful images has not achieved the intended objective.

---

## The version I would build

**Keep the existing diffusion backend and four-dimensional noise block. Fit linear and diagonal quadratic response models from paired terminal rollouts. Suppress an action only when both tape-specific predictions indicate meaningful degradation after an error allowance. Apply capped soft weights at a fixed intervention radius, retain unfiltered exploration, and reuse the frozen model only at the exact original checkpoint.**

The acceptance criteria should then be sequential:

**First**, the predictions must distinguish harmful from useful mixed directions on fresh continuations better than matched random reweighting.

**Second**, the resulting sampler must add more good-distinct images than random or positive-guided sampling given the same scouting information.

**Third**, the complete method—including scouting—must outperform alternatives allowed to allocate the same total compute freely.

The refined idea is therefore not “learn where diffusion must never go.” It is:

> **Learn just enough about local degradation to avoid some wasted rollouts, while explicitly limiting the damage that an incorrect prediction can do to exploration.**