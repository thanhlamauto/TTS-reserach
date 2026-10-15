# Theory and scope, written before any new neural pilot

This campaign seeks a reliable *offline selector of fixed Static-BFS policies*, not a test-time controller. A policy includes event indices, exact score multipliers, Max-history convention, resampling kernel, RNG convention and terminal selection. Let

\[
J(\pi)=\mathbb E_{x,s}[R(\pi;x,s)],\qquad \pi=A(D_{\rm profile}).
\]

A policy comparison estimates J(pi). A compiler comparison targets E_D[J(A(D))] and its variability across independently sampled profiling sets. One compilation, bootstrap perturbations of the same rows, or multiple copies of a report are not independent compilation replicates. CPU synthetic results below do not establish diffusion performance.

## Proposition 1: surrogate-selector decomposition

Let Pi be a candidate class, C subset Pi a frozen nonempty finite shortlist, and suppose the maxima exist (or use supremum limits). For all pi in C assume |J(pi)-g(pi)| <= e_model and |g(pi)-g_hat(pi)| <= e_est. Let the selector return pi_hat in C with g_hat(pi_hat) >= max_C g_hat - e_opt. Define pi_C* maximizing J on C and coverage_gap = J*_Pi - J*_C. Then

\[
J^*_{\Pi}-J(\hat\pi)\leq
\mathrm{coverage\_gap}+2e_{\rm model}+2e_{\rm est}+e_{\rm opt}.
\]

Proof: add and subtract J*_C. The remaining difference equals
J(pi_C*)-g(pi_C*) + g(pi_C*)-g_hat(pi_C*) + g_hat(pi_C*)-g_hat(pi_hat) + g_hat(pi_hat)-g(pi_hat) + g(pi_hat)-J(pi_hat).
The five summands are bounded above by e_model,e_est,e_opt,e_est,e_model respectively. Adding coverage_gap proves the claim. No distributional assumption is needed for this deterministic implication; establishing the uniform hypotheses is a separate statistical problem. A few residuals or a good fit on selected paths do not establish them. Coverage_gap remains UNKNOWN on real data.

For a full-policy selector with a valid shortlist certificate J*_C-J(pi_hat)<=epsilon_sel, the correct decomposition is coverage_gap+epsilon_sel. Graph approximation errors may affect shortlist coverage, but are not added a second time to this full-policy certificate.

## Proposition 2: finite multiplier sensitivity

Fix one population of N finite scalar scores r_i and approximations rhat_i, with max_i|rhat_i-r_i|<=eta and finite beta>=0. Define w_i=exp(beta r_i)/Z and what_i=exp(beta rhat_i)/Zhat. Each exponential numerator ratio lies in [exp(-beta eta),exp(beta eta)]. Summing positive terms gives the same bounds for Zhat/Z. Multiplication by Z/Zhat gives

\[
 e^{-2\beta\eta}\leq \hat w_i/w_i\leq e^{2\beta\eta}.
\]

At beta=0 both distributions are uniform. For large beta use log ratios to avoid numerical underflow. This proposition does NOT cover beta=infinity. Here r is a hypothetical target score on the *same population*; it is not automatically terminal continuation value. The implemented SD1.5 score is a path maximum of beta_t times intermediate ImageReward, initialized at zero and inherited through resampling. To apply the proposition to this historical score, a bound on the resulting effective-score errors is needed; errors can also change ancestry and future populations. Thus no terminal-reward or schedule-optimality guarantee follows. Temperature-by-uncertainty is an empirical, falsifiable heuristic unless those target and error assumptions are justified.

## Proposition 3: simultaneous pair bounds imply shortlist optimality

Suppose a frozen finite shortlist C has simultaneous intervals [L(rho,pi),U(rho,pi)] covering every J(rho)-J(pi) at every predeclared look with family probability at least1-alpha. If a returned pi_hat satisfies max_rho U(rho,pi_hat)<=epsilon, then J*_C-J(pi_hat)<=epsilon on this event. Proof: choose a J-maximizer rho in C and use its covered upper endpoint. This also handles a data-dependent chosen pi_hat because all pairs and looks were covered simultaneously. Superiority versus a baseline requires a *lower* bound above the registered margin; epsilon-optimality is a different statement.

One concrete construction for bounded planning rewards: average paired seeds within each independent task block, so D_task in[-1,1]. At a fixed task count n, Hoeffding gives two-sided radius sqrt(2 log(2ML/alpha)/n), for M unordered pair contrasts and L predeclared looks. A union bound controls those finite looks even with dependence between policies within a task. Blocks must be independently sampled from the frozen task distribution; tasks sharing a map do not establish generalization to other maps. A predictable block-allocation scheme must not select tasks using their yet-unobserved outcomes. Applying fixed-n intervals to arbitrary outcome-dependent subsets is not justified. Adaptive sample sizes beyond the registered finite looks need a valid confidence-sequence argument with verified assumptions.

ImageReward has no verified bounded/sub-Gaussian support here. We do not clip it or apply the bounded theorem to it. Historical cluster bootstrap intervals are descriptive approximations, not finite-sample or anytime-valid certificates. Confirmation will remain fixed-sample if approximate inference is all that is supported.

## Block covariance and precision planning (assumptions, not a theorem about IR tails)

For paired differences D_ps = mu + a_p + e_ps, assume exchangeable prompt effects and conditional seed residuals, independent across prompts, with variances B and W. Then Var(mean_{P,S} D)=B/P+W/(PS). Estimate W by mean within-prompt seed variance, and B_raw=sampleVar(seed means)-W/S. Retain negative B_raw in reports; use max(0,B_raw) only for a nonnegative design forecast. Covariance between policies is measured rather than assumed beneficial. Shared graph edge vectors must remain in a common prompt-seed block; for edge coefficient vector v, Var(v' delta)=v'Cov(delta)v, not sum of independent-edge variances.

At a fixed trajectory budget PS, the W term is unchanged by reallocating seeds to prompts while the B term falls as P grows. Seed replication is still needed to estimate W, diagnose seed generalization and make an appropriate seed bundle. Per-prompt setup cost, dependence and uncertain moment estimates can change this allocation conclusion. The reported80%-power planning numbers use a normal approximation with alpha/3, not a promise of power.

## Primary-source audit and limits of transfer

- [Classical-search diffusion paper, v2](https://arxiv.org/html/2505.23614v2), sections4.1.1 and4.2, AppendixC/D: BFS uses score multipliers, history-dependent scoring and a specified resampling kernel. Local Langevin convergence does not prove finite-budget static schedule selection. Source implementation is pinned separately; the prior TPU adapter uses Max+SSP, while PointMaze keeps original rounded offspring and TFG.
- [Lalitha et al., UAI2023](https://proceedings.mlr.press/v216/lalitha23a/lalitha23a.pdf), sections2/3: heterogeneous-variance fixed-budget identification assumes specific arm reward models; the unknown-variance analysis imposes Gaussian assumptions. Shared-tree, clustered, data-selected IR observations do not automatically satisfy these hypotheses. No BAI rate is imported here.
- [Howard et al., confidence sequences](https://arxiv.org/html/1810.08240v5), Definition1/Theorem4: valid time-uniform inference rests on martingale/moment or boundedness conditions. Repeated ordinary bootstrap is not a substitute.
- [Del Moral, Doucet, Jasra, adaptive SMC](https://arxiv.org/pdf/1203.0464), sections3/4: results concern specified Feynman-Kac flows, potentials and kernels. We have not identified our terminal IR objective with that target. ESS and ancestry are diagnostics, not quality objectives.
- [PyTorch/XLA v2.4 troubleshooting](https://github.com/pytorch/xla/blob/v2.4.0/TROUBLESHOOTING.md): use installed-version metrics, identify host fallbacks and recompilation, and synchronize execution for timing. No upgrade is required or authorized.

These are scope summaries, not claims of a new theorem beyond the elementary propositions above. Synthetic tests use exact known Gaussian variances only in the toy setting, retain shared-block covariance, test additive/high-noise, interaction and missing-optimum cases, and explicitly separate within-shortlist certification from unknown real coverage.
