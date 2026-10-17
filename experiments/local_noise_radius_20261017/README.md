# Bounded radius calibration, then one fresh-anchor screen

Parent9ff97dd passed all replay/norm checks but produced1/24 harmful mixed actions at r=.025 and algebraically flat filters. This child addresses the proposal's requirement for a calibrated fixed action radius. It does not search for a winning method.

Run both r=.05 and .10 on TRAIN prompts1/2/3 using the same initial seeds, basis and noise keys as the parent, paired across radius. All27 suffixes and4-root replay gates are recomputed and charged transparently; no uncharged scouting reuse. Parent noise-action and response-model implementations are imported unchanged, source-hash authenticated, and reuse only their compatible compilation cache.

Select the smallest radius with20–60% actual mixed-action degradation beyond−.05, at least10% actual improvement beyond+.05, and max actual state45 RMS change<=.15. Finish both levels, report both, never read filter performance for selection. If neither passes, stop; no third radius or delta adjustment. These are development headroom criteria, not a claim of optimal action strength.

If a radius qualifies, freeze its3-anchor empirical error allowances before screening TRAIN prompts4/5/6 with fresh initial prefix seed2026101713. Keep δ=.05, scale=.1, gamma1, rho.25, floor.2, both models and every stage1 gate unchanged. Previously used TRAIN prompt text is reused, so this is fresh-anchor/seed evidence, not unseen-prompt or final confirmation.

No semantic labels, final24, VALIDATION/TEST or external pool enter calibration. No actual guided batch, optimizer, end-to-end comparison, additional radii or automatic next stage are part of this node.

Existing TPU v5p-8,3 chip workers; inherited fixed run command dispatches the committed script. No dependency/backend change. Persistent1200s cap. If screen opens:243 terminal rows,21780 actual nominal sampling NFE,540 padding,22320 physical steps; diagnostic14940 plus replay6840. Parent14520 sampling remains a separately counted sunk pilot cost. If no radius qualifies:162 rows,14520 nominal,360 padding,14880 physical. No GitHub push.
