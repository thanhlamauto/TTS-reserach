# Cross-process Diffusers metadata identity repair

First run49c411f4-ab39-4f82-b92e-2d09c3a7fd2b failed before any denoising.
The exact old/current saved scheduler configs differ only in the order of
`_use_default_values`, a set-derived list of keys supplied by defaults. The
old adapter included that list order in its schedule fingerprint.

The repair leaves the actual scheduler parameters unchanged. It requires
identical key sets and all non-metadata values, and identical default-key sets.
Only the ignored metadata list is registered in its original saved order.
The full old hash must then match the CURRENT100timesteps, all1000alphas,
final alpha, eta, prediction type and scheduler config. If it does not match,
execution fails. No unconditional hash alias is accepted. The14endpoint exact
replay gate remains mandatory. The failed attempt and cap clock are retained.

This is an implementation identity fix, not a new schedule, changed model or
relaxed numerical tolerance. A regression test rejects numeric parameter or
default-key-set changes.
