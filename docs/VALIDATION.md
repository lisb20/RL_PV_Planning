# Release validation — 2026-11-06

Validated on Python 3.12.4 with the versions in `requirements-tested.txt`. All training checks below are smoke checks, not claims of convergence or new paper results.

|Check|Result|
|---|---|
|Core unit tests|3 passed: terminal GAE, graph padding/action/value shapes and gradients, bundled data checksums|
|IEEE end-to-end training|2 updates, 2 trajectories × 2 steps per update; losses finite and artifacts saved|
|Beijing end-to-end training|1 update, 2 trajectories × 2 steps, 2 worker processes; 803-agent model updated and saved|
|Resume|One update plus a resumed second update matched uninterrupted 2-update actor/critic tensors exactly on CPU|
|Serial vs. parallel|1-worker and 2-worker IEEE runs matched actor/critic tensors and best deployment exactly on CPU|
|Actual-candidate selection|`best_plan.npy` matched its indexed `candidates/*.npz` vector exactly|
|Historical plan evaluation|Both bundled deployment vectors evaluated through complete 96-snapshot AC simulations|
|Historical policy inference|Both iteration-149 actor checkpoints completed 16-step trajectory sampling|
|Reference methods|Uniform, demand, 3-individual/1-generation GA, and A2C smoke runs completed|
|Package build/install|Built a wheel and installed it into an isolated environment sharing existing dependencies|
|Installed command outside checkout|`pv-evaluate` read packaged IEEE inputs and evaluated the example from `/tmp`|
|Learning curve|Plotting script produced a PNG from the 2-update IEEE training log|
|Syntax|All package, script, and test modules compiled|

Independent evaluation of `examples/ieee_plan.npy` through the installed package returned:

- PV generation: 193,649.471588 MWh.
- Ramping metric in the historical reporting convention: 146,141.959836 USD.
- Installed-area fraction: 0.511352251175.

These are example-deployment metrics, not values interpolated to a common PV-generation target. CPU reproducibility was checked in the same software/hardware environment; the checks do not assert bitwise agreement across GPU types or library versions.

Smoke outputs were written outside the repository. Long-running historical training and existing experiment directories were not modified by this release preparation.
