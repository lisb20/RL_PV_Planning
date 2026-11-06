# Simulation data

Paths below are relative to `pv_planning/data/`.

|File|Use / source|
|---|---|
|`case300.mat`|IEEE 300-bus case distributed in the original release; derived from MATPOWER's `case300.m`|
|`power_2class.mat`|Residential and industrial 48-sample demand templates, interpolated to 96 samples|
|`solarbej.npy`|24-sample representative PV profile, interpolated and normalized to 96 samples|
|`config.yaml`|Original IEEE release configuration, retained for provenance|
|`beijing/config.yaml`|Synthetic Beijing electrical model settings|
|`beijing/grid.json.gz`|Upper-level synthetic Beijing network used by the original training entry point|
|`beijing/power_map0221.npy`|803 × 96 representative demand matrix|
|`beijing/roof_areas.json`|PV-suitable area by 110-kV consumer facility ID, converted from the original pickle|
|`manifest.json`|SHA-256 checksums of all bundled inputs|

MATPOWER case source: https://github.com/MATPOWER/matpower/blob/master/data/case300.m

The Beijing input was taken from the pre-existing `Release-bej` dataset. As in its training script, lines whose endpoints have facility prefixes below 5 and points with prefixes below 5 are retained (2,675 records). This removes unused lower-level geometry. Compression is lossless. The demand matrix is unchanged. Roof areas are converted to JSON without changing the numeric values. Facility prefix constants are included locally, removing the runtime dependency on the internal city-protocol package.

The Beijing topology is synthetic and spatially grounded. This release bundles simulator-ready aggregates, not the complete raw GABLE building collection or city data-preprocessing pipeline. The representative PV and load templates do not contain a calendar or establish an annual seasonal dataset.

Original source locations used to prepare this release:

- IEEE environment/model: `Release/IEEE_env.py`, `Release/model.py`.
- Beijing environment/network: `Release-bej/RL_env.py`, `Release-bej/flow.py`, `Release-bej/util.py`.
- Beijing area mapping: `BEJ/flow/trans110_2_area.pkl`.
- Historical deployment examples: the May 19 IEEE and May 21 18:52:50 Beijing folders recorded in `examples/provenance.json`.

Existing upstream data and software terms continue to apply. This repository update does not relicense third-party materials.
