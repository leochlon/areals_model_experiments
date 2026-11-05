# Field-training harness

`field_benchmark.py` fixes the observation sets, fine-tunes a geometry checkpoint into a field model (`train-areal`), runs the baselines (nearest observed value, Gaussian processes, GABI), scores hidden-region nRMSE and integrated force error, and runs the hidden-value perturbation check. `python field_benchmark.py --help` lists the stages; the module docstring has a worked example.

The field studies used three versions of the file:

| Version | Used by |
|---|---|
| `v1/` | `09_reserved_cars`, `11_architecture_ablations` |
| `kit/` | `02_simulation_efficiency`, `04_masked_vs_unmasked`, `05_gaussian_fields`, `08_expanded_benchmarks` |
| `native/` | `06_sensor_completion`, `07_gabi_baseline`, `12_attention_analysis` |

`kit/` and `native/` differ in about 100 lines, `v1/` and the others in about 500. `scripts/setup_workspace.py` copies each version to the directories its studies import it from.
