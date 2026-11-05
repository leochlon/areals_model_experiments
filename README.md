# AREALS: code and results supplement

Code supplement for **Large Geometry Models Are Priors for Fast Physics Pretraining** (AREALS).

AREALS is a masked surface autoencoder pretrained on raw meshes to reconstruct the distance and the normal line, the two geometric quantities through which the flow equation sees a shape. It is then fine-tuned to complete surface fields from sparse sensors.

This repository contains:

- **`src/areal_tokens/`**: the model package used for the paper's production prior and every field study that starts from it, with its tests and model presets (`configs/`).
- **`experiments/`**: one folder per experiment in the paper, with the code that produced its results: data preparation, training, evaluation and statistics.
- **`results/`**: the stored result files that the paper's tables and figures are built from.
- **`paper_tables/`**: a module per experiment that rebuilds every table from `results/` and checks each printed number against the paper.

## Quick start

```bash
pip install -e ".[mesh]"          # Python >= 3.10; installs areal_tokens and the `areal` CLI
pip install pytest
python -m paper_tables --check    # rebuild all tables from results/ and check 1,740 printed numbers
pytest                            # package tests and the table checks
```

`python -m paper_tables` prints every rebuilt table as Markdown; `--only <module>` restricts it to one experiment.

## Paper results and where they come from

| Paper result | Experiment folder | Table module |
|---|---|---|
| Production geometry prior (Section 3.2, Appendix E) | [`experiments/01_pretraining`](experiments/01_pretraining) | `pretraining` |
| Simulation efficiency (Figure 1, Section 4.2, Appendix C) | [`experiments/02_simulation_efficiency`](experiments/02_simulation_efficiency) | `simulation_efficiency` |
| Matched pretraining targets (Table 1, Tables M.1–M.3, Section 4.3) | [`experiments/03_pretraining_targets`](experiments/03_pretraining_targets) | `pretraining_targets` |
| Masked vs. unmasked pretraining (Table C.4) | [`experiments/04_masked_vs_unmasked`](experiments/04_masked_vs_unmasked) | `masked_vs_unmasked` |
| Gaussian fields (Tables C.2, C.3) | [`experiments/05_gaussian_fields`](experiments/05_gaussian_fields) | `gaussian_fields` |
| Sensors in place of the condition: DoubleDelta (Table 2, Tables D.1, D.2, D.4, D.6, Figure D.1) | [`experiments/06_sensor_completion`](experiments/06_sensor_completion) | `sensor_completion` |
| GABI baseline (Tables D.3, D.5) | [`experiments/07_gabi_baseline`](experiments/07_gabi_baseline) | `gabi_baseline` |
| Expanded benchmarks (Table 2, Table C.1, Table C.5) | [`experiments/08_expanded_benchmarks`](experiments/08_expanded_benchmarks) | `expanded_benchmarks` |
| Reserved DrivAerML cars (Table C.5) | [`experiments/09_reserved_cars`](experiments/09_reserved_cars) | `reserved_cars` |
| Frozen areal tokens in a surrogate (Table L.1, Appendix L) | [`experiments/10_frozen_surrogate`](experiments/10_frozen_surrogate) | `frozen_surrogate` |
| Decoder transfer and design choices (Table E.1, Appendices F, G; Section 4.6) | [`experiments/11_architecture_ablations`](experiments/11_architecture_ablations) | `architecture_ablations` |
| Attention range and error by region (Table K.1, Figure K.1, Section 4.5) | [`experiments/12_attention_analysis`](experiments/12_attention_analysis) | `attention_analysis` |
| Distance tubes (Tables J.1, J.2) | [`experiments/13_distance_tubes`](experiments/13_distance_tubes) | `distance_tubes` |
| Surfel and symmetry readouts (Table I.1) | [`experiments/14_symmetry_readouts`](experiments/14_symmetry_readouts) | `symmetry_readouts` |
| Potential-flow example and method figures (Figures 2, B.1–B.3, G.1, K.1) | [`experiments/15_figures`](experiments/15_figures) | `figures` |

[`experiments/README.md`](experiments/README.md) gives a short description of each folder. Each folder's own README explains its contents and the steps to rerun it, and lists the result files behind each number.

## Layout

```
src/areal_tokens/      model package (encoder, query decoder, field encoder, training and evaluation CLI)
configs/               model presets (base: 3.2M parameters, 256 areal tokens; surfels; symmetry; tiny)
tests/                 package tests and the paper-table check
experiments/           one folder per experiment (the code behind each result, with paths generalised)
shared/field_benchmark/  the three versions of the field-training harness used by the field studies
results/               stored result files read by paper_tables/
paper_tables/          rebuilds the paper's tables from results/
scripts/               workspace setup for rerunning experiments
data/README.md         datasets and where to obtain them
REPRODUCIBILITY.md     compute, rerunning, and what is not included
```

## Rerunning experiments

Each experiment ran in a fixed directory under `/workspace`, and its scripts refer to those paths. The setup script recreates that layout from this repository:

```bash
python scripts/setup_workspace.py --python-env /path/to/venv   # copies studies, harness versions and the package into /workspace
```

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) describes the compute used, the inputs each experiment needs (datasets and upstream checkpoints), and what is not included.

## License

MIT; see [`LICENSE`](LICENSE). Third-party code included in `experiments/` (the GABI baseline, the AeroJEPA tutorial surrogate, Point-MAE and Utonia adapters) keeps its original license.
