# Data

No dataset is redistributed here. All datasets are public and keep their own licences; obtain them from their publishers. The preparation code for each lives in the experiment that uses it.

| Dataset | Reference | Used for | Preparation code |
|---|---|---|---|
| Objaverse-XL (Sketchfab subset) | Deitke et al., 2023 | Geometry pretraining: 61,822 / 1,844 / 1,825 meshes for train / validation / test, plus the 1,668- and 14,304-mesh cohorts | `experiments/01_pretraining/objaverse_base_corpus`, `objaverse_corpus_expansion`; `experiments/03_pretraining_targets/cohort_and_windsor_data` |
| WindsorML | Ashton et al., 2024a; Hugging Face `neashton/windsorml` | Pressure fine-tuning, 274 / 35 / 35 after geometry deduplication | `experiments/02_simulation_efficiency/windsor_drivaernet/source/prepare_data.py` |
| DrivAerML | Ashton et al., 2024b; Hugging Face `neashton/drivaerml` | Official 400 / 34 / 50 split; 420 / 16 / 48 reserved-car split | `experiments/08_expanded_benchmarks`, `experiments/09_reserved_cars` |
| DrivAerNet++ | Elrefaie et al., 2024 | 6,492 / 778 / 848 after deduplication, split by geometry hash | `experiments/08_expanded_benchmarks/existing_data_expansion/source/export_native.py` |
| DoubleDelta | Shen and Alonso, 2026; Hugging Face `yirens/double-delta-aero` | Wings at nine angles of attack, Mach 0.3; nine-angle, extrapolated-angle and held-out-angle splits | `experiments/06_sensor_completion`, `experiments/08_expanded_benchmarks`, `experiments/07_gabi_baseline/gabi_graph_ablation/prepare_graphs.py` |
| BlendedNet++ | Sung et al., 2025 | Unseen aircraft at Mach numbers above the training range | `experiments/08_expanded_benchmarks` |

Notes:
- **Revisions:** the exact dataset revisions, split files and case lists are recorded in each experiment's protocol files (for example `results/expanded_benchmarks/protocol.json` and the `split*.json` files in the experiment folders).
- **Reading raw files:** the original code read raw files through an object-store helper specific to the original infrastructure, which is not included. Supplying an equivalent `storage.py` that returns the same files from a local copy is enough to rerun preparation; see each experiment README.
- **Path layout:** the experiments expect prepared data under `/workspace/runs/<study>/datasets/` or `data/`, as each README states.
