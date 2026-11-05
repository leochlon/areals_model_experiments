# Experiments

One folder per experiment in the paper, in the order the paper presents them. Each folder holds the code that produced its results, with paths generalised: data preparation, training, evaluation and statistics. Each folder's README covers:
- the paper sections, tables and figures it supports;
- the steps to rerun it;
- the inputs it needs;
- the result files in `results/` behind each number.

| Folder | What it does | Paper |
|---|---|---|
| `01_pretraining` | Builds the 61,822-mesh Objaverse-XL corpus and pretrains the production prior (masked distance and normal-line reconstruction, 10,626 updates) | §3.2, App. E |
| `02_simulation_efficiency` | Fine-tunes the prior and the same architecture from scratch on 1/16 to all of WindsorML, DrivAerML and DrivAerNet++ (8,000 updates, three seeds) | Fig. 1, §4.2, App. C |
| `03_pretraining_targets` | Pretrains matched priors on 1,668 meshes (d, nn⊤; + Hessian; patch coefficients), a coefficient prior inside AREALS on 14,304 meshes, and fine-tunes each on WindsorML | Table 1, §4.3, App. M |
| `04_masked_vs_unmasked` | Pretrains three masked and three unmasked geometry seeds and transfers them to DoubleDelta and DrivAerML pressure | Table C.4 |
| `05_gaussian_fields` | Field adaptation on Gaussian random fields with exact posterior means, with a frozen or trainable trunk | Tables C.2, C.3, §4.4 |
| `06_sensor_completion` | DoubleDelta sensor completion with the condition unreported: references, scratch baseline, three fine-tuning seeds, explicit condition inference and held-out angles | Table 2, Tables D.1–D.6, §4.4 |
| `07_gabi_baseline` | GABI on the same DoubleDelta observation sets, with straight-line and along-mesh graphs | Tables D.3–D.5 |
| `08_expanded_benchmarks` | Pretrained vs. scratch sensor models on DoubleDelta at 18–19°, BlendedNet++, DrivAerML and DrivAerNet++ | Table 2, Tables C.1, C.5 |
| `09_reserved_cars` | Completion on the 48 reserved DrivAerML cars over 25 observation settings, with hidden-value checks | Table C.5, App. H |
| `10_frozen_surrogate` | Frozen areal tokens as the geometry input of the AeroJEPA tutorial surrogate | Table L.1, App. L |
| `11_architecture_ablations` | Decoder transfer vs. Point-MAE, coordinate interface, normal targets, head initialisation, density shift, finite tokens and resampling | Table E.1, Apps. F, G, §4.6 |
| `12_attention_analysis` | Attention range of pretrained, fine-tuned and scratch models and field error by region on 16 DoubleDelta wings | Table K.1, Fig. K.1, §4.5 |
| `13_distance_tubes` | Tubes of the decoder's distance: area, Betti numbers, paths, persistence and hidden-area coverage | Tables J.1, J.2, §4.6 |
| `14_symmetry_readouts` | Surfel and symmetry readouts of the package (evaluation outputs not included) | Table I.1, §3.2 |
| `15_figures` | Potential-flow example on a car mesh and the method figures | Figs. 2, B.1–B.3, G.1, K.1 |

Shared pieces:
- **Model package:** `src/areal_tokens` is used by every folder except `03_pretraining_targets`, which ships its own variant with sign-free normal inputs, and the complete-input arm of `13_distance_tubes`.
- **Field-training harness:** `shared/field_benchmark/` holds its three versions; see its README.

The hidden-value perturbation checks and independent rescoring of Appendix H live with the studies they check: `06_sensor_completion`, `07_gabi_baseline`, `09_reserved_cars` and `11_architecture_ablations`.
