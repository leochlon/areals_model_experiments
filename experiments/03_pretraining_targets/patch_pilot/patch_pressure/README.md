# Patch encoder pressure fine-tuning — corrected protocol (v2)

This package attaches a pressure query decoder to the existing polynomial patch
encoder. It loads the `train_mesh_geometry.py` checkpoint format used by the
22,750-update model. The geometry architecture and its checkpoint format are
unchanged. It trains against actual CFD pressure samples and predicts Cp at
surface or volume queries.

**The main benchmark uses held-out cars, area weights, the reference pressure
standard deviation, and independent validation tuning for pretrained/scratch.**
A separate single-car diagnostic now defaults to spatial blocks with a gap.

## Install and check the actual device

```bash
pip install -r requirements.txt
pip install vtk    # for VTK input; unnecessary for NPZ input
python test_pressure.py --integration --device cuda
```

The device option reaches the neural checks, geometry pretraining, pressure
training, inference, and rate-search integration test. An explicit CUDA request
fails when CUDA is unavailable; it never silently becomes a CPU test. Use
`--device cpu` for CPU validation. If your CUDA backend requires nondeterministic
algorithms, add `--allow-nondeterministic`; evaluation and inference inherit this
setting from the checkpoint unless explicitly overridden.

## 1. Convert CFD samples with physical area weights

For point-associated surface pressure coefficient:

```bash
python prepare_pressure.py convert \
  --input car266_surface.vtp --pressure-key Cp --kind cp \
  --association point --measure area --out car266_pressure.npz
```

Point areas are computed by triangulating the surface and assigning one third
of each triangle's area to each vertex. Cell-associated values can use
`--association cell --measure area`. The converter uses native CFD values;
there is no polynomial field interpolation. VTK multiblock files require an
explicit `--block` flat index, with available indices listed on an error.

For NPZ point data, provide `xyz[N,3]`, the pressure array, and triangle indices:

```bash
python prepare_pressure.py convert \
  --input car266_export.npz --xyz-key xyz --triangles-key faces \
  --pressure-key Cp --kind cp --measure area --out car266_pressure.npz
```

If areas are already available, use `--weights-key vertex_area --measure area`.
For static pressure in Pa, use `--kind static --density RHO --speed U_INF
--reference-pressure P_INF`, substituting the actual CFD case values. For
kinematic pressure in m²/s² use `--kind kinematic`, with the reference pressure
in those same units. The target is

```text
static input:    Cp = (p - p_inf) / (0.5 * rho * U_inf**2)
kinematic input: Cp = (p_kin - p_kin_inf) / (0.5 * U_inf**2)
```

Coordinates must be in the mesh's original frame and units. `--xyz-scale` and
`--xyz-shift` explicitly transform them; area/volume weights scale with the
square/cube of the coordinate scale. Keep geometry mesh units consistent across
cars. Zero-measure samples are removed and counted in conversion metadata.

Area is the default measure. Volume-cell work can choose `--measure volume`.
An explicitly different sampling experiment can choose `--measure equal` and
train with `--weighting provided`. Existing v1 files with uniform weights must
be re-exported with areas for the area-weighted benchmark. The trainer rejects
files lacking area provenance under its default `--weighting area`.

## 2. Main experiment: held-out cars and independent rate searches

Use `cases.example.json` as the schema. Supply the actual paper's car IDs,
training budgets, validation/test splits and operating conditions. The example
values are placeholders, not recovered paper settings. Do not randomly split
one car's points and label that experiment held-out-car transfer.

Each entry names a unique CFD case `id`, a `car_id`, `split`, prepared `geometry`
NPZ/JSON, `field` NPZ, and `conditions`. Optional `mesh_index` selects one mesh
from a geometry corpus. Conditions follow `condition_names`; a common fixed
condition can use empty arrays. Paths are relative to the manifest. Each field
contains `xyz`, `cp`, positive `weights`, and metadata with `weight_measure`.
Cars, source meshes and CFD files cannot cross CFD splits.

Run the grid on both initialization arms:

```bash
python tune_pressure.py \
  --manifest cases.json --geometry-checkpoint geometry_pretrained.pt \
  --out tuned_comparison --seeds 17 \
  --encoder-lrs 2e-5 2e-4 1e-3 --decoder-lrs 2e-4 \
  --device cuda --steps 20000
```

Replace `17` with your experiment's actual seed list. Other training options
(e.g. `--queries`, `--patience`, `--encoder-chunk`) are forwarded to both arms.
The protocol fixes these before training:

1. The same candidate rate grid and seeds for both initialization arms.
2. The same maximum update budget, validation schedule, architecture and data.
3. Within each arm, the rate pair minimizing mean validation score across seeds.
4. Ties resolved by the first registered grid entry.
5. Test scoring only after **every** candidate finishes and selection is saved.

`protocol.json` is written first. `selection.json` records candidate validation
reports and independently selected rates. The comparison checks that the test
checkpoints are the actual validation-selected checkpoints. Unselected trials
receive no test scores. `summary.json` contains paired results per seed.
Resume an interrupted search with the same command plus `--resume`.

A scratch encoder can receive a different learning rate from the pretrained
encoder. A single-run trainer now defaults both encoder and decoder to `2e-4`;
these defaults are not a substitute for tuning. Different learning rates are
accepted by `compare`; a verified tuning claim additionally requires the saved
selection receipt.

Early stopping uses validation patience. `status.json` and the tuning summary
report whether runs stopped on patience or reached their update cap. A capped
run is never labeled converged. Increase the planned budget when needed.

For a fixed recipe, train and score separately:

```bash
python train_patch_pressure.py train \
  --manifest cases.json --geometry-checkpoint geometry_pretrained.pt \
  --out pressure_run --device cuda

python train_patch_pressure.py evaluate \
  --manifest cases.json --checkpoint pressure_run/best.pt \
  --split test --out pressure_run/test_metrics.json --device cuda
```

Training saves `validation_selected.json` and does not evaluate test data by
default. `--evaluate-test` is available only for a recipe fixed in advance;
the tuning runner forbids it. Pressure checkpoints contain the full fine-tuned
model. `--encoder-mode frozen` instead trains only the decoder.

## 3. Preserve the exact paper budgets and pretraining corpus

For the patch arm, prepare/pretrain on the same accepted geometry corpus used
in the intended comparison, using the bundled geometry script:

```bash
python train_mesh_geometry.py prepare \
  --manifest objaverse_meshes.json --out objaverse_patches.json
python train_mesh_geometry.py train \
  --data objaverse_patches.json --out objaverse_geometry --device cuda
```

Specify the intended model size, pretraining budget and preparation options.
Preparation reports rejected patches and coverage; check the accepted mesh IDs
across arms. The Windsor-only checkpoint remains a Windsor-only checkpoint.
These commands do not turn it into an Objaverse-pretrained model.

`run_pressure_study.py` accepts the exact budget/seed manifests rather than
inventing new splits. Fill `study.template.json` with your real checkpoint,
its prepared geometry cache, and `runs` entries of this form:

```json
{"budget": 16, "seed": 17, "manifest": "actual_splits/budget16_seed17.json"}
```

Those numbers illustrate the schema. Use the paper's actual budgets and seeds.
The budget counts unique labeled training cars. Every run must contain
explicit held-out validation/test cars. The geometry cache hash must match the
source checkpoint. The runner also fingerprints the set of geometry training
mesh hashes; an `expected_training_source_meshes_sha256` pins the required
corpus when you have the comparison corpus's fingerprint.

```bash
python run_pressure_study.py --study study.json --out paper_patch_arm --audit-only
python run_pressure_study.py --study study.json --out paper_patch_arm
```

This evaluates the **polynomial-patch encoder** and labels results accordingly.
Original AREALS and Hessian-model adapters are not present in this package.
Per-architecture gains are system comparisons. A target-only ablation also
requires a common backbone, decoder, masking, sampling and pretraining budget.
The current patch pretraining uses complete geometry; this is recorded in the
study audit and must not be relabeled masked AREALS pretraining.

The original paper PDF/run configurations are unavailable here. No Figure 2
split, Appendix E learning rate, Objaverse run, or original-AREALS result has
been fabricated or changed.

## 4. Single-car diagnostic: blocked regions and nearest neighbour

```bash
python prepare_pressure.py single-case \
  --geometry car266_patches.npz --field car266_pressure.npz \
  --car-id WindsorML_266 --condition-names --conditions \
  --split-method blocked --split-axis x --buffer 0.02 --out car266_blocked

python pressure_baseline.py \
  --manifest car266_blocked/cases.json --split test --out nearest_test.json
```

Blocked splitting uses contiguous end regions on the selected axis for
validation/test and the middle for training. Buffer strips are omitted from
training; `--buffer` is measured as a fraction of the mesh diagonal. The audit
checks actual nearest-training-point distances. `auto` chooses the axis with
the largest sampled extent. Empty regions are rejected. Duplicate coordinates
stay together. The output records held-out fractions, omitted samples, axis
boundaries and measured separation.

`--split-method random` retains an explicitly labeled interpolation diagnostic.
The nearest-neighbour baseline uses only training CFD pressures for that car
and operating condition; duplicate training coordinates receive their weighted
mean. It is evaluated at the same held-out samples, with the same area/std
metric. It cannot use a held-out car's labels as interpolation training data.
A spatially blocked single-car result is still a within-car result.

To train the car-266 decoder, use the fixed-recipe commands above with the
blocked manifest and the 22,750 checkpoint. For a single-car rate-search
**diagnostic**, tuning requires the explicit `--diagnostic-same-car` option.

## 5. Metrics and inference

The headline `case_mean_nrmse_reference_std` is the mean of per-case

```text
sqrt(sum(area * (Cp_pred - Cp_true)**2)
     / sum(area * (Cp_true - weighted_mean(Cp_true))**2))
```

Only the reference is centered in the denominator. Prediction bias remains in
the numerator. This implements the area/standard-deviation definition provided
in the review. Zero reference variance produces `null`; it cannot pass as zero
error. Constant-field diagnostics can choose `--selection-metric mse`.

Reports also retain Cp RMSE, MAE, bias, maximum error, the older reference-RMS
normalization, and training-standard-deviation normalization. They record the
weight measure and query indices. Physical pressure RMSE is reported when the
converter recorded the case's dynamic pressure. No operator-action error is
relabeled as pressure prediction error.

```bash
python train_patch_pressure.py predict \
  --geometry car266_patches.npz --checkpoint pressure_run/best.pt \
  --queries query_coordinates.npz --conditions \
  --out predicted_pressure.npz --device cuda
```

The query file needs only `xyz[N,3]`. Supply condition values in training order
if the condition vector was nonempty. Optional `--dynamic-pressure Q
--reference-pressure P` writes `pressure_predicted = P + Q*Cp_predicted`.
Inference needs no CFD labels, fitted reference coefficients or original
geometry checkpoint.

## Architecture and verification

Existing patch tokens and predicted polynomial coefficients feed the decoder,
with patch origin, frame, radius and mesh scale. Queries read nearby patches
and a global whole-car context. Conditions enter both. Queries never attend to
other queries, so inference is independent of query chunking. Encoder chunks
and activation checkpointing bound memory; every patch enters global context.
Fine-tuning minimizes area-weighted squared Cp error with training-only scaling.

Tests cover the metric's gauge invariance and retained bias, conservation of
lumped areas, VTK/NPZ conversion, area provenance, blocked-gap enforcement,
nearest-neighbour values, split leakage, checkpoint compatibility, gradients,
query/patch ordering, normal-sign invariance and reference-target isolation.

Integration performs **actual geometry-only pretraining**, then learns a field
whose spatial variation depends on geometry. It checks geometry swaps after
removing field means, resume, label-free inference, independent rate selection,
test isolation and rejection of a changed selection winner. These are
manufactured-field software checks, not Windsor accuracy or simulation-savings
results.
