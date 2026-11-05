# geometry-latents

Machine-learning surrogates for CFD take a shape and predict what the flow does to it: surface pressure, shear, forces. Each has to be told what the shape is, and the usual ways are poor. Feeding raw points makes the field model learn geometry from its only signal, field labels, one expensive simulation per example. Squeezing the shape into a fixed set of numbers throws away the local context that pressure depends on. Geometry itself is cheap: thousands of car meshes exist with no simulation attached.

This package learns a description of a surface from geometry alone. It hides most of a shape, encodes the visible part into vector tokens pinned to positions on the surface, and trains a decoder to predict the distance to the true surface and the normal at points around it. The encoder can only do that by learning what the shapes look like. The result per shape is a packet of tokens with positions, normals, area weights and the shape's centre, scale and area, ready to hand to a field model. The method is general; this release targets car surfaces as the geometry input to aerodynamic pressure models and was trained on ShapeNet-Car.

## Install

In an environment that already has the PyTorch build for your GPU (Python 3.10 or later):

```bash
pip install ".[mesh]"
```

## Run

```bash
geometry-latents prepare-mesh --manifest meshes.json --out prepared --points 16384 --queries 32768
geometry-latents validate     --manifest prepared/manifest.json
geometry-latents train        --manifest prepared/manifest.json --config configs/base.json --out runs/base --device cuda
geometry-latents evaluate     --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/test.json --masks 0 0.6
geometry-latents probe        --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/probe
geometry-latents encode       --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/embeddings
geometry-latents train-field  --checkpoint runs/base/best.pt --manifest prepared/manifest.json --fields fields --out runs/field --device cuda
```

`train` writes `best.pt`, `last.pt` and `metrics.jsonl`; add `--resume runs/base/last.pt` to continue an interrupted run. `make-demo` builds a synthetic dataset for a CPU smoke test with `configs/tiny.json`.

Presets: `tiny` (smoke test), `base` (3.2M parameters, 256 tokens), `anchors1024` (16M parameters, 1024 tokens; about 20 GB on an A100 at batch 16).

`train-field` turns a geometry checkpoint into a shape-and-field encoder: two input columns are added to its first layer, a standardised field value and a presence flag, and the model is fine-tuned to fill in hidden patches of the field from the visible part and the shape, under a random sign convention and random scales on the positive and negative parts so that no single convention is learned. With no values given it predicts the field from geometry alone. It writes `field_best.pt`, loaded with `geom_latents.load_field_encoder`; `encode` gives shape-and-field tokens for a car whose field is known, and `predict` the field at any point. Twenty epochs on 451 cars take about three minutes on an A100.

## Data

A JSON manifest listing meshes or prepared NPZ files:

```json
{"schema": "geometry_latents_v1", "units": "metres",
 "records": [{"id": "car_001", "family": "design_001", "split": "train", "path": "cars/car_001.obj"}]}
```

Variants of one design share a `family`, and a family stays in one split. `prepare-mesh` turns triangle meshes into the NPZ format (area-weighted surface samples with normals and weights, plus query points with exact distances and normals).

`train-field` also needs one NPZ per record in the `--fields` directory, named `<id>.npz`, with `points` [N,3] in the same coordinates as the mesh, `values` [N] and `weights` [N]; each sample point takes the value of its nearest field point.

Two things to get right before preparation. Normals must be consistently outward; a mesh with inconsistent winding gives random normal targets. Shapes must be in the axis convention of the checkpoint you use; the model is deliberately not rotation-invariant, and the ShapeNet-Car checkpoints expect Y up with the car's length along Z. Position and scale do not matter.

## Large, uncurated corpora

Everything here is off by default; the defaults reproduce 0.2.0 exactly. For a corpus such as Objaverse, where meshes are unoriented, unaligned and occasionally unreadable:

- `"train": {"normal_loss": "unoriented"}` scores the normal by the angle between lines, so a flipped or double-sided face is not an error; `prepare-mesh` now also reports `winding_consistent` per mesh (`watertight` does not detect flipped faces; neither flag detects double-sided duplicates).
- `"data": {"augment_rotation": "so3"}` (or `"octahedral"` for the 24 axis-aligned rotations) rotates each training example, targets included; `"normal_sign_flip_probability": 0.5` negates random input normals so the encoder stops trusting their sign. Validation and export are never augmented.
- `"data": {"bad_records": "skip"}` makes the loader report an unreadable record once and substitute the next one instead of ending the run. The principled route is `validate --skip-bad --filtered-manifest clean.json`, which writes the manifest without the failures.
- `prepare-mesh --manifest ... --workers 16 --backend igl --skip-bad` converts in parallel with libigl's distance kernel (exact; trimesh's is off by about 1e-5 on a few near-surface queries per mesh, and where two faces are equidistant either backend may return either face's normal), lists failures in `failures.json`, and `--resume` continues a partial directory. Worker processes run with one BLAS thread each; without that, parallel conversion and validation are slower than sequential.
- `validate --workers 16 --out audit.json` then `train --audit audit.json` audits the files once instead of at every start, and `"train": {"warmup_steps": 2000}` gives a warmup that is not a whole epoch.

`pip install ".[mesh,fast]"` adds numba (a compiled farthest-point loop with identical anchors, five to ten times less loader time per sample) and libigl. With many loader workers, run training with `OMP_NUM_THREADS=1` for the same reason as above.

## Version

0.3.0 adds the opt-in options above for large corpora. Checkpoints and prepared files are unchanged; a run started under 0.2.0 resumes under 0.3.0.

0.2.0 adds `train-field` and `geom_latents.field`. Geometry checkpoints are unchanged.

0.1.1 fixes `train` crashing at start on a bare `--device cuda`. Checkpoints are unchanged from 0.1.0.
