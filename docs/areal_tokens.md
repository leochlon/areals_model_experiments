# Areal

Machine-learning surrogates for CFD take a shape and predict what the flow does to it: surface pressure, shear, forces. Each has to be told what the shape is, and the usual ways are poor. Feeding raw points makes the field model learn geometry from its only signal, field labels, one expensive simulation per example. Squeezing the shape into a fixed set of numbers throws away the local context that pressure depends on. Geometry itself is cheap: thousands of car meshes exist with no simulation attached.

Areal learns a description of a surface from geometry alone. It hides most of a shape, encodes the visible part into *areal tokens*, vectors pinned to positions on the surface that each carry a normal and an area, and trains a decoder to predict the distance to the true surface and the normal at points around it. The encoder can only do that by learning what the shapes look like. The result per shape is a packet of tokens with positions, normals, area weights and the shape's centre, scale and area: a quadrature rule as well as a representation, ready to hand to a field model or to integrate a predicted field over. The method is general; this release targets car surfaces as the geometry input to aerodynamic pressure models and was trained on ShapeNet-Car.

## Install

In an environment that already has the PyTorch build for your GPU (Python 3.10 or later):

```bash
pip install ".[mesh]"
```

## Run

```bash
areal prepare-mesh --manifest meshes.json --out prepared --points 16384 --queries 32768
areal validate     --manifest prepared/manifest.json
areal train        --manifest prepared/manifest.json --config configs/base.json --out runs/base --device cuda
areal evaluate     --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/test.json --masks 0 0.6
areal probe        --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/probe
areal encode       --checkpoint runs/base/best.pt --manifest prepared/manifest.json --out runs/base/embeddings
areal train-field  --checkpoint runs/base/best.pt --manifest prepared/manifest.json --fields fields --out runs/field --device cuda
```

`train` writes `best.pt`, `last.pt` and `metrics.jsonl`; add `--resume runs/base/last.pt` to continue an interrupted run. `make-demo` builds a synthetic dataset for a CPU smoke test with `configs/tiny.json`.

Presets: `tiny` (smoke test), `base` (3.2M parameters, 256 tokens), `surfels` (`base` with the normal and surfel options below: eight surfels per token and the completeness flag), `symmetry` (`surfels` with the symmetry-map prior below, 16 maps), `anchors1024` (16M parameters, 1024 tokens; about 20 GB on an A100 at batch 16).

`train-field` turns a geometry checkpoint into a shape-and-field encoder: two input columns are added to its first layer, a standardised field value and a presence flag, and the model is fine-tuned to fill in hidden patches of the field from the visible part and the shape, under a random sign convention and random scales on the positive and negative parts so that no single convention is learned. With no values given it predicts the field from geometry alone. It writes `field_best.pt`, loaded with `areal_tokens.load_field_encoder`; `encode` gives shape-and-field tokens for a car whose field is known, and `predict` the field at any point. Twenty epochs on 451 cars take about three minutes on an A100.

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

## Normals

Unsigned distance has a kink at the surface, so a normal read off its slope is unreliable exactly where it matters, and the normal target has no sign on meshes with inconsistent winding or open edges. Options, all off by default:

- `"model": {"normal_head": "projector"}` predicts the normal as a line: the symmetric matrix n nᵀ, read out as its top eigenvector. A unit-vector output must flip sign somewhere around every open boundary edge and wherever the sign convention changes, and each flip leaves a wall of near-90-degree errors; the projector has no sign and no walls. Requires `"train": {"normal_loss": "unoriented"}`. Checkpoints made with it have a six-output normal head.
- `"model": {"normal_prior": "surfels", "distance_prior": "surfels"}` (projector head) reads the packet as surfels: every anchor is a disk through its position, perpendicular to its normal, half the median anchor spacing in radius. The distance to the nearest disk and a soft-min blend of the nearby disks' normal projectors are near-exact wherever the surface is visible, and a learned per-query gate, fed with how far the query projects outside its nearest disk, mixes them with the heads, which take over where the surface is hidden. Read this way with no training at all, a 256-token packet already gives 10.4 degrees and a normalised distance error of 0.019 on full input; 512 tokens give 8.6 and 0.013, 1024 give 6.8 and 0.009.
- `"normal_prior": "surfels_residual"` adds the normal head to the surfel projector instead of blending, so it learns corrections for the disks' flatness. `"distance_prior": "union"` treats the shape as the visible surfels plus an unseen part: the distance head learns only the distance to the unseen part, the prediction is the soft minimum of the two, and the nearer part decides whose normal is used. Union decoding is the most accurate on full input; the gated blend is better at completing hidden regions. `"normal_prior": "anchors"` (a kernel average of anchor projectors) is superseded by the surfel options.
- `"model": {"surfels_per_token": 4}` (with a surfel prior) keeps more surfels than tokens: the farthest-point samples of the input that extend the anchors, carried in the packet beside the tokens. On visible geometry the readout's accuracy is set by surfel spacing, not by the network. On 41 held-out meshes (170 training meshes, 24 epochs), trained full-input normal error falls from 10.4 degrees at one surfel per token to 7.4 at four and 6.1 at eight (median 2.7 to 1.7, share over 45 degrees 7.3% to 3.3%), and at 60% hidden from 14.5 to 12.4. Untrained, the readout at eight per token (5.5) is within half a degree of reading the nearest raw input sample (5.2); denser still keeps paying (32 per token from 16384 input points: 3.4). Token count and attention cost are unchanged; each extra surfel is six numbers. `encode` exports the surfels and `load_latents` reads them back.
- `"model": {"input_condition": "complete"}` gives the decoder one number per shape: whether its input is the whole surface, as it is for `encode` and for downstream geometry. Masked pretraining otherwise teaches the surfel gate a caution that is wrong for complete inputs: on full input the trained model's distance error was four times that of its own untrained readout. With the flag, full-input distance nRMSE falls from 0.074 to 0.021 at one surfel per token and from 0.074 to 0.012 at four, full-input normal error at four from 7.4 to 6.9 degrees (median 2.0 to 1.2), and the 60%-hidden results are unchanged. With eight surfels per token and the flag (the `surfels` preset), full input gives 5.7 degrees (median 0.8, 3.3% over 45) and a distance nRMSE of 0.011, against 10.4 degrees and 0.074 for the gated surfel prior alone; 60% hidden gives 12.4 degrees (17.4 on the hidden part, 5.4 on the visible). Packets saved without the flag are read as complete.
- `"normal_prior": "surfels_denoise"` treats the surfel normal as a noisy observation the decoder reads per query, with its noise level and the readout's evidence; `"normal_target": "eps"` predicts the noise instead of the correction, diffusion style, and `"hint_noise"` trains on that fraction of hints drawn from randomly rotated true normals. In the same A/B both were worse than the gated blend (full input 10.8 and 11.3 degrees against 10.4; 60% hidden 14.5 and 16.3 against 14.5). Kept for trials at scale; not recommended.
- `"train": {"normal_penalty": "angle"}` penalises the angle itself rather than 1 − cos, which barely penalises moderate errors (10 degrees costs 0.015). Applies to the vector head.
- Every `evaluate` report now carries `normal_breakdown`: median, share under 10 and 20 degrees and over 45, by distance from the surface and by the accuracy of the predicted distance at the same query. `evaluate --normal-readout gradient` reads normals off the distance field's gradient instead, as a diagnostic.

`prepare-mesh` loads geometry only, without decoding textures (a 12 MB textured GLB needed over 4.5 GB before; the triangles are byte-identical), and a worker process killed for memory or by a crash no longer hangs `prepare-mesh --workers` or `validate --workers`: its record is retried alone and reported as failed.

`pip install ".[mesh,fast]"` adds numba (a compiled farthest-point loop with identical anchors, five to ten times less loader time per sample) and libigl. With many loader workers, run training with `OMP_NUM_THREADS=1` for the same reason as above.

## Symmetry

A hidden part of a shape is often a copy of a visible one: the far side of a mirror plane, another leg, wheel or bolt. The trunk alone does not exploit this; the options below hand it the copies. All are off by default.

- `"model": {"symmetry_prior": "maps"}` (with the gated surfel priors) finds the visible input's rigid symmetries by transformation-space voting on curvature frames (Mitra, Guibas and Pauly 2006): reflections, rotations and translations between repeated parts, each kept only where it carries the input onto itself over whole regions. For each of the best `symmetry_maps` maps (8 by default, 16 in the `symmetry` preset), the surfel readout at the query's image is carried back and blended with the direct readout and the heads by a learned gate, fed per map with whether the map holds on the disks around the query, how far the image lies from any disk, and the map's support; the gate starts nearly closed. On the 41 held-out meshes (170 training meshes, 24 epochs, one seed), 60% hidden: hidden-region normal error falls from 17.6 to 15.7 degrees (mean over meshes; better on 32 of 41), and from 10.9 to 6.9 on the 21% of hidden points where a repeated part is found, where the gate gives the maps 61% of the weight; removing the maps at inference restores the baseline. Pooled 60%-hidden error falls from 12.4 to 11.4 degrees; full-input, visible and distance accuracy are unchanged. Voting costs 0.2 to 0.5 s per sample in the loader (give training `workers`), and a training step about 1.8 times the `surfels` preset's. `encode` exports the maps with each packet and `load_latents` reads them back; a packet without them decodes as one where no symmetry was found.
- `"model": {"mirror_prior": "surfels"}` is the special case of one global mirror plane: 17.6 to 17.1 degrees in the same test. Superseded by `symmetry_prior`; the two are exclusive.
- `"data": {"mask_patch_mix": "mixed"}` trains on holes from a quarter to four times `mask_patches`. In the same test hidden-region error rose by 0.6 degrees and 60%-hidden distance nRMSE from 0.082 to 0.111. Not recommended.

The ceiling is higher. With symmetries detected on the complete test shapes, 63% of hidden points (33% of the irregular ones) lie where a mirror, a repeated part or a continuous symmetry maps them onto visible input. Continuous symmetries are surfaces that slide along themselves: flat, extruded and revolved. They cover 50% of hidden points, and extrusions carry sharp edges into holes. They are not yet used.

## Version

0.6.0 adds `symmetry_prior`, `mirror_prior` and `mask_patch_mix` above. Default checkpoints, outputs and prepared files are unchanged; packets gain optional symmetry arrays only when `symmetry_prior` is on.

0.5.0 adds `surfels_per_token`, `input_condition` and the denoising prior above. Default checkpoints, outputs and prepared files are unchanged; packets gain optional surfel and completeness arrays only when those options are on.

0.4.0 adds the surfel, union and anchor priors above; default checkpoints and outputs are unchanged.

0.3.1 adds the normal options above, the evaluation breakdown, geometry-only mesh loading and crash-safe worker pools. Defaults, checkpoints and prepared files are unchanged.

0.3.0 renames the project from geometry-latents to Areal (distribution `areal-tokens`, module `areal_tokens`, command `areal`) and adds the opt-in options above for large corpora. Checkpoints, prepared files and the manifest schema are unchanged; a run started under 0.2.0 resumes under 0.3.0.

0.2.0 adds `train-field` and the field module. Geometry checkpoints are unchanged.

0.1.1 fixes `train` crashing at start on a bare `--device cuda`. Checkpoints are unchanged from 0.1.0.
