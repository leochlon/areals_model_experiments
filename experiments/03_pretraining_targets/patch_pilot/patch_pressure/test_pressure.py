#!/usr/bin/env python3
"""Independent interface, invariance, leakage, metric and training checks.

python test_pressure.py                 # fast unit and VTK tests
python test_pressure.py --integration   # also training, resume and inference
All generated fields are MANUFACTURED; these are not Windsor CFD results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

import train_mesh_geometry as geo
from pressure_model import PressureModel, TokenPatchEncoder
from pressure_data import Cases, load_geometry, tensor_scene, case_metrics
from prepare_pressure import parser as preparation_parser, lumped_vertex_areas
from train_patch_pressure import restore, parser as training_parser, initialize, check_evaluation_split, setup
from pressure_baseline import nearest_predictions

ROOT = Path(__file__).resolve().parent
CONFIG = dict(degree=3, width=32, heads=4, depth=1, latents=2)
TEST_DEVICE = "cpu"
TEST_ALLOW_NONDETERMINISTIC = False


def analytic_patches(alpha, uv):
    alpha = np.asarray(alpha)
    au, av, cross = .1+.03*alpha, .05-.02*alpha, .04*alpha
    u, v = uv[..., 0], uv[..., 1]
    height = alpha[:, None] + au[:, None]*u**2 + av[:, None]*v**2 + cross[:, None]*u*v
    normals = np.stack((-(2*au[:, None]*u+cross[:, None]*v),
                        -(2*av[:, None]*v+cross[:, None]*u), np.ones_like(height)), -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    inputs = np.concatenate((uv, height[..., None], normals), -1)
    coeff = np.zeros((len(alpha), 10)); coeff[:, 0] = alpha
    coeff[:, geo.powers(3).index((2, 0))] = au
    coeff[:, geo.powers(3).index((0, 2))] = av
    coeff[:, geo.powers(3).index((1, 1))] = cross
    return inputs, coeff


def manufactured_pressure(xyz, alpha, flow):
    x, y = xyz[:, 0], xyz[:, 1]
    # Mixed geometry/location dependence survives subtracting any constant mean.
    return .2 + .4*x - .3*y + alpha*(1.2+1.8*np.sin(np.pi*x)*np.cos(np.pi*y)) + .15*flow + .2*(flow-1.5)*x


def fixture(root, pretrain_steps=0, device="cpu"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(991)
    torch.manual_seed(99)
    encoder = geo.PatchEncoder(**CONFIG).to(device)
    pretraining = dict(steps=pretrain_steps, target="analytic geometry polynomial coefficients", pressure_labels_used=False)
    if pretrain_steps:
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.)
        pretrain_rng = np.random.default_rng(441)
        x, truth = analytic_patches(np.linspace(-.18, .18, 16), pretrain_rng.uniform(-.8, .8, (16, 32, 2)))
        check_x = torch.tensor(x, device=device, dtype=torch.float32)
        check_y = torch.tensor(truth, device=device, dtype=torch.float32)
        with torch.no_grad(): initial = float((encoder(check_x)-check_y).square().mean())
        for _ in range(pretrain_steps):
            x, truth = analytic_patches(pretrain_rng.uniform(-.2, .2, 16), pretrain_rng.uniform(-.8, .8, (16, 32, 2)))
            optimizer.zero_grad(set_to_none=True)
            loss = (encoder(torch.tensor(x, device=device, dtype=torch.float32))-torch.tensor(truth, device=device, dtype=torch.float32)).square().mean()
            loss.backward(); optimizer.step()
        with torch.no_grad(): final = float((encoder(check_x)-check_y).square().mean())
        assert final < .2*initial, "Geometry-only pretraining did not learn"
        pretraining.update(initial_coefficient_mse=initial, final_coefficient_mse=final)
    torch.save(dict(version=geo.VERSION, model_config=CONFIG,
                    model={k: v.cpu() for k, v in encoder.state_dict().items()}, step=pretrain_steps,
                    pretraining=pretraining), root / "geometry.pt")
    (root/"geometry_pretraining.json").write_text(json.dumps(pretraining, indent=2))
    entries = []
    # Cross geometry and operating condition in training. Assigning a unique
    # condition to each geometry would permit memorizing shape from the flow
    # condition and would not identify the intended geometry dependence.
    design = [(a, u, "train") for a in (-.15, -.05, .05, .15) for u in (1., 2.)]
    design += [(-.02, 1., "val"), (.08, 2., "val"), (-.1, 1., "test"), (.10, 2., "test")]
    for j, (alpha, flow, split) in enumerate(design):
        origin = np.array([[x, y, 0] for x in (-.3, 0., .3) for y in (-.25, .25)])
        uv = rng.uniform(-.8, .8, (len(origin), 24, 2))
        inputs, coefficients = analytic_patches(np.full(len(origin), alpha), uv)
        metadata = dict(version=geo.VERSION, degree=3,
                        meshes=[dict(center=[0., 0., 0.], scale=1., sampled_coverage=1.,
                                     sha256=hashlib.sha256(f"manufactured-{alpha}".encode()).hexdigest())])
        np.savez_compressed(root / f"geometry_{j}.npz", inputs=inputs, origin=origin,
                            frame=np.tile(np.eye(3), (len(origin), 1, 1)), radius=np.full(len(origin), .2),
                            coeff=coefficients, mesh_index=np.zeros(len(origin), dtype=np.int64),
                            metadata=np.array(json.dumps(metadata)))
        xyz = rng.uniform(-.4, .4, (256, 3))
        xyz[:, 2] = 0.  # Fixed planar probe surface, independent of geometry alpha.
        condition = [flow, 2.]
        cp = manufactured_pressure(xyz, alpha, flow)
        np.savez_compressed(root / f"field_{j}.npz", xyz=xyz, cp=cp, weights=np.full(len(cp), .64/len(cp)),
                            metadata=np.array(json.dumps(dict(weight_measure="area", weighting="uniform Monte Carlo quadrature on fixed square probe plane"))))
        entries.append(dict(id=f"case_{j}", car_id=f"car_{alpha}", split=split,
                            geometry=f"geometry_{j}.npz", field=f"field_{j}.npz", conditions=condition))
    doc = dict(version="patch-pressure-data-v1", evaluation_scope="held_out_cars", condition_names=["condition_a", "condition_b"], cases=entries)
    (root / "cases.json").write_text(json.dumps(doc))
    return root / "cases.json"


class Checks(unittest.TestCase):
    def setUp(self):
        setup(17, 1, not TEST_ALLOW_NONDETERMINISTIC)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = fixture(self.root)
        self.model = PressureModel(CONFIG, 2, width=32, global_tokens=4, global_depth=1, encoder_chunk=2).to(TEST_DEVICE)
        self.g = load_geometry(self.root / "geometry_0.npz")
        self.scene = tensor_scene(self.g, TEST_DEVICE)
        self.q = torch.randn(11, 3, device=TEST_DEVICE) * .15
        self.c = torch.tensor([1.1, 2.2], device=TEST_DEVICE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_original_encoder_interface(self):
        original = geo.PatchEncoder(**CONFIG).to(TEST_DEVICE)
        original.load_state_dict(self.model.encoder.state_dict(), strict=True)
        direct = original(self.scene["inputs"])
        tokens = self.model.encoder.encode_tokens(self.scene["inputs"])
        self.assertTrue(torch.equal(direct, self.model.encoder.head(tokens.mean(1))))

    def test_query_chunk_and_order_invariance(self):
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode_scene(self.scene, self.c)
            whole = self.model.decode(encoded, self.q)
            chunks = torch.cat([self.model.decode(encoded, q) for q in self.q.split(3)])
            perm = torch.randperm(len(self.q))
            reordered = self.model.decode(encoded, self.q[perm])
        torch.testing.assert_close(whole, chunks, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(whole[perm], reordered, atol=2e-6, rtol=2e-6)

    def test_geometry_permutations_and_normal_signs(self):
        self.model.eval()
        changed = {k: v.clone() for k, v in self.scene.items()}
        changed["inputs"][:, :, 3:] *= -1
        changed["inputs"] = changed["inputs"][:, torch.randperm(changed["inputs"].shape[1])]
        perm = torch.randperm(len(changed["inputs"]))
        changed = {k: v[perm] if k != "scale" else v for k, v in changed.items()}
        with torch.no_grad():
            a = self.model(self.scene, self.c, self.q)
            b = self.model(changed, self.c, self.q)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)

    def test_every_parameter_receives_finite_gradient(self):
        loss = self.model(self.scene, self.c, self.q).square().mean()
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(float(param.grad.abs().max()), 0., name)

    def test_checkpointed_gradient_matches_direct_and_finite_difference(self):
        a = self.model
        b = copy.deepcopy(a); b.config["gradient_checkpointing"] = False
        for m in (a, b):
            m(self.scene, self.c, self.q).square().mean().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5)
        # Independent central difference through the whole pressure predictor.
        param = b.encoder.head[-1].weight
        index = np.unravel_index(int(param.grad.abs().argmax()), param.shape)
        derivative, saved, eps = float(param.grad[index]), float(param.detach()[index]), 1e-3
        with torch.no_grad():
            param[index] = saved + eps; plus = float(b(self.scene, self.c, self.q).square().mean())
            param[index] = saved - eps; minus = float(b(self.scene, self.c, self.q).square().mean())
            param[index] = saved
        self.assertAlmostEqual((plus - minus) / (2 * eps), derivative, delta=max(2e-4, abs(derivative) * .02))

    def test_frozen_encoder_is_unchanged(self):
        self.model.freeze_encoder(True)
        before = {k: v.clone() for k, v in self.model.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        self.model(self.scene, self.c, self.q).square().mean().backward(); optimizer.step()
        self.assertTrue(all(torch.equal(v, self.model.encoder.state_dict()[k]) for k, v in before.items()))

    def test_geometry_and_conditions_influence_pressure(self):
        self.model.eval()
        changed = {k: v.clone() for k, v in self.scene.items()}
        changed["inputs"][:, :, 2] += .3
        with torch.no_grad():
            a = self.model(self.scene, self.c, self.q)
            b = self.model(changed, self.c, self.q)
            c = self.model(self.scene, self.c + 1, self.q)
        self.assertGreater(float((a-b).abs().max()), 1e-6)
        self.assertGreater(float((a-c).abs().max()), 1e-6)

    def test_reference_geometry_targets_are_unused(self):
        path = self.root / "geometry_0.npz"
        with np.load(path) as z: contents = {k: z[k] for k in z.files}
        contents["coeff"] = np.full_like(contents["coeff"], 1e12)
        np.savez_compressed(path, **contents)
        changed = tensor_scene(load_geometry(path), TEST_DEVICE)
        self.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.scene, self.c, self.q), self.model(changed, self.c, self.q), atol=0, rtol=0)

    def test_test_labels_do_not_affect_normalization(self):
        a = Cases(self.manifest).normalization()
        path = self.root / "field_11.npz"
        with np.load(path) as z: content = {k: z[k] for k in z.files}
        content["cp"] *= 1e9; np.savez_compressed(path, **content)
        b = Cases(self.manifest).normalization()
        self.assertEqual(a, b)

    def test_held_out_car_leakage_rejected(self):
        doc = json.loads(self.manifest.read_text()); doc["cases"][-1]["car_id"] = doc["cases"][0]["car_id"]
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "crosses"):
            Cases(self.manifest)

    def test_external_evaluation_cannot_relabel_training_car(self):
        data = Cases(self.manifest)
        state = dict(condition_names=data.condition_names, data_fingerprint=data.audit["fingerprint"], training_data_audit=copy.deepcopy(data.audit))
        check_evaluation_split(state, data, "test")
        doc = json.loads(self.manifest.read_text())
        doc["cases"] = [doc["cases"][0]]
        doc["cases"][0]["split"] = "test"
        self.manifest.write_text(json.dumps(doc))
        changed = Cases(self.manifest, require_train=False)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_evaluation_split(state, changed, "test")

    def test_same_car_split_and_duplicate_coordinates(self):
        field = self.root / "field_0.npz"
        with np.load(field) as z: content = {k: z[k] for k in z.files}
        content = {k: np.concatenate((v, v[:10])) if k != "metadata" else v for k, v in content.items()}
        np.savez_compressed(field, **content)
        args = preparation_parser().parse_args(["single-case", "--geometry", str(self.root/"geometry_0.npz"),
                                               "--field", str(field), "--car-id", "car0", "--out", str(self.root/"split"),
                                               "--condition-names", "a", "b", "--conditions", "1", "2", "--split-method", "random"])
        args.func(args)
        path = self.root / "split/cases.json"
        data = Cases(path); self.assertEqual(data.scope, "held_out_queries_same_car")
        doc = json.loads(path.read_text()); doc["cases"][1]["indices"] = doc["cases"][0]["indices"]
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "leak"):
            Cases(path)

    def test_metrics_have_known_values(self):
        m = case_metrics([2., 4.], [1., 2.], [1., 3.], 2.)
        self.assertAlmostEqual(m["nrmse_reference_rms"], 1.)
        self.assertAlmostEqual(m["rmse_cp"], np.sqrt(13 / 4))
        self.assertAlmostEqual(m["nrmse_train_std"], np.sqrt(13 / 4) / 2)
        self.assertAlmostEqual(m["nrmse_reference_std"], np.sqrt(52/3))
        self.assertIsNone(case_metrics([1.], [0.], [1.], 1.)["nrmse_reference_rms"])
        self.assertIsNone(case_metrics([3.], [2.], [1.], 1.)["nrmse_reference_std"])

    def test_std_metric_gauge_invariance_and_bias_retention(self):
        a = case_metrics([2., 4.], [1., 2.], [1., 3.], 2.)
        b = case_metrics([102., 104.], [101., 102.], [1., 3.], 2.)
        self.assertAlmostEqual(a["nrmse_reference_std"], b["nrmse_reference_std"])
        self.assertNotAlmostEqual(a["nrmse_reference_rms"], b["nrmse_reference_rms"])
        biased = case_metrics([2., 3.], [1., 2.], [1., 1.], 1.)
        self.assertAlmostEqual(biased["nrmse_reference_std"], 2.)

    def test_lumped_areas_conserve_surface_area(self):
        xyz = np.array([[0.,0,0], [2.,0,0], [0.,1,0], [2.,1,0]])
        faces = np.array([[0,1,2], [1,3,2]])
        areas = lumped_vertex_areas(xyz, faces)
        np.testing.assert_allclose(areas, [1/3, 2/3, 2/3, 1/3])
        self.assertAlmostEqual(float(areas.sum()), 2.)
        np.testing.assert_allclose(lumped_vertex_areas(xyz*3, faces), areas*9)

    def test_blocked_split_buffer_is_real(self):
        args = preparation_parser().parse_args(["single-case", "--geometry", str(self.root/"geometry_0.npz"),
                                               "--field", str(self.root/"field_0.npz"), "--car-id", "car0", "--out", str(self.root/"blocked"),
                                               "--condition-names", "a", "b", "--conditions", "1", "2", "--buffer", ".05"])
        args.func(args)
        path = self.root/"blocked/cases.json"
        data = Cases(path, weighting="area")
        self.assertEqual(data.scope, "held_out_regions_same_car")
        self.assertTrue(all(s["min_distance_diagonal"] >= .05 for s in data.audit["query_separation"]))
        doc = json.loads(path.read_text()); doc["split_definition"]["buffer_fraction_diagonal"] = 1.
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "buffer"):
            Cases(path)

    def test_nearest_baseline_uses_only_training_values(self):
        pred, distance = nearest_predictions(np.array([[0.,0,0], [0.,0,0], [1.,0,0]]),
                                             np.array([1.,3.,5.]), np.array([1.,3.,1.]), np.array([[.1,0,0],[.9,0,0]]))
        np.testing.assert_allclose(pred, [2.5,5.]); np.testing.assert_allclose(distance, [.1,.1])

    def test_area_requirement_is_enforced(self):
        path = self.root/"field_0.npz"
        with np.load(path) as z: contents = {k: z[k] for k in z.files if k != "metadata"}
        np.savez_compressed(path, **contents)
        with self.assertRaisesRegex(ValueError, "area weighting required"):
            Cases(self.manifest, weighting="area")

    def test_manufactured_geometry_changes_spatial_pattern(self):
        xyz = np.array([[-.3,.1,0], [0.,.1,0], [.3,.1,0]])
        difference = manufactured_pressure(xyz,.1,1.) - manufactured_pressure(xyz,-.1,1.)
        self.assertGreater(float(np.std(difference)), .1)

    def test_determinism_flags_inherit_by_default(self):
        base = ["evaluate", "--manifest", "a", "--checkpoint", "b", "--out", "c"]
        self.assertIsNone(training_parser().parse_args(base).allow_nondeterministic)
        self.assertTrue(training_parser().parse_args(base+["--allow-nondeterministic"]).allow_nondeterministic)
        self.assertFalse(training_parser().parse_args(base+["--deterministic"]).allow_nondeterministic)

    def test_tuning_rejects_random_step_zero_checkpoint(self):
        from tune_pressure import run
        args = argparse.Namespace(manifest=str(self.manifest), geometry_checkpoint=str(self.root/"geometry.pt"),
                                  out=str(self.root/"tuning"), encoder_lrs=[.001], decoder_lrs=[.001],
                                  seeds=[17], diagnostic_same_car=False, resume=False)
        with self.assertRaisesRegex(ValueError, "actual training updates"):
            run(args, [])

    def test_matched_decoder_initialization(self):
        base = ["train", "--manifest", str(self.manifest), "--geometry-checkpoint", str(self.root/"geometry.pt"), "--out", str(self.root/"run"),
                "--width", "32", "--global-tokens", "4", "--global-depth", "1"]
        if TEST_ALLOW_NONDETERMINISTIC:
            base.append("--allow-nondeterministic")
        stats = Cases(self.manifest).normalization()
        _, a = initialize(training_parser().parse_args(base), stats)
        _, b = initialize(training_parser().parse_args(base + ["--init", "scratch"]), stats)
        self.assertEqual(a["decoder_initial_sha256"], b["decoder_initial_sha256"])
        self.assertNotEqual(a["encoder_initial_sha256"], b["encoder_initial_sha256"])

    def test_npz_static_and_kinematic_conversion(self):
        xyz = np.arange(9).reshape(3, 3)
        for kind, p, reference in [("static", np.array([100., 104., 108.]), 100.), ("kinematic", np.array([50., 52., 54.]), 50.)]:
            source, dest = self.root/f"{kind}.npz", self.root/f"{kind}_converted.npz"
            np.savez_compressed(source, xyz=xyz, p=p)
            args = preparation_parser().parse_args(["convert", "--input", str(source), "--out", str(dest), "--pressure-key", "p", "--kind", kind,
                                                   "--density", "2", "--speed", "2", "--reference-pressure", str(reference), "--xyz-scale", ".001", "--measure", "equal"])
            args.func(args)
            with np.load(dest) as z:
                np.testing.assert_allclose(z["cp"], [0., 1., 2.]); np.testing.assert_allclose(z["xyz"], xyz*.001)

    def test_vtk_native_cell_and_point_conversion(self):
        try:
            from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkTriangle, vtkCellArray
            from vtkmodules.vtkCommonCore import vtkPoints
            from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
            from vtkmodules.util.numpy_support import numpy_to_vtk
        except ImportError:
            self.skipTest("optional vtk package unavailable")
        data = vtkPolyData(); pts = vtkPoints()
        for xyz in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]: pts.InsertNextPoint(xyz)
        data.SetPoints(pts); triangle = vtkTriangle()
        for i in range(3): triangle.GetPointIds().SetId(i, i)
        cells = vtkCellArray(); cells.InsertNextCell(triangle); data.SetPolys(cells)
        cell = numpy_to_vtk(np.array([.7])); cell.SetName("Cp"); data.GetCellData().AddArray(cell)
        point = numpy_to_vtk(np.array([.1, .2, .3])); point.SetName("Cp"); data.GetPointData().AddArray(point)
        source = self.root / "source.vtp"
        writer = vtkXMLPolyDataWriter(); writer.SetFileName(str(source)); writer.SetInputData(data); writer.Write()
        for association in ("cell", "point"):
            dest = self.root/f"vtk_{association}.npz"
            command = ["convert", "--input", str(source), "--out", str(dest), "--pressure-key", "Cp", "--kind", "cp", "--association", association]
            if association == "cell": command += ["--measure", "area"]
            args = preparation_parser().parse_args(command); args.func(args)
            with np.load(dest) as z:
                np.testing.assert_allclose(z["cp"], [.7] if association == "cell" else [.1, .2, .3])
                if association == "cell":
                    np.testing.assert_allclose(z["xyz"], [[1/3, 1/3, 0]], atol=1e-7)
                    np.testing.assert_allclose(z["weights"], [.5])
                else:
                    np.testing.assert_allclose(z["weights"], [1/6]*3)


def run_command(arguments, logfile):
    with Path(logfile).open("w") as log:
        result = subprocess.run([sys.executable, str(ROOT/"train_patch_pressure.py")] + arguments,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(Path(logfile).read_text())


def integration(root, steps=500, device="cpu", allow_nondeterministic=False):
    root = Path(root)
    manifest = fixture(root / "data", pretrain_steps=200, device=device)
    flags = ["--allow-nondeterministic"] if allow_nondeterministic else []
    common = ["train", "--manifest", str(manifest), "--geometry-checkpoint", str(root/"data/geometry.pt"),
              "--device", device, "--width", "32", "--global-tokens", "4", "--global-depth", "1",
              "--encoder-chunk", "3", "--neighbors", "3", "--queries", "64", "--validation-queries", "128",
              "--query-chunk", "64", "--lr", ".001", "--encoder-lr", ".001", "--patience", "0", "--cases-per-step", "2"] + flags
    run_command(common + ["--steps", "6", "--validate-every", "3", "--out", str(root/"whole")], root/"whole.log")
    run_command(common + ["--steps", "6", "--validate-every", "3", "--stop-after", "2", "--out", str(root/"resumed")], root/"pause.log")
    run_command(common + ["--steps", "6", "--validate-every", "3", "--resume", "--out", str(root/"resumed")], root/"resume.log")
    a = torch.load(root/"whole/last.pt", weights_only=True, map_location="cpu")
    b = torch.load(root/"resumed/last.pt", weights_only=True, map_location="cpu")
    bitwise = all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    optimizer_equal = all(torch.equal(v, b["optimizer"]["state"][k][name])
                          for k, state in a["optimizer"]["state"].items() for name, v in state.items())
    rng_equal = a["rng"] == b["rng"] and torch.equal(a["torch_rng"], b["torch_rng"])
    cuda_rng_equal = len(a["cuda_rng"]) == len(b["cuda_rng"]) and all(torch.equal(x,y) for x,y in zip(a["cuda_rng"], b["cuda_rng"]))
    if not allow_nondeterministic:
        assert bitwise and optimizer_equal and rng_equal and cuda_rng_equal, "Deterministic resume differs"
        assert a["validation_history"] == b["validation_history"], "Resume validation history differs"
    else:
        assert all(torch.allclose(v, b["model"][k], atol=1e-5, rtol=1e-4) for k,v in a["model"].items()), "Resume diverged"
    run_command(common + ["--steps", str(steps), "--validate-every", "50", "--out", str(root/"learn")], root/"learn.log")
    assert not (root/"learn/test_metrics.json").exists(), "Training scored test before explicit evaluation"
    initial = json.loads((root/"learn/validation_initial.json").read_text())
    state = torch.load(root/"learn/best.pt", weights_only=True, map_location="cpu")
    run_command(["evaluate", "--manifest", str(manifest), "--checkpoint", str(root/"learn/best.pt"),
                 "--out", str(root/"learn/test_metrics.json"), "--device", device], root/"evaluate.log")
    final = json.loads((root/"learn/test_metrics.json").read_text())
    assert final["deterministic"] == (not allow_nondeterministic), "Evaluation changed determinism policy"
    assert state["best"] < initial["selection_score"] * .35, "Manufactured field did not learn"
    assert final["case_balanced_rmse_cp"] < .07, "Manufactured held-out-car error too high"
    with np.load(root/"data/field_10.npz") as z: xyz = z["xyz"]
    np.savez_compressed(root/"queries_only.npz", xyz=xyz)
    run_command(["predict", "--geometry", str(root/"data/geometry_10.npz"), "--checkpoint", str(root/"learn/best.pt"),
                 "--queries", str(root/"queries_only.npz"), "--conditions", "1", "2", "--out", str(root/"predictions.npz"),
                 "--device", device, "--query-chunk", "37"], root/"predict.log")
    with np.load(root/"predictions.npz") as z:
        assert np.isfinite(z["cp_predicted"]).all() and len(z["cp_predicted"]) == len(xyz)
        assert json.loads(str(z["metadata"]))["deterministic"] == (not allow_nondeterministic)
    # Remove constant offsets before testing whether the trained predictor
    # uses geometry to reproduce a DIFFERENT SPATIAL pressure pattern.
    model, _ = restore(root/"learn/best.pt", device); model.eval()
    centered_correct, centered_wrong = [], []
    with torch.no_grad():
        for target_index, wrong_index, flow in ((10,6,1.),(11,0,2.)):
            with np.load(root/f"data/field_{target_index}.npz") as z: q, truth = z["xyz"], z["cp"]
            values = []
            for gi in (target_index, wrong_index):
                g = load_geometry(root/f"data/geometry_{gi}.npz")
                pred = model(tensor_scene(g, device), torch.tensor([flow,2.], device=device),
                             torch.tensor(q, device=device, dtype=torch.float32)).cpu().numpy()
                values.append(float(np.mean(((pred-pred.mean())-(truth-truth.mean()))**2)))
            centered_correct.append(values[0]); centered_wrong.append(values[1])
    assert np.mean(centered_wrong) > 1.5*np.mean(centered_correct), "Spatial geometry dependence was not learned"
    # Exercise the actual validation-only rate search, with a truly pretrained
    # geometry checkpoint and an independently initialized scratch encoder.
    tuning = [sys.executable, str(ROOT/"tune_pressure.py"), "--manifest", str(manifest),
              "--geometry-checkpoint", str(root/"data/geometry.pt"), "--out", str(root/"tuning"),
              "--encoder-lrs", ".0001", ".001", "--decoder-lrs", ".001", "--seeds", "17",
              "--device", device, "--steps", "60", "--validate-every", "20", "--patience", "0",
              "--width", "32", "--global-tokens", "4", "--global-depth", "1", "--queries", "64",
              "--encoder-chunk", "3", "--validation-queries", "128", "--cases-per-step", "2"] + flags
    with (root/"tuning.log").open("w") as log:
        completed = subprocess.run(tuning, stdout=log, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, (root/"tuning.log").read_text()
    from tune_pressure import verify_selection
    selection_path = root/"tuning/selection.json"
    assert verify_selection(selection_path)
    selection = json.loads(selection_path.read_text())
    assert len(selection["trials"]) == 4
    for trial in selection["trials"]:
        directory = (root/"tuning"/trial["validation_path"]).parent
        expected = trial["grid_index"] == selection["winner_grid"][trial["arm"]]
        assert (directory/"selected_test.json").exists() == expected, "Unselected candidate was tested"
    tampered = copy.deepcopy(selection); tampered["winner_grid"]["scratch"] = 1-tampered["winner_grid"]["scratch"]
    fake = root/"tuning/wrong_selection.json"; fake.write_text(json.dumps(tampered))
    try:
        verify_selection(fake)
        raise AssertionError("Non-minimal selection was accepted")
    except ValueError:
        pass
    output = dict(manufactured_fields_only=True, device=device, deterministic=not allow_nondeterministic,
                  resume_bitwise_weights=bitwise, resume_bitwise_optimizer=optimizer_equal,
                  resume_rng_identical=rng_equal and cuda_rng_equal, query_only_inference=True,
                  evaluation_and_inference_inherit_determinism=True,
                  actual_geometry_pretraining_steps=200, geometry_dependent_spatial_pattern=True,
                  centered_correct_geometry_rmse=float(np.sqrt(np.mean(centered_correct))),
                  centered_wrong_geometry_rmse=float(np.sqrt(np.mean(centered_wrong))),
                  initial_validation_nrmse_std=initial["selection_score"], best_validation_nrmse_std=state["best"],
                  held_out_manufactured_rmse_cp=final["case_balanced_rmse_cp"],
                  held_out_manufactured_nrmse_std=final["case_mean_nrmse_reference_std"],
                  independent_rate_search_verified=True, unselected_candidates_never_tested=True,
                  tampered_selection_rejected=True, tuning_steps_per_candidate=60,
                  steps=steps, best_step=state["step"], real_cfd_accuracy_measured=False)
    (root/"integration_results.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--integration", action="store_true")
    p.add_argument("--out", default="pressure_checks"); p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="auto"); p.add_argument("--allow-nondeterministic", action="store_true")
    args = p.parse_args()
    TEST_DEVICE = str(geo.choose_device(args.device))  # explicit CUDA requests never silently fall back
    TEST_ALLOW_NONDETERMINISTIC = args.allow_nondeterministic
    setup(17, 1, not args.allow_nondeterministic)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful(): sys.exit(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = dict(unit_tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), device=TEST_DEVICE, cuda_tested=TEST_DEVICE.startswith("cuda"),
                  deterministic=not args.allow_nondeterministic, real_cfd_accuracy_measured=False)
    if args.integration:
        setup(17, 1, not args.allow_nondeterministic)
        report["integration"] = integration(out/"integration", args.steps, TEST_DEVICE, args.allow_nondeterministic)
    (out/"checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
