#!/usr/bin/env python3
"""Convert existing CFD samples; make an explicit same-car query split.

No CFD solve and no polynomial field interpolation. VTK values are used at
their native points or cell centers, according to the chosen association.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
from train_mesh_geometry import json_write, sha256
from pressure_data import load_field, load_geometry


def lumped_vertex_areas(xyz, triangles):
    xyz, triangles = np.asarray(xyz, dtype=np.float64), np.asarray(triangles)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Lumped areas require xyz[N,3] and triangles[M,3]")
    if triangles.dtype.kind not in "iu" or not len(triangles) or triangles.min() < 0 or triangles.max() >= len(xyz):
        raise ValueError("Invalid triangle vertex indices")
    faces = xyz[triangles]
    areas = .5 * np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1)
    weights = np.zeros(len(xyz), dtype=np.float64)
    np.add.at(weights, triangles.reshape(-1), np.repeat(areas/3, 3))
    return weights


def vtk_samples(path, pressure_key, association, weights_key=None, measure="area", block=None):
    try:
        from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader
        from vtkmodules.vtkIOLegacy import vtkGenericDataObjectReader
        from vtkmodules.vtkFiltersCore import vtkCellCenters
        from vtkmodules.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise ImportError("VTK import requires: pip install vtk; NPZ import does not") from exc
    reader = vtkGenericDataObjectReader() if Path(path).suffix.lower() == ".vtk" else vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(path)); reader.Update()
    if reader.GetErrorCode():
        raise ValueError(f"VTK reader failed with code {reader.GetErrorCode()}")
    data = reader.GetOutputDataObject(0)
    if data is None:
        raise ValueError("VTK file has no readable data")
    if data.IsA("vtkCompositeDataSet"):
        iterator = data.NewIterator()
        iterator.InitTraversal()
        candidates, selected = [], None
        while not iterator.IsDoneWithTraversal():
            leaf, index = iterator.GetCurrentDataObject(), iterator.GetCurrentFlatIndex()
            if leaf is not None and leaf.IsA("vtkDataSet"):
                candidates.append(index)
                if block == index:
                    selected = leaf
            iterator.GoToNextItem()
        if selected is None:
            raise ValueError(f"Select a dataset with --block; available flat indices: {candidates}")
        data = selected
    elif block is not None:
        raise ValueError("--block only applies to composite VTK data")
    if not data.IsA("vtkDataSet"):
        raise ValueError("VTK object is not a dataset")
    point_areas = None
    if measure == "area" and association == "point" and not weights_key:
        if any(data.GetCell(i).GetCellDimension() != 2 for i in range(data.GetNumberOfCells())):
            raise ValueError("Point area weighting requires a surface mesh containing only 2D cells")
        from vtkmodules.vtkFiltersCore import vtkTriangleFilter
        if not data.IsA("vtkPolyData"):
            from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
            surface = vtkGeometryFilter(); surface.SetInputData(data); surface.Update()
            data = surface.GetOutput()
        triangles = vtkTriangleFilter(); triangles.SetInputData(data)
        triangles.PassVertsOff(); triangles.PassLinesOff(); triangles.Update()
        data = triangles.GetOutput()
        xyz = vtk_to_numpy(data.GetPoints().GetData())
        ids = np.asarray([[data.GetCell(i).GetPointId(j) for j in range(3)]
                          for i in range(data.GetNumberOfCells())], dtype=np.int64)
        point_areas = lumped_vertex_areas(xyz, ids)
    elif measure != "equal" and not weights_key:
        if association != "cell":
            raise ValueError("Volume weights require cell association or a supplied weights-key")
        from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
        size = vtkCellSizeFilter()
        size.SetInputData(data); size.Update()
        data = size.GetOutput()
        weights_key = "Area" if measure == "area" else "Volume"
    attrs = data.GetPointData() if association == "point" else data.GetCellData()
    names = [attrs.GetArrayName(i) for i in range(attrs.GetNumberOfArrays())]
    pressure = attrs.GetArray(pressure_key)
    if pressure is None:
        raise ValueError(f"No {association} pressure array {pressure_key!r}; available: {names}")
    values = vtk_to_numpy(pressure)
    if values.ndim != 1:
        raise ValueError("Pressure array must be scalar")
    if association == "cell":
        centers = vtkCellCenters()
        centers.SetInputData(data); centers.Update()
        xyz = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
    else:
        # GetPoints is not provided by every structured VTK dataset.
        xyz = np.asarray([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
    if point_areas is not None:
        weights = point_areas
    elif weights_key:
        array = attrs.GetArray(weights_key)
        if array is None:
            raise ValueError(f"No weight array {weights_key!r}; available: {names}")
        weights = vtk_to_numpy(array).copy()
    else:
        weights = np.ones(len(values))
    return xyz, values.copy(), weights, dict(association=association, block=block, weight_measure=measure,
                                            weighting="lumped vertex areas" if point_areas is not None else weights_key or "uniform samples")


def convert(args):
    if Path(args.input).suffix.lower() == ".npz":
        if args.block is not None:
            raise ValueError("--block is a VTK option")
        with np.load(args.input, allow_pickle=False) as z:
            xyz = np.asarray(z[args.xyz_key], dtype=np.float64)
            values = np.asarray(z[args.pressure_key], dtype=np.float64).reshape(-1)
            if args.weights_key:
                weights = np.asarray(z[args.weights_key], dtype=np.float64).reshape(-1)
            elif args.measure == "area" and args.association == "point" and args.triangles_key in z:
                weights = lumped_vertex_areas(xyz, z[args.triangles_key])
            elif args.measure == "equal":
                weights = np.ones(len(values))
            else:
                raise ValueError("NPZ area/volume weighting needs --weights-key, or point xyz plus a triangle array (--triangles-key)")
        meta = dict(association="provided samples", weight_measure=args.measure,
                    weighting=args.weights_key or ("lumped vertex areas" if args.measure == "area" else "uniform samples"))
    else:
        xyz, values, weights, meta = vtk_samples(args.input, args.pressure_key, args.association,
                                                args.weights_key, args.measure, args.block)
    if xyz.shape != (len(values), 3) or weights.shape != values.shape or len(values) == 0:
        raise ValueError("CFD coordinates, scalar pressures and weights must align")
    if not all(np.isfinite(a).all() for a in (xyz, values, weights)) or np.any(weights < 0):
        raise ValueError("Nonfinite CFD samples or negative weights")
    if not np.isfinite(args.xyz_scale) or args.xyz_scale <= 0 or not np.isfinite(args.xyz_shift).all():
        raise ValueError("Finite positive coordinate scale and finite shift are required")
    xyz = xyz * args.xyz_scale + np.asarray(args.xyz_shift)
    weights = weights * args.xyz_scale ** {"equal": 0, "area": 2, "volume": 3}[args.measure]
    if args.kind == "cp":
        cp = values
        q = None
    else:
        if args.density is None or args.speed is None or args.reference_pressure is None:
            raise ValueError("static/kinematic input requires density, speed and explicit reference-pressure")
        if not np.isfinite([args.density, args.speed, args.reference_pressure]).all() or min(args.density, args.speed) <= 0:
            raise ValueError("Positive density/speed and finite reference pressure are required")
        q = .5 * args.density * args.speed ** 2
        denominator = q if args.kind == "static" else .5 * args.speed ** 2
        cp = (values - args.reference_pressure) / denominator
        meta.update(dynamic_pressure=q, pressure_units="Pa", density=args.density, speed=args.speed,
                    reference_static_pressure=args.reference_pressure * (args.density if args.kind == "kinematic" else 1.))
    original_count = len(cp)
    ids = np.flatnonzero(weights > 0)
    if not len(ids):
        raise ValueError("No positive-measure CFD samples")
    meta["zero_measure_samples_removed"] = int(len(cp)-len(ids))
    if args.max_points and len(ids) > args.max_points:
        ids = np.sort(np.random.default_rng(args.seed).choice(ids, args.max_points, replace=False))
    meta.update(source_sha256=sha256(args.input), input_pressure_key=args.pressure_key, input_kind=args.kind,
                xyz_scale=args.xyz_scale, xyz_shift=args.xyz_shift, original_samples=original_count,
                retained_samples=len(ids), seed=args.seed,
                coordinates="original mesh coordinate frame and units after explicit scale/shift",
                conversion="Cp=(p-p_reference)/(0.5*rho*U^2); for kinematic input divide by 0.5*U^2")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xyz=xyz[ids], cp=cp[ids], weights=weights[ids],
                        source_indices=ids, metadata=np.array(json.dumps(meta)))
    # The same reader used by training checks the final alignment and values.
    load_field(args.out)
    print(f"Saved {len(ids):,} native CFD samples: {args.out}")


def single_case(args):
    g = load_geometry(args.geometry, args.mesh_index)
    f = load_field(args.field)
    if len(args.conditions) != len(args.condition_names) or not np.isfinite(args.conditions).all():
        raise ValueError("conditions and condition-names must align")
    if min(args.val_fraction, args.test_fraction) <= 0 or args.val_fraction + args.test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and leave training samples")
    out = Path(args.out).resolve()
    if (out / "cases.json").exists():
        raise ValueError("Output already contains cases.json; use a new directory")
    # Duplicated VTK vertices must stay in one split.
    _, inverse = np.unique(f["xyz"], axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    nv, nt = max(1, round(count * args.val_fraction)), max(1, round(count * args.test_fraction))
    if nv + nt >= count:
        raise ValueError("Too few distinct coordinates for three splits")
    definition = dict(method=args.split_method, seed=args.seed,
                      requested_validation_fraction=args.val_fraction, requested_test_fraction=args.test_fraction)
    if args.split_method == "random":
        order = np.random.default_rng(args.seed).permutation(count)
        groups = np.zeros(count, dtype=np.int8)
        groups[order[:nv]] = 1; groups[order[nv:nv + nt]] = 2
        assignments = groups[inverse]
        scope = "held_out_queries_same_car"
    else:
        if args.buffer <= 0:
            raise ValueError("Blocked splitting requires a positive --buffer in mesh-diagonal units")
        axis = int(np.argmax(np.ptp(f["xyz"], axis=0))) if args.split_axis == "auto" else "xyz".index(args.split_axis)
        coord = (f["xyz"][:, axis] - g["center"][axis]) / g["scale"]
        low, high = np.quantile(np.unique(coord), [args.val_fraction, 1-args.test_fraction])
        assignments = np.full(len(coord), -1, dtype=np.int8)
        assignments[coord <= low] = 1
        assignments[coord >= high] = 2
        assignments[(coord > low + args.buffer) & (coord < high - args.buffer)] = 0
        if any(not (assignments == i).any() for i in range(3)):
            raise ValueError("Blocked split left an empty region; choose another axis, smaller buffer or fractions")
        scope = "held_out_regions_same_car"
        definition.update(axis="xyz"[axis], val_upper_coordinate_diagonal=float(low),
                          test_lower_coordinate_diagonal=float(high), buffer_fraction_diagonal=args.buffer,
                          omitted_buffer_samples=int((assignments < 0).sum()))
    definition["actual_fractions_of_all_samples"] = {s: float((assignments == i).mean())
                                                     for i, s in enumerate(("train", "val", "test"))}
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, split in enumerate(("train", "val", "test")):
        filename = f"{split}_indices.npz"
        ids = np.flatnonzero(assignments == index)
        np.savez_compressed(out / filename, indices=ids)
        entries.append(dict(id=f"{args.car_id}_{split}", car_id=args.car_id, split=split,
                            geometry=os.path.relpath(Path(args.geometry).resolve(), out), mesh_index=args.mesh_index,
                            field=os.path.relpath(Path(args.field).resolve(), out), indices=filename, conditions=args.conditions))
    json_write(out / "cases.json", dict(version="patch-pressure-data-v1", evaluation_scope=scope,
                                       condition_names=args.condition_names, cases=entries,
                                       geometry_source_sha256=g["source_sha256"], split_definition=definition))
    from pressure_data import Cases
    audit = Cases(out / "cases.json").audit
    json_write(out / "split_audit.json", audit)
    print(f"Saved {out / 'cases.json'}: {scope}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    c = s.add_parser("convert")
    c.add_argument("--input", required=True); c.add_argument("--out", required=True)
    c.add_argument("--pressure-key", required=True); c.add_argument("--xyz-key", default="xyz")
    c.add_argument("--triangles-key", default="faces")
    c.add_argument("--weights-key"); c.add_argument("--association", choices=["point", "cell"], default="point")
    c.add_argument("--measure", choices=["equal", "area", "volume"], default="area"); c.add_argument("--block", type=int)
    c.add_argument("--kind", choices=["cp", "static", "kinematic"], required=True)
    c.add_argument("--density", type=float); c.add_argument("--speed", type=float); c.add_argument("--reference-pressure", type=float)
    c.add_argument("--xyz-scale", type=float, default=1.); c.add_argument("--xyz-shift", nargs=3, type=float, default=[0., 0., 0.])
    c.add_argument("--max-points", type=int, default=0); c.add_argument("--seed", type=int, default=17); c.set_defaults(func=convert)
    d = s.add_parser("single-case")
    d.add_argument("--geometry", required=True); d.add_argument("--mesh-index", type=int, default=0)
    d.add_argument("--field", required=True); d.add_argument("--car-id", required=True); d.add_argument("--out", required=True)
    d.add_argument("--condition-names", nargs="*", required=True); d.add_argument("--conditions", nargs="*", type=float, required=True)
    d.add_argument("--val-fraction", type=float, default=.1); d.add_argument("--test-fraction", type=float, default=.2)
    d.add_argument("--split-method", choices=["blocked", "random"], default="blocked")
    d.add_argument("--split-axis", choices=["auto", "x", "y", "z"], default="auto")
    d.add_argument("--buffer", type=float, default=.02, help="Minimum training/evaluation gap divided by mesh diagonal")
    d.add_argument("--seed", type=int, default=17); d.set_defaults(func=single_case)
    return p


if __name__ == "__main__":
    p = parser(); args = p.parse_args()
    if args.seed < 0 or getattr(args, "max_points", 0) < 0:
        p.error("seed/max-points must be nonnegative")
    args.func(args)
