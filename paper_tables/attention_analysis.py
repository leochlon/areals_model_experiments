"""Attention range and field error by region (Section 4.5, Table K.1, Appendix K, Figures B.2 and K.1, Table A.1).

Rebuilt from results/attention_analysis/ and results/attention_analysis_extra/. Code: experiments/12_attention_analysis/.

- decoder_results.json, encoder_results.json: `theory_tests.py attention` on the exported attention profiles of the
  three checkpoints (pretrained, fine-tuned, from scratch) for the 144 DoubleDelta test cases. Each wing contributes
  the mean over its nine cases. Intervals are Student-t over the 16 wings, and p-values come from exact sign flips.
- region_decomposition.json: `recompute.py` (angle decomposition). The paper uses its variant
  nine_angle_mean_fixed_masks, whose gain is 1 - SSE(fine-tuned) / SSE(scratch) against each wing's nine-angle mean field.
- attention_analysis_extra/attention_per_wing.csv: per-wing means written by `audit_export.py` (the "every wing" claims).
- attention_analysis_extra/export_audit.json: the export's record. It holds the checkpoint paths, the training
  metadata embedded in each checkpoint (Table A.1) and the checks of the export.

Shares are printed as percentages. Distance ratios are attention-weighted mean distances divided by those of attention
in proportion to area.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from paper_tables._core import Table, as_printed, fmt, load_csv, load_json

CHECKPOINTS = [("pretrained", "Pretrained"), ("finetuned", "Fine-tuned"), ("scratch", "From scratch")]


def _per_wing(rows: list[dict], component: str, key: str) -> dict[str, dict[str, float]]:
    """{checkpoint: {wing: value}} for one attention component from attention_per_wing.csv."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["component"] == component:
            out[row["checkpoint"]][row["group"]] = float(row[key])
    return out


def _sign_flip_p(x: np.ndarray) -> float:
    """Exact one-sided sign-flip p-value for a positive mean (all 2^n sign patterns), as in theory_tests.py."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    return float(((signs * x).mean(1) >= x.mean() - 1e-15).mean())


def tables() -> list[Table]:
    results = {"Decoder": load_json("attention_analysis", "decoder_results.json"),
               "Encoder": load_json("attention_analysis", "encoder_results.json")}
    regions = load_json("attention_analysis", "region_decomposition.json")
    per_wing_rows = load_csv("attention_analysis_extra", "attention_per_wing.csv")
    audit = load_json("attention_analysis_extra", "export_audit.json")

    dec = results["Decoder"]["per_checkpoint"]
    enc = results["Encoder"]["per_checkpoint"]
    dec_contrast = results["Decoder"]["contrasts"]["finetuned - pretrained"]
    enc_contrast = results["Encoder"]["contrasts"]["finetuned - pretrained"]

    # ---------------- Table K.1 ----------------
    k1 = Table(
        key="Table K.1",
        title="Attention range on the 16 DoubleDelta test wings, no sensors: mean attention distance relative to "
        "attention in proportion to area (95% interval over wings) and the share of attention (%) by distance from "
        "the query, in units of the RMS radius R",
        columns=["", "Checkpoint", "Distance ratio", "Within 0.25R", "Beyond 0.5R", "Beyond R"],
    )
    paper_k1 = {
        "Decoder": {
            "pretrained": ((0.254, 0.251, 0.256), 51.5, 17.0, 3.5),
            "finetuned": ((0.406, 0.400, 0.412), 27.5, 40.0, 11.0),
            "scratch": ((0.786, 0.775, 0.796), 10.6, 72.4, 42.3),
            "area": (1, 3.2, 88.5, 63.6),
        },
        "Encoder": {
            "pretrained": ((0.524, 0.522, 0.527), 23.4, 46.7, 21.2),
            "finetuned": ((0.648, 0.642, 0.654), 18.9, 59.3, 31.9),
            "scratch": ((0.983, 0.982, 0.985), 4.1, 86.5, 60.6),
            "area": (1, 3.3, 88.2, 62.5),
        },
    }
    for component, res in results.items():
        per = res["per_checkpoint"]
        paper = paper_k1[component]
        for key, label in CHECKPOINTS:
            s = per[key]
            (p_ratio, p_lo, p_hi), p_near, p_tail05, p_tail1 = paper[key]
            ratio = k1.check(f"{component} {label}: distance ratio", s["ratio"]["mean"], p_ratio, 3)
            lo = k1.check(f"{component} {label}: ratio 95% lower", s["ratio"]["ci95"][0], p_lo, 3)
            hi = k1.check(f"{component} {label}: ratio 95% upper", s["ratio"]["ci95"][1], p_hi, 3)
            near = k1.check(f"{component} {label}: within 0.25R (%)", 100 * s["near"]["mean"], p_near, 1)
            tail05 = k1.check(f"{component} {label}: beyond 0.5R (%)", 100 * s["tail0.5"]["mean"], p_tail05, 1)
            tail1 = k1.check(f"{component} {label}: beyond R (%)", 100 * s["tail1"]["mean"], p_tail1, 1)
            k1.rows.append([component if key == "pretrained" else "", label,
                            f"{fmt(ratio)} [{fmt(lo)}, {fmt(hi)}]", fmt(near, 1), fmt(tail05, 1), fmt(tail1, 1)])
        # Attention in proportion to area: the same for every checkpoint (same queries, anchors and area weights).
        base = per["pretrained"]
        same = all(per[k][f] == base[f] for k, _ in CHECKPOINTS for f in ("D0", "near_base", "tail0.5_base",
                                                                          "tail1_base"))
        p_ratio, p_near, p_tail05, p_tail1 = paper["area"]
        ratio = k1.check(f"{component} in proportion to area: distance ratio", base["D0"] / base["D0"], p_ratio, 0)
        near = k1.check(f"{component} in proportion to area: within 0.25R (%)", 100 * base["near_base"], p_near, 1)
        tail05 = k1.check(f"{component} in proportion to area: beyond 0.5R (%)", 100 * base["tail0.5_base"],
                          p_tail05, 1)
        tail1 = k1.check(f"{component} in proportion to area: beyond R (%)", 100 * base["tail1_base"], p_tail1, 1)
        k1.rows.append(["", "In proportion to area", fmt(ratio, 0), fmt(near, 1), fmt(tail05, 1), fmt(tail1, 1)])
        k1.notes.append(f"{component}: area-proportional reference identical across the three checkpoints: "
                        f"{'yes' if same else 'NO'} (D0 = {base['D0']:.6f} R). Its distance ratio is D0 / D0.")
    k1.notes.append("Source: per_checkpoint.<ckpt>.{ratio, near, tail0.5, tail1} in attention_analysis/"
                    "{decoder,encoder}_results.json (wing means, Student-t over 16 wings); reference row from "
                    "near_base, tail0.5_base and tail1_base.")

    # ---------------- Attention range in the text ----------------
    text = Table(
        key="Section 4.5, Appendix K, Figures B.2 and K.1",
        title="Attention range: numbers quoted in the text (shares in %)",
        columns=["Claim", "Rebuilt", "Paper"],
    )

    def claim(label: str, value: float, paper: float, decimals: int, shown: str | None = None):
        text.check(label, value, paper, decimals)
        text.rows.append([label, shown if shown is not None else fmt(value, decimals), fmt(paper, decimals)])

    near_pre = 100 * dec["pretrained"]["near"]["mean"]
    near_area = 100 * dec["pretrained"]["near_base"]
    tail_pre, tail_ft, tail_sc = (100 * dec[k]["tail0.5"]["mean"] for k in ("pretrained", "finetuned", "scratch"))
    for where in ("Section 4.5", "Figure B.2(C)", "Appendix K"):
        claim(f"{where}: pretrained decoder attention within 0.25R (Table K.1 prints {near_pre:.1f})",
              as_printed(near_pre, 1), 52, 0, shown=f"{as_printed(near_pre, 1):.1f}")
        claim(f"{where}: attention in proportion to area within 0.25R", near_area, 3, 0)
    claim("Appendix K: pretrained decoder share beyond 0.5R", tail_pre, 17, 0)
    claim("Appendix K: attention in proportion to area beyond 0.5R", 100 * dec["pretrained"]["tail0.5_base"], 88, 0)
    for where in ("Section 4.5", "Figure K.1(C)"):
        claim(f"{where}: decoder share beyond 0.5R after pretraining", tail_pre, 17, 0)
        claim(f"{where}: decoder share beyond 0.5R after fine-tuning", tail_ft, 40, 0)
    claim("Appendix K: fine-tuned decoder share beyond 0.5R", tail_ft, 40, 0)
    claim("Appendix K: increase beyond 0.5R (fine-tuned minus pretrained)", dec_contrast["mean"], 0.230, 3)
    claim("Appendix K: increase, 95% lower", dec_contrast["ci95"][0], 0.220, 3)
    claim("Appendix K: increase, 95% upper", dec_contrast["ci95"][1], 0.239, 3)
    claim("Appendix K: exact one-sided sign-flip p, log2", math.log2(dec_contrast["p_one_sided"]), -16, 0,
          shown=f"p = {dec_contrast['p_one_sided']:.6g} = 2^{math.log2(dec_contrast['p_one_sided']):.0f}")
    claim("Appendix K: decoder distance ratio, pretrained", dec["pretrained"]["ratio"]["mean"], 0.25, 2)
    claim("Appendix K: decoder distance ratio, fine-tuned", dec["finetuned"]["ratio"]["mean"], 0.41, 2)
    claim("Appendix K: last decoder layer beyond 0.5R, pretrained", 100 * dec["pretrained"]["tail0.5_layer"][-1], 12, 0)
    claim("Appendix K: last decoder layer beyond 0.5R, fine-tuned", 100 * dec["finetuned"]["tail0.5_layer"][-1], 46, 0)
    claim("Appendix K: encoder self-attention beyond 0.5R, pretrained", 100 * enc["pretrained"]["tail0.5"]["mean"],
          47, 0)
    claim("Appendix K: encoder self-attention beyond 0.5R, fine-tuned", 100 * enc["finetuned"]["tail0.5"]["mean"],
          59, 0)
    claim("Appendix K: encoder increase beyond 0.5R", enc_contrast["mean"], 0.126, 3)
    claim("Appendix K: encoder increase, 95% lower", enc_contrast["ci95"][0], 0.123, 3)
    claim("Appendix K: encoder increase, 95% upper", enc_contrast["ci95"][1], 0.129, 3)
    for where in ("Section 4.5", "Appendix K"):
        claim(f"{where}: from-scratch decoder share beyond 0.5R", tail_sc, 72, 0)
    claim("Appendix K: from-scratch encoder share beyond 0.5R", 100 * enc["scratch"]["tail0.5"]["mean"], 86, 0)

    # "On every wing": per-wing increase in the share beyond 0.5R, from audit_export.py's per-wing means.
    n_wings = results["Decoder"]["groups"]
    for component, contrast, where in (("decoder", dec_contrast, ("Section 4.5", "Figure K.1(C)", "Appendix K")),
                                       ("encoder", enc_contrast, ("Appendix K",))):
        tail = _per_wing(per_wing_rows, component, "tail05")
        wings = sorted(tail["pretrained"])
        diff = np.array([tail["finetuned"][w] - tail["pretrained"][w] for w in wings])
        for place in where:
            claim(f"{place}: {component} increase beyond 0.5R positive on every wing (wings of {n_wings})",
                  int((diff > 0).sum()), 16, 0)
        text.notes.append(
            f"{component} per-wing increase (attention_analysis_extra/attention_per_wing.csv): {len(wings)} wings, "
            f"smallest {diff.min():+.4f}; mean {diff.mean():.10f} vs stored contrast {contrast['mean']:.10f}; exact "
            f"sign-flip p from these wings {_sign_flip_p(diff):.6g} vs stored {contrast['p_one_sided']:.6g}.")
    per_layer = dec_contrast["per_layer"]
    text.notes.append(f"Decoder increase beyond 0.5R by layer (first, last): {per_layer[0]:+.3f}, {per_layer[-1]:+.3f}; "
                      "the last layer carries most of the change.")
    text.notes.append("Source: attention_analysis/{decoder,encoder}_results.json (per_checkpoint and contrasts."
                      "'finetuned - pretrained'); 'last decoder layer' is tail0.5_layer[-1] (the second of two "
                      "cross-attention layers).")

    # ---------------- Field error by region ----------------
    region = Table(
        key="Appendix K 'Field error by region'",
        title="Share of the from-scratch model's squared error against each wing's nine-angle mean field that "
        "fine-tuned AREALS removes (%, 95% interval over 16 wings)",
        columns=["Region", "Error removed", "Wings with a reduction"],
    )
    fixed = regions["variants"]["nine_angle_mean_fixed_masks"]["regions"]
    paper_regions = [
        ("Whole surface", "all", (58.8, 53.7, 63.8)),
        ("Most curved third, ordinary pressures", "local", (55.6, 45.3, 66.0)),
        ("Flattest third, extreme pressures", "global", (61.4, 51.2, 71.6)),
    ]
    for label, key, (p_mean, p_lo, p_hi) in paper_regions:
        s = fixed[key]
        mean = region.check(f"{label}: error removed (%)", 100 * s["mean"], p_mean, 1)
        lo = region.check(f"{label}: 95% lower (%)", 100 * s["ci95"][0], p_lo, 1)
        hi = region.check(f"{label}: 95% upper (%)", 100 * s["ci95"][1], p_hi, 1)
        wings = region.check(f"{label}: every wing (wings with a reduction, of {s['wings']})", s["positive_wings"],
                             16, 0)
        region.rows.append([label, f"{fmt(mean, 1)} [{fmt(lo, 1)}, {fmt(hi, 1)}]", f"{wings}/{s['wings']}"])
    # Section 4.5: "removes a similar share ... on the most curved and the flattest thirds of every wing".
    for label, key in (("most curved third", "local"), ("flattest third", "global")):
        region.check(f"Section 4.5: error removed on the {label} of every wing (wings with a reduction, of "
                     f"{fixed[key]['wings']})", fixed[key]["positive_wings"], 16, 0)
    gap =regions["variants"]["nine_angle_mean_fixed_masks"]["contrasts"]["local_minus_global"]
    region.notes.append("Source: variants.nine_angle_mean_fixed_masks.regions.{all,local,global} in "
                        "attention_analysis/region_decomposition.json. 'local' = top curvature tercile and lower two "
                        "pressure-extremity terciles; 'global' = bottom curvature tercile and top extremity tercile.")
    region.notes.append(f"'A similar share' (Section 4.5): curved minus flat {100 * gap['mean']:+.1f} points "
                        f"[{100 * gap['ci95'][0]:+.1f}, {100 * gap['ci95'][1]:+.1f}], "
                        f"{gap['positive_wings']}/{gap['wings']} wings positive.")
    original = regions["variants"]["original"]["regions"]["all"]
    region.notes.append(f"For reference, the score against each angle's own field (variant 'original', which includes "
                        f"angle variation that a model without sensors or angle cannot predict) gives "
                        f"{100 * original['mean']:.1f}% on the whole surface; the paper does not use it.")

    # ---------------- Table A.1 row and Appendix K protocol ----------------
    a1 = Table(
        key="Table A.1 row 'Attention range and field error by region' and Appendix K protocol",
        title="Checkpoints, training budget and evaluation set",
        columns=["Item", "Rebuilt", "Paper", "Source"],
    )
    models = audit["models"]
    observations = {c["observations"] for c in audit["cases"]}
    heads = {m["config"]["model"]["heads"] for m in models.values()}
    rows = [
        ("A.1: geometry checkpoint (production, epoch)", models["pretrained"]["epoch"], 11,
         "export_audit.json models.pretrained.epoch"),
        ("A.1: field training cases, fine-tuned", models["finetuned"]["benchmark"]["train_cars"], 2160,
         "export_audit.json models.finetuned.benchmark"),
        ("A.1: field training cases, from scratch", models["scratch"]["benchmark"]["train_cars"], 2160,
         "export_audit.json models.scratch.benchmark"),
        ("A.1: epochs, fine-tuned", models["finetuned"]["benchmark"]["epochs"], 40,
         "export_audit.json models.finetuned.benchmark"),
        ("A.1: epochs, from scratch", models["scratch"]["benchmark"]["epochs"], 40,
         "export_audit.json models.scratch.benchmark"),
        ("A.1: seeds", len({models[k]["benchmark"]["seed"] for k in ("finetuned", "scratch")}), 1,
         "export_audit.json models.*.benchmark.seed"),
        ("A.1 / Appendix K: test cases", results["Decoder"]["shapes"], 144, "decoder_results.json shapes"),
        ("A.1 / Appendix K / Table K.1: test wings", results["Decoder"]["groups"], 16, "decoder_results.json groups"),
        ("Appendix K: test cases (exported)", len(audit["cases"]), 144, "export_audit.json cases"),
        ("Appendix K: angles per wing", regions["angles_per_wing"], 9, "region_decomposition.json"),
        ("Appendix K: field error by region, wings", regions["wings"], 16, "region_decomposition.json"),
        ("A.1: observations per case ('None')", max(observations), 0, "export_audit.json cases.observations"),
        ("Appendix K: decoder cross-attention layers measured ('both')", len(dec["pretrained"]["tail0.5_layer"]), 2,
         "decoder_results.json tail0.5_layer"),
        ("Appendix K: attention heads ('all four')", max(heads) if len(heads) == 1 else math.nan, 4,
         "export_audit.json models.*.config.model.heads"),
    ]
    for label, value, paper, source in rows:
        a1.check(label, value, paper, 0)
        a1.rows.append([label, f"{value:g}", f"{paper:g}", source])
    a1.notes.append(f"Observations per case (Table A.1 'None'): {sorted(observations)}; decoder depth in each "
                    f"checkpoint's config: {sorted({m['config']['model']['decoder_depth'] for m in models.values()})}.")
    base_ok = all(models[k]["benchmark"]["base_checkpoint"] == models["pretrained"]["path"] for k in ("finetuned",
                                                                                                    "scratch"))
    a1.notes.append(f"Both field models record the pretrained checkpoint as their base: "
                    f"{'yes' if base_ok else 'NO'}; from-scratch flag: fine-tuned "
                    f"{models['finetuned']['benchmark']['scratch']}, from scratch {models['scratch']['benchmark']['scratch']}.")
    checks = audit["checks"]
    a1.notes.append("Export checks (export_audit.json): matched architecture and training "
                    f"{checks['matched_checkpoint_architecture_and_training']}; native prediction max error "
                    f"fine-tuned {checks['finetuned_native_prediction_max_error']}, scratch "
                    f"{checks['scratch_native_prediction_max_error']}; hidden field values excluded "
                    f"{checks['hidden_field_values_excluded']}; all 144 cases exported {checks['all_144_cases_exported']}.")
    worst = max(m["capture"][side]["max_output_error"] for m in models.values() for side in ("encoder", "decoder"))
    a1.notes.append(f"Recomputed attention reproduces the fused attention output to {worst:.2g} (max over checkpoints "
                    "and layers). The 2,048 area-sampled queries per case are fixed in run_real.py; no "
                    "results file records that count.")

    return [k1, text, region, a1]
