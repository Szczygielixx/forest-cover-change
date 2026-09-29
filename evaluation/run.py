import argparse
import csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation, binary_erosion
from training.support import read, write, transition, transition_metrics, confusion, select_first

ID = "full_v2_20260916"


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def context_labels(labels):
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    known = labels != 255
    certain = binary_erosion(known, structure=cross, border_value=0)
    edge = (labels == 1) & binary_dilation(labels == 0, structure=cross) | (
        labels == 0
    ) & binary_dilation(labels == 1, structure=cross)
    return np.where(certain, np.where(edge, 2, 1), 0).astype(np.uint8)


def pair_context(a, b):
    return np.where((a == 0) | (b == 0), 0, np.where((a == 2) | (b == 2), 2, 1)).astype(np.uint8)


def flatten_metrics(metrics, prefix=""):
    keys = ["n", "accuracy", "precision", "recall", "f1", "iou"]
    result = {prefix + k: metrics[k] for k in keys}
    tn, fp, fn, tp = np.asarray(metrics["confusion_rows_truth_cols_prediction"]).ravel()
    result.update({prefix + k: int(v) for k, v in zip(["tn", "fp", "fn", "tp"], [tn, fp, fn, tp])})
    return result


def area_transition(row, pred, ref, valid, pixel_ha=0.09):
    known = valid & (ref != 255)
    pc = np.bincount(pred[valid], minlength=4)
    matched_pc = np.bincount(pred[known], minlength=4)
    rc = np.bincount(ref[known], minlength=4)
    return {
        **row,
        "pixel_area_ha": pixel_ha,
        "valid_pixels": int(valid.sum()),
        "missing_pixels": int((~valid).sum()),
        "valid_ha": float(valid.sum() * pixel_ha),
        "reference_known_pixels": int(known.sum()),
        "forest_before_ha": float((pc[2] + pc[3]) * pixel_ha),
        "forest_after_ha": float((pc[1] + pc[3]) * pixel_ha),
        "loss_ha": float(pc[2] * pixel_ha),
        "gain_ha": float(pc[1] * pixel_ha),
        "net_ha": float((int(pc[1]) - int(pc[2])) * pixel_ha),
        "matched_model_loss_ha": float(matched_pc[2] * pixel_ha),
        "matched_model_gain_ha": float(matched_pc[1] * pixel_ha),
        "reference_loss_ha": float(rc[2] * pixel_ha),
        "reference_gain_ha": float(rc[1] * pixel_ha),
        "reference_net_ha": float((int(rc[1]) - int(rc[2])) * pixel_ha),
    }


def geotiff(path, array, area, description, nodata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.tif")
    floating = np.issubdtype(array.dtype, np.floating)
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=area["crs"],
        transform=Affine(*area["transform"]),
        nodata=nodata,
        tiled=True,
        compress="deflate",
        predictor=3 if floating else 2,
    ) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)
        dst.update_tags(campaign=ID, reference="MapBiomas Collection 11", pixel_area_ha="0.09")
    temporary.replace(path)
    with rasterio.open(path) as src:
        assert src.transform == Affine(
            *area["transform"]
        ) and src.crs == rasterio.crs.CRS.from_string(area["crs"])
        np.testing.assert_array_equal(src.read(1), array)


def aggregate_methods(rows):
    aggregated = []
    for method in sorted({r["method"] for r in rows}):
        for period in ["2015_2020", "2020_2025", "both_periods"]:
            group = [
                r
                for r in rows
                if r["method"] == method and (r["period"] == period or period == "both_periods")
            ]
            variants = sorted({r["variant"] for r in group})
            entry = {
                "method": method,
                "period": period,
                "AOIs": 4,
                "variants": len(variants),
                "averaging": "equal_AOI_then_equal_period_then_equal_seed",
            }
            for metric in [
                "loss_f1",
                "loss_precision",
                "loss_recall",
                "gain_f1",
                "transition_accuracy",
            ]:
                per_seed = []
                for variant in variants:
                    values = [
                        r[metric]
                        for r in group
                        if r["variant"] == variant and r[metric] is not None
                    ]
                    if values:
                        per_seed.append(float(np.mean(values)))
                entry[metric + "_mean"] = float(np.mean(per_seed)) if per_seed else None
                entry[metric + "_seed_std"] = (
                    float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
                )
                entry[metric + "_seed_min"] = float(min(per_seed)) if per_seed else None
                entry[metric + "_seed_max"] = float(max(per_seed)) if per_seed else None
            aggregated.append(entry)
    return aggregated


def main():
    p = argparse.ArgumentParser(
        description="Evaluate saved maps against MapBiomas on held-out test areas."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--areas", type=Path, default=Path(__file__).resolve().parents[1] / "configs/areas.json"
    )
    p.add_argument(
        "--maps", action="store_true", help="Also export GeoTIFF state, score and change maps."
    )
    args = p.parse_args()
    bundle, report_dir = (args.data, args.output)
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("Choose an empty output directory")
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "maps"
    areas = {a["area_id"]: a for a in read(args.areas)["areas"] if a["split"] == "test"}
    variants = []
    for root in args.runs:
        assert read(root / "completion.json")["status"] == "succeeded"
        selected = read(root / "selection.json")
        assert selected["status"] == "selection_frozen_before_test"
        assert selected["chosen_trial"] == select_first(read(root / "tuning_summary.json"))["trial"]
        final = read(root / "final_summary.json")
        for item in final:
            frozen = next((f for f in selected["final_models"] if f["variant"] == item["variant"]))
            for field in ["method", "params", "seed", "selection_score"]:
                assert frozen[field] == item[field]
            variants.append((root, item))
    assert len({item["variant"] for root, item in variants}) == len(variants)

    def export_map(*arguments):
        if args.maps:
            geotiff(*arguments)

    valid, reference, contexts = ({}, {}, {})
    for area_id, area in areas.items():
        for year in [2015, 2020, 2025]:
            key = f"{area_id}_{year}"
            with np.load(bundle / "test" / (key + ".npz"), allow_pickle=False) as z:
                assert "labels" not in z.files
                n, o = (int(z["core_size"]), int(z["core_offset"]))
                valid[key] = z["valid"][o : o + n, o : o + n]
            with rasterio.open(args.prepared / "test" / key / "labels_mapbiomas.tif") as src:
                assert src.transform == Affine(*area["transform"])
                assert src.crs == rasterio.crs.CRS.from_string(area["crs"])
                reference[key] = src.read(1)
            assert set(np.unique(reference[key])).issubset({0, 1, 255})
            contexts[key] = context_labels(reference[key])
    state_rows = []
    change_rows = []
    context_rows = []
    area_rows = []
    common3_rows = []
    decade_rows = []
    matrices = {}
    state_maps = change_maps = 0
    for root, item in variants:
        variant, method = (item["variant"], item["method"])
        source = root / "predictions/test" / variant
        predictions = {}
        matrices[variant] = {"state": {}, "change": {}}
        for area_id, area in areas.items():
            for year in [2015, 2020, 2025]:
                key = f"{area_id}_{year}"
                with np.load(source / "state" / (key + ".npz"), allow_pickle=False) as z:
                    pred, score = (z["prediction"], z["score"])
                    np.testing.assert_array_equal(z["valid"], valid[key])
                assert (
                    pred.shape == (area["pixels"], area["pixels"])
                    and (pred[~valid[key]] == 255).all()
                )
                assert np.isfinite(score[valid[key]]).all() and np.isnan(score[~valid[key]]).all()
                predictions[key] = pred
                metric = confusion(reference[key], pred, valid[key])
                matrices[variant]["state"][key] = metric
                matched = valid[key] & (reference[key] != 255)
                base = {
                    "variant": variant,
                    "method": method,
                    "seed": item["seed"],
                    "area_id": area_id,
                    "year": year,
                }
                state_rows.append(
                    {
                        **base,
                        **flatten_metrics(metric),
                        "valid_pixels": int(valid[key].sum()),
                        "forest_ha": float(((pred == 1) & valid[key]).sum() * 0.09),
                        "matched_model_forest_ha": float(((pred == 1) & matched).sum() * 0.09),
                        "reference_forest_ha": float(
                            ((reference[key] == 1) & matched).sum() * 0.09
                        ),
                    }
                )
                export_map(
                    figures_dir / variant / "state" / (key + "_class.tif"),
                    pred,
                    area,
                    "0 nonforest; 1 forest; 255 unavailable",
                    255,
                )
                description = {
                    "ndvi": "NDVI index",
                    "rf": "forest probability from RF",
                    "svm": "signed SVM decision function",
                    "unet": "forest sigmoid output",
                }[method]
                export_map(
                    figures_dir / variant / "state" / (key + "_score.tif"),
                    score,
                    area,
                    description,
                    float("nan"),
                )
                state_maps += 1
            common_valid = (
                valid[f"{area_id}_2015"] & valid[f"{area_id}_2020"] & valid[f"{area_id}_2025"]
            )
            for a, b in [(2015, 2020), (2020, 2025)]:
                ka, kb = (f"{area_id}_{a}", f"{area_id}_{b}")
                key = f"{area_id}_{a}_{b}"
                expected = transition(predictions[ka], predictions[kb])
                truth = transition(reference[ka], reference[kb])
                mask = valid[ka] & valid[kb]
                with np.load(source / "change" / (key + ".npz"), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["transition"], expected)
                    np.testing.assert_array_equal(z["valid"], mask)
                metric = transition_metrics(truth, expected, mask)
                matrices[variant]["change"][key] = metric
                base = {
                    "variant": variant,
                    "method": method,
                    "seed": item["seed"],
                    "area_id": area_id,
                    "period": f"{a}_{b}",
                }
                cm = np.asarray(metric["confusion4"])
                change_rows.append(
                    {
                        **base,
                        "reference_known_pixels": metric["n"],
                        "transition_accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
                        **flatten_metrics(metric["loss"], "loss_"),
                        **flatten_metrics(metric["gain"], "gain_"),
                    }
                )
                area_rows.append(area_transition(base, expected, truth, mask))
                common3_rows.append(area_transition(base, expected, truth, common_valid))
                context = pair_context(contexts[ka], contexts[kb])
                for code, name in [(0, "context_unknown"), (1, "interior"), (2, "boundary_30m")]:
                    sub = mask & (context == code)
                    m = transition_metrics(truth, expected, sub)
                    counts = np.asarray(m["confusion4"])
                    n = m["n"]
                    errors = int(counts.sum() - np.trace(counts))
                    context_rows.append(
                        {
                            **base,
                            "context": name,
                            "n": n,
                            "errors": errors,
                            "error_rate": errors / n if n else None,
                            **flatten_metrics(m["loss"], "loss_"),
                            **flatten_metrics(m["gain"], "gain_"),
                        }
                    )
                export_map(
                    figures_dir / variant / "change" / (key + "_transition.tif"),
                    expected,
                    area,
                    "0 stable nonforest; 1 gain; 2 loss; 3 stable forest; 255 unavailable",
                    255,
                )
                change_maps += 1
            first, last = (f"{area_id}_2015", f"{area_id}_2025")
            decade = transition(predictions[first], predictions[last])
            truth = transition(reference[first], reference[last])
            decade_rows.append(
                area_transition(
                    {
                        "variant": variant,
                        "method": method,
                        "seed": item["seed"],
                        "area_id": area_id,
                        "period": "2015_2025",
                    },
                    decade,
                    truth,
                    common_valid,
                )
            )
            export_map(
                figures_dir / variant / "change" / f"{area_id}_2015_2025_transition.tif",
                decade,
                area,
                "0 stable nonforest; 1 gain; 2 loss; 3 stable forest; 255 unavailable",
                255,
            )
    for name, rows in [
        ("state_metrics_and_area", state_rows),
        ("change_metrics", change_rows),
        ("area_changes_pairwise", area_rows),
        ("area_changes_common3", common3_rows),
        ("area_changes_decade_common3", decade_rows),
        ("errors_by_context", context_rows),
        ("method_comparison", aggregate_methods(change_rows)),
    ]:
        csv_write(report_dir / (name + ".csv"), rows)
    write(report_dir / "confusion_matrices.json", matrices)
    print(
        f"Evaluated {len(variants)} variants: {state_maps} state maps, {change_maps} five-year change maps."
    )


if __name__ == "__main__":
    main()
