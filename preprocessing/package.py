from __future__ import annotations
import csv
import hashlib
import os
from pathlib import Path
import numpy as np
import rasterio
from .common import ROOT, load, write, sha, config, fingerprint


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Empty CSV")
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def supervised_sample(label, valid, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for c in (0, 1):
        candidates = np.flatnonzero(valid & (label == c))
        if len(candidates) < per_class:
            raise ValueError(f"Class {c}: {len(candidates)} < {per_class}")
        selected.append(rng.permutation(candidates)[:per_class])
    flat = np.column_stack(selected).ravel()
    return np.column_stack(np.unravel_index(flat, label.shape)).astype(np.int32)


def read_prepared(area, year):
    path = ROOT / "data/prepared" / area["split"] / f"{area['area_id']}_{year}"
    manifest = load(path / "manifest.json")
    if manifest["config_sha256"] != fingerprint():
        raise ValueError("Prepared config mismatch")
    for f in manifest["files"]:
        if sha(ROOT / f["path"]) != f["sha256"]:
            raise ValueError("Prepared checksum mismatch")
    with rasterio.open(path / "features_context.tif") as src:
        features = src.read()
    with rasterio.open(path / "valid_context.tif") as src:
        valid = src.read(1).astype(bool)
    with rasterio.open(path / "labels_mapbiomas.tif") as src:
        labels = src.read(1)
    return (path, manifest, features, valid, labels)


def package():
    cfg = config()
    budget_cfg = load(ROOT / "configs/sampling.json")
    areas = load(ROOT / "configs/areas.json")["areas"]
    scene_rows = []
    summaries = []
    all_x = []
    all_y = []
    all_rank = []
    all_image = []
    export = ROOT / "data/bundles" / cfg["dataset_id"]
    export.mkdir(parents=True, exist_ok=True)
    if (export / "release.json").exists():
        raise ValueError("Release already exists; verify or create an explicit revision")
    portable = {"train": [], "validation": [], "test": []}
    ctx = cfg["preprocessing"]["context_pixels"]
    npix = cfg["candidate_grid"]["aoi_pixels"]
    train_index = 0
    sample_table = []
    for area in areas:
        for year in area["years"]:
            path, m, features, valid, labels = read_prepared(area, year)
            if m["valid_fraction_core"] < cfg["selection"]["min_qa_three_day_fraction_each_year"]:
                raise ValueError(
                    f"Full-band quality below declared support: {area['area_id']} {year}; preserve outputs and review explicitly"
                )
            ident = f"{area['area_id']}_{year}"
            corevalid = valid[ctx : ctx + npix, ctx : ctx + npix]
            if not np.array_equal(np.isfinite(features).all(axis=0), valid):
                raise ValueError("Feature-mask mismatch")
            labels_out = np.full(labels.shape, 255, dtype=np.uint8)
            if area["split"] == "train":
                coords = supervised_sample(
                    labels,
                    corevalid,
                    budget_cfg["per_class_per_aoi_year"],
                    stable_seed(f"{cfg['labels']['seed']}:{ident}"),
                )
                r, c = coords.T
                y = labels[r, c]
                x = features[:, r + ctx, c + ctx].T
                labels_out[r, c] = y
                ranks = np.arange(len(y), dtype=np.int32)
                all_x.append(x)
                all_y.append(y)
                all_rank.append(ranks)
                all_image.append(np.full(len(y), train_index, dtype=np.int16))
                for rank, (rr, cc), label in zip(ranks, coords, y):
                    sample_table.append(
                        {
                            "area_id": area["area_id"],
                            "year": year,
                            "row": int(rr),
                            "col": int(cc),
                            "label": int(label),
                            "rank_in_image": int(rank),
                        }
                    )
                train_index += 1
            elif area["split"] == "validation":
                labels_out = labels
            dest = export / area["split"] / f"{ident}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            contents = {
                "features": features,
                "valid": valid,
                "core_offset": np.int32(ctx),
                "core_size": np.int32(npix),
                "area_id": np.array(area["area_id"]),
                "year": np.int32(year),
            }
            if area["split"] != "test":
                contents["labels"] = labels_out
            np.savez_compressed(dest, **contents)
            portable[area["split"]].append(
                {
                    "area_id": area["area_id"],
                    "year": year,
                    "file": dest.relative_to(export).as_posix(),
                    "prepared_manifest_sha256": sha(path / "manifest.json"),
                    "valid_fraction": m["valid_fraction_core"],
                }
            )
            summaries.append(
                {
                    k: m[k]
                    for k in (
                        "area_id",
                        "split",
                        "year",
                        "valid_fraction_core",
                        "eligible_label_counts",
                        "valid_days_quantiles_core",
                    )
                }
            )
            for source in m["sources"]:
                scene_rows.append(
                    {
                        "area_id": area["area_id"],
                        "split": area["split"],
                        "year": year,
                        "granule_id": source["granule_id"],
                        "date": source["date"],
                        "qa_valid_pixels_with_context": source["qa_valid_pixels_with_context"],
                        "bands_read": source["bands_downloaded"],
                        "complete_valid_pixels_with_context": source.get(
                            "complete_valid_pixels_with_context", ""
                        ),
                        "pixels_used_after_day_deduplication": source.get(
                            "pixels_contributed_after_day_deduplication", 0
                        ),
                        "skip_reason": source.get("skip_reason", ""),
                    }
                )
    X = np.concatenate(all_x)
    y = np.concatenate(all_y)
    rank = np.concatenate(all_rank)
    image = np.concatenate(all_image)
    target = budget_cfg["target_total"]
    if (
        len(y) != target
        or np.bincount(y).tolist() != [target // 2, target // 2]
        or (not np.isfinite(X).all())
    ):
        raise ValueError("Common sample mismatch")
    common = export / "train/common_samples.npz"
    np.savez_compressed(common, X=X, y=y, rank_in_image=rank, image_index=image)
    save_csv(export / "train/common_sample_locations.csv", sample_table)
    budgets = {}
    for budget in budget_cfg["normalization_budgets"]:
        if budget % 12:
            raise ValueError("Budget must divide into 12 images")
        take = rank < budget // 12
        xx = X[take].astype(np.float64)
        mean = xx.mean(axis=0)
        std = xx.std(axis=0, ddof=0)
        std[std == 0] = 1
        budgets[str(budget)] = {
            "count": int(take.sum()),
            "per_image": budget // 12,
            "class_counts": np.bincount(y[take], minlength=2).tolist(),
            "selection": "rank_in_image < per_image",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "ddof": 0,
        }
    write(
        export / "train/normalization_by_budget.json",
        {
            "feature_order": cfg["preprocessing"]["feature_order"],
            "fit_source": "only_selected_train_pixels",
            "budgets": budgets,
        },
    )
    for split in portable:
        write(
            export / split / "index.json",
            {
                "split": split,
                "dataset_id": cfg["dataset_id"],
                "config_sha256": fingerprint(),
                "feature_order": cfg["preprocessing"]["feature_order"],
                "images": portable[split],
                "labels": "absent"
                if split == "test"
                else "sparse_common_supervised_only"
                if split == "train"
                else "auxiliary_MapBiomas",
            },
        )
    write(
        export / "release.json",
        {
            "dataset_id": cfg["dataset_id"],
            "training_labels": len(y),
            "config_sha256": fingerprint(),
            "images": {k: len(v) for k, v in portable.items()},
        },
    )
    print(f"Prepared {len(y)} labels and {sum(map(len, portable.values()))} images: {export}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Package prepared composites and shared labels."
    ).parse_args()
    package()
