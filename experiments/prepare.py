import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from training.support import sha


def coordinates(labels, valid, count, identifier):
    seed = int.from_bytes(hashlib.sha256(f"42:{identifier}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    selected = []
    for cls in [0, 1]:
        eligible = np.flatnonzero(valid & (labels == cls))
        assert len(eligible) >= count // 2
        selected.append(rng.permutation(eligible)[: count // 2])
    return np.column_stack(
        np.unravel_index(np.column_stack(selected).ravel(), labels.shape)
    ).astype(np.int32)


def main():
    p = argparse.ArgumentParser(
        description="Extend the same training locations to 120k and 240k labels."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    BASE, prepared, output = (args.data, args.prepared, args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "nested_samples.npz").exists():
        raise ValueError("Sample file already exists")
    index = json.loads((BASE / "train/index.json").read_text(encoding="utf-8"))
    with (BASE / "train/common_sample_locations.csv").open(encoding="utf-8", newline="") as f:
        old_locations = list(csv.DictReader(f))
    old = dict(np.load(BASE / "train/common_samples.npz"))
    parts, audits = ([], [])
    for i, entry in enumerate(index["images"]):
        ident = f"{entry['area_id']}_{entry['year']}"
        with np.load(BASE / entry["file"]) as z:
            image = {k: z[k] for k in z.files}
        n, o = (int(image["core_size"]), int(image["core_offset"]))
        label_file = prepared / "train" / ident / "labels_mapbiomas.tif"
        manifest = json.loads((label_file.parent / "manifest.json").read_text())
        label_hash = sha(label_file)
        assert any((x["sha256"] == label_hash for x in manifest["files"]))
        with rasterio.open(label_file) as src:
            dense = src.read(1)
        coords = coordinates(dense, image["valid"][o : o + n, o : o + n], 20000, ident)
        r, c = coords.T
        previous = [
            x
            for x in old_locations
            if x["area_id"] == entry["area_id"] and int(x["year"]) == entry["year"]
        ]
        assert len(previous) == 5000
        np.testing.assert_array_equal(
            coords[:5000], [[int(x["row"]), int(x["col"])] for x in previous]
        )
        x, y = (image["features"][:, r + o, c + o].T, dense[r, c])
        take = old["image_index"] == i
        np.testing.assert_array_equal(x[:5000], old["X"][take])
        np.testing.assert_array_equal(y[:5000], old["y"][take])
        np.testing.assert_array_equal(image["labels"][r[:5000], c[:5000]], y[:5000])
        assert len(np.unique(r * n + c)) == 20000
        assert np.isfinite(x).all()
        for per_image in [5000, 10000, 20000]:
            assert np.bincount(y[:per_image]).tolist() == [per_image // 2] * 2
        parts.append(
            dict(
                X=x,
                y=y,
                row=r,
                col=c,
                rank_in_image=np.arange(20000, dtype=np.int32),
                image_index=np.full(20000, i, np.int16),
            )
        )
        audits.append(
            dict(
                image=ident,
                base_image_sha256=sha(BASE / entry["file"]),
                dense_training_label_sha256=label_hash,
                old_5000_exact=True,
                total=20000,
                unique=True,
            )
        )
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    subset = arrays["rank_in_image"] < 5000
    for k in old:
        np.testing.assert_array_equal(arrays[k][subset], old[k])
    np.savez_compressed(output / "nested_samples.npz", **arrays)
    print(output / "nested_samples.npz")


if __name__ == "__main__":
    main()
