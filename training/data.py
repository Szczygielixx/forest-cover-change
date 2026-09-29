from pathlib import Path
import json
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_images(root, split):
    root = Path(root)
    index = read_json(root / split / "index.json")
    images = []
    for entry in index["images"]:
        with np.load(root / entry["file"], allow_pickle=False) as z:
            image = {k: z[k] for k in z.files}
        image["key"] = f"{entry['area_id']}_{entry['year']}"
        assert int(image["year"]) in (2015, 2020), "Experiment must not read 2025/test data"
        n, o = (int(image["core_size"]), int(image["core_offset"]))
        image["core_valid"] = image["valid"][o : o + n, o : o + n].astype(bool)
        assert image["features"].shape == (9, n + 2 * o, n + 2 * o)
        assert image["labels"].shape == (n, n)
        assert np.isfinite(image["features"][:, image["valid"]]).all()
        if split == "train":
            assert np.all(image["core_valid"][image["labels"] != 255])
        images.append(image)
    return images


def common_samples(root):
    root = Path(root)
    with np.load(root / "train/common_samples.npz", allow_pickle=False) as z:
        x, y, ranks, indices = (z[k] for k in ("X", "y", "rank_in_image", "image_index"))
    assert x.shape == (60000, 9) and np.isfinite(x).all()
    assert np.array_equal(np.bincount(y), [30000, 30000])
    for i in range(12):
        assert np.array_equal(np.sort(ranks[indices == i]), np.arange(5000))
        assert np.array_equal(np.bincount(y[indices == i]), [2500, 2500])
    norm = read_json(root / "train/normalization_by_budget.json")["budgets"]["60000"]
    mean, std = (np.asarray(norm["mean"]), np.asarray(norm["std"]))
    np.testing.assert_allclose(x.mean(axis=0, dtype=np.float64), mean, rtol=1e-06)
    np.testing.assert_allclose(x.std(axis=0, dtype=np.float64), std, rtol=1e-06)
    assert (std > 0).all()
    return (x, y, mean.astype(np.float32), std.astype(np.float32))


def normalized_context(image, mean, std):
    x = (image["features"] - mean[:, None, None]) / std[:, None, None]
    return np.where(image["valid"][None], x, 0).astype(np.float32)


def tiles(n, size=64):
    return [(r, c) for r in range(0, n, size) for c in range(0, n, size)]


def patch(context, labels, row, col):
    x = np.zeros((9, 128, 128), dtype=np.float32)
    source = context[:, row : row + 128, col : col + 128]
    x[:, : source.shape[1], : source.shape[2]] = source
    y = np.full((64, 64), 255, dtype=np.uint8)
    source_y = labels[row : row + 64, col : col + 64]
    y[: source_y.shape[0], : source_y.shape[1]] = source_y
    return (x, y)


def confusion(y, p, valid):
    mask = np.asarray(valid, bool) & (y != 255) & (p != 255)
    cm = np.bincount(2 * y[mask].astype(int) + p[mask].astype(int), minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    div = lambda a, b: a / b if b else None
    return {
        "n": int(mask.sum()),
        "confusion_rows_truth_cols_prediction": cm.tolist(),
        "accuracy": div(tp + tn, mask.sum()),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
    }


def loss_map(before, after):
    valid = (before != 255) & (after != 255)
    return np.where(valid, (before == 1) & (after == 0), 255).astype(np.uint8)
