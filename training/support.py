import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import (
    common_samples as common_samples,
    confusion,
    load_images as load_images,
    normalized_context,
    patch,
    tiles,
)


def report(stage, *progress, **values):
    print(stage, *progress, values or "", flush=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_test(root, selection_path):
    selection = read(selection_path)
    assert selection["status"] == "selection_frozen_before_test"
    index = read(Path(root) / "test/index.json")
    images = []
    for entry in index["images"]:
        with np.load(Path(root) / entry["file"], allow_pickle=False) as z:
            assert "labels" not in z.files, "Test labels must never reach a training worker"
            im = {k: z[k] for k in z.files}
        n, o = (int(im["core_size"]), int(im["core_offset"]))
        im["core_valid"] = im["valid"][o : o + n, o : o + n].astype(bool)
        assert im["features"].shape == (9, n + 2 * o, n + 2 * o)
        assert np.isfinite(im["features"][:, im["valid"]]).all()
        assert int(im["year"]) in (2015, 2020, 2025)
        im["key"] = f"{entry['area_id']}_{entry['year']}"
        images.append(im)
    assert len(images) == 12
    return images


def transition(before, after):
    valid = (before != 255) & (after != 255)
    return np.where(valid, 2 * before.astype(int) + after.astype(int), 255).astype(np.uint8)


def transition_metrics(truth, pred, valid):
    mask = valid & (truth != 255) & (pred != 255)
    cm = np.bincount(4 * truth[mask].astype(int) + pred[mask].astype(int), minlength=16).reshape(
        4, 4
    )
    collapsed = np.array([0, 2, 1, 0])
    cm3 = np.zeros((3, 3), dtype=np.int64)
    for a in range(4):
        for b in range(4):
            cm3[collapsed[a], collapsed[b]] += cm[a, b]
    return {
        "n": int(mask.sum()),
        "confusion4": cm.tolist(),
        "confusion3_unchanged_loss_gain": cm3.tolist(),
        "loss": confusion((truth == 2).astype(np.uint8), (pred == 2).astype(np.uint8), mask),
        "gain": confusion((truth == 1).astype(np.uint8), (pred == 1).astype(np.uint8), mask),
    }


def pairs(images):
    for area in sorted({str(im["area_id"]) for im in images}):
        by_year = {int(im["year"]): im for im in images if str(im["area_id"]) == area}
        for a, b in ((2015, 2020), (2020, 2025)):
            if a in by_year and b in by_year:
                yield (area, by_year[a], by_year[b])


def validation_metrics(images, predictions):
    result = {
        "reference": "MapBiomas_Collection_11_auxiliary_validation",
        "state": {},
        "change": {},
    }
    for im in images:
        result["state"][im["key"]] = confusion(
            im["labels"], predictions[im["key"]]["prediction"], im["core_valid"]
        )
    scores = []
    for area, before, after in pairs(images):
        truth = transition(before["labels"], after["labels"])
        pred = transition(
            predictions[before["key"]]["prediction"], predictions[after["key"]]["prediction"]
        )
        stats = transition_metrics(truth, pred, before["core_valid"] & after["core_valid"])
        assert stats["loss"]["f1"] is not None, "Undefined validation loss F1"
        assert sum(stats["loss"]["confusion_rows_truth_cols_prediction"][1]) > 0
        scores.append(stats["loss"]["f1"])
        result["change"][area] = stats
    assert len(scores) == 2
    result["selection_score"] = float(np.mean(scores))
    return result


def select_first(candidates, tolerance=1e-06):
    winner = candidates[0]
    for candidate in candidates[1:]:
        if candidate["selection_score"] > winner["selection_score"] + tolerance:
            winner = candidate
    return winner


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    generator.set_state(state["loader"].cpu())


class SparseDataset(Dataset):
    def __init__(self, images, mean, std, expected_count=60000):
        self.images = images
        self.contexts = [normalized_context(im, mean, std) for im in images]
        self.items = []
        self.label_count = 0
        for i, im in enumerate(images):
            assert int(im["core_offset"]) == 32
            for r, c in tiles(int(im["core_size"])):
                count = int((im["labels"][r : r + 64, c : c + 64] != 255).sum())
                if count:
                    self.items.append((i, r, c))
                    self.label_count += count
        assert self.label_count == expected_count

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, r, c = self.items[index]
        x, y = patch(self.contexts[i], self.images[i]["labels"], r, c)
        if random.random() < 0.5:
            x, y = (x[:, :, ::-1], y[:, ::-1])
        turns = random.randrange(4)
        return (
            torch.from_numpy(np.rot90(x, turns, axes=(1, 2)).copy()),
            torch.from_numpy(np.rot90(y, turns).copy()),
        )


def predict(method, model, images, mean, std, device="cuda", batch_size=16):
    result = {}
    for im in images:
        n, o = (int(im["core_size"]), int(im["core_offset"]))
        valid = im["core_valid"]
        score = np.full((n, n), np.nan, np.float32)
        pred = np.full((n, n), 255, np.uint8)
        if method == "unet":
            context = normalized_context(im, mean, std)
            owned = tiles(n)
            model.eval()
            blank = np.full((n, n), 255, np.uint8)
            with torch.no_grad():
                for start in range(0, len(owned), batch_size):
                    positions = owned[start : start + batch_size]
                    xx = np.stack([patch(context, blank, r, c)[0] for r, c in positions])
                    outputs = model(torch.from_numpy(xx).to(device)).sigmoid().cpu().numpy()
                    for (r, c), value in zip(positions, outputs):
                        h, w = (min(64, n - r), min(64, n - c))
                        score[r : r + h, c : c + w] = value[:h, :w]
            pred[valid] = (score[valid] >= 0.5).astype(np.uint8)
        elif method == "ndvi":
            score = im["features"][6, o : o + n, o : o + n].copy()
            pred[valid] = (score[valid] >= model["threshold"]).astype(np.uint8)
        else:
            x = im["features"][:, o : o + n, o : o + n].transpose(1, 2, 0)[valid]
            if method == "svm":
                x = np.ascontiguousarray((x - mean) / std, dtype=np.float32)
            pp, ss = (np.empty(len(x), np.uint8), np.empty(len(x), np.float32))
            for start in range(0, len(x), 32768):
                chunk = x[start : start + 32768]
                if method == "rf":
                    probabilities = model.predict_proba(chunk)
                    pp[start : start + len(chunk)] = model.classes_[probabilities.argmax(axis=1)]
                    ss[start : start + len(chunk)] = probabilities[:, 1]
                elif method == "svm":
                    pp[start : start + len(chunk)] = np.asarray(model.predict(chunk)).reshape(-1)
                    ss[start : start + len(chunk)] = np.asarray(
                        model.decision_function(chunk)
                    ).reshape(-1)
                else:
                    raise ValueError(method)
            pred[valid], score[valid] = (pp, ss)
        score[~valid] = np.nan
        assert np.isfinite(score[valid]).all()
        assert set(np.unique(pred)).issubset({0, 1, 255})
        result[im["key"]] = {"prediction": pred, "score": score, "valid": valid}
    return result


def save_predictions(directory, images, predictions, variant):
    directory = Path(directory)
    states, changes = ([], [])
    for im in images:
        filename = im["key"] + ".npz"
        npz_save(
            directory / "state" / filename,
            **predictions[im["key"]],
            area_id=im["area_id"],
            year=im["year"],
            variant=variant,
        )
        states.append(filename)
    for area, before, after in pairs(images):
        a, b = (int(before["year"]), int(after["year"]))
        value = transition(
            predictions[before["key"]]["prediction"], predictions[after["key"]]["prediction"]
        )
        filename = f"{area}_{a}_{b}.npz"
        npz_save(
            directory / "change" / filename,
            transition=value,
            valid=value != 255,
            area_id=area,
            years=[a, b],
            variant=variant,
        )
        changes.append(filename)
    write(
        directory / "index.json",
        {
            "variant": variant,
            "state": states,
            "change": changes,
            "transition_codes": {
                "0": "nonforest_stable",
                "1": "gain",
                "2": "loss",
                "3": "forest_stable",
                "255": "nodata",
            },
        },
    )


def synchronize(method, device="cuda"):
    if device == "cuda":
        if method == "svm":
            import cupy

            cupy.cuda.runtime.deviceSynchronize()
        elif method == "unet":
            torch.cuda.synchronize()


def prediction_timed(method, model, images, mean, std, device="cuda"):
    synchronize(method, device)
    start = time.perf_counter()
    outputs = predict(method, model, images, mean, std, device)
    synchronize(method, device)
    return (outputs, time.perf_counter() - start)
