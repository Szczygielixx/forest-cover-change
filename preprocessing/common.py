from __future__ import annotations
import csv
import hashlib
import json
import os
import warnings
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as coordinates
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def config():
    cfg = load(ROOT / "configs/data.json")
    p = cfg["preprocessing"]
    fixed = {
        "reject_fmask_bits": [1, 2, 3, 4],
        "reject_aerosol_levels": [3],
        "retain_water": True,
        "quality_fill": 255,
        "reflectance_fill": -9999,
        "reflectance_saturation": 12000,
        "reflectance_scale": 0.0001,
        "reflectance_offset": 0.0,
        "reject_indices_outside_minus1_plus1": True,
        "same_day": "first_complete_valid_vector_in_lexicographic_granule_order",
        "composite": "per_band_median_then_indices",
    }
    if any((p.get(k) != v for k, v in fixed.items())):
        raise ValueError("QA/compositing rule change requires an explicit implementation review")
    if cfg["hls"]["bands"] != BANDS or cfg["hls"]["platform"] != "LANDSAT-8":
        raise ValueError("This pipeline is restricted to Landsat 8 and the reviewed six bands")
    return cfg


def fingerprint():
    return sha(ROOT / "configs/data.json")


def profile(area, context=0):
    t = Affine(*area["transform"]) * Affine.translation(-context, -context)
    n = area["pixels"] + 2 * context
    return dict(
        driver="GTiff",
        width=n,
        height=n,
        crs=area["crs"],
        transform=t,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )


def save_tif(path, values, grid, names, nodata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if values.ndim == 2:
        values = values[None]
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(
        tmp, "w", **grid, count=len(values), dtype=values.dtype.name, nodata=nodata
    ) as dst:
        dst.write(values)
        dst.descriptions = tuple(names)
    os.replace(tmp, path)


def qa_mask(qa):
    if qa.dtype != np.uint8:
        raise ValueError("QA must be uint8")
    return (qa != 255) & (qa & 30 == 0) & (qa >> 6 & 3 != 3)


def translate_grid(source, target_crs):
    t = source.transform
    corners = [(0, 0), (source.width, 0), (0, source.height), (source.width, source.height)]
    old = np.array([t * xy for xy in corners])
    x, y = coordinates(source.crs, target_crs, old[:, 0].tolist(), old[:, 1].tolist())
    delta = np.column_stack([x, y]) - old
    if not np.allclose(delta, delta[0], atol=0.0001, rtol=0):
        raise ValueError("Different UTM zone requires an explicit resampling decision")
    return Affine(
        t.a, t.b, t.c + round(float(delta[0, 0]), 6), t.d, t.e, t.f + round(float(delta[0, 1]), 6)
    )


def aligned_window(source, grid):
    t = translate_grid(source, grid["crs"])
    g = grid["transform"]
    if not np.allclose([t.a, t.b, t.d, t.e], [g.a, g.b, g.d, g.e], atol=1e-07, rtol=0):
        raise ValueError("Pixel orientation/resolution differs")
    col, row = ~t * (g.c, g.f)
    if not np.allclose([col, row], np.rint([col, row]), atol=1e-05, rtol=0):
        raise ValueError("Fractional pixel offset")
    return Window(round(col), round(row), grid["width"], grid["height"])


def binary(classes, cfg):
    with (ROOT / cfg["labels"]["legend"]).open(encoding="utf-8-sig", newline="") as f:
        known = [int(r["class_id"]) for r in csv.DictReader(f)]
    out = np.full(classes.shape, 255, dtype=np.uint8)
    valid = np.isin(classes, known) & ~np.isin(classes, cfg["labels"]["unknown_codes"])
    out[valid] = np.isin(classes[valid], cfg["labels"]["forest_codes"]).astype(np.uint8)
    return out


def indices(bands, epsilon=1e-06):
    nir = bands[3]
    result = []
    for j in (2, 4, 5):
        denominator = nir + bands[j]
        z = np.full(nir.shape, np.nan, dtype=np.float32)
        np.divide(
            nir - bands[j],
            denominator,
            out=z,
            where=np.isfinite(denominator) & (np.abs(denominator) > epsilon),
        )
        result.append(z)
    return np.stack(result)


def merge_day(day, raw, qa):
    valid = qa_mask(qa) & np.all((raw != -9999) & (raw != 12000), axis=0)
    take = valid & ~np.isfinite(day[0])
    day[:, take] = raw[:, take].astype(np.float32) * np.float32(0.0001)
    return (valid, take)


def composite_daily(daily, minimum=3, epsilon=1e-06):
    days = sorted(daily)
    stack = np.stack([daily[d] for d in days])
    good = np.isfinite(stack).all(axis=1)
    if not np.array_equal(np.isfinite(stack).any(axis=1), good):
        raise ValueError("Partial daily vectors are forbidden")
    count = good.sum(axis=0).astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    ix = indices(med, epsilon)
    features = np.concatenate([med, ix])
    valid = (count >= minimum) & np.isfinite(features).all(axis=0) & (np.abs(ix) <= 1).all(axis=0)
    features[:, ~valid] = np.nan
    ordinal = np.array([date.fromisoformat(d).timetuple().tm_yday for d in days], dtype=np.uint16)[
        :, None, None
    ]
    first = np.where(good, ordinal, 65535).min(axis=0).astype(np.uint16)
    last = np.where(good, ordinal, 0).max(axis=0).astype(np.uint16)
    last[count == 0] = 65535
    return (features, valid, count, first, last)


def evidence(path):
    path = Path(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def good_cache(path, meta, key):
    if not Path(path).exists() or not Path(meta).exists():
        return False
    m = load(meta)
    if m.get("key") != key:
        raise ValueError(f"Cache rules differ: {path}")
    if m.get("sha256") != sha(path):
        raise ValueError(f"Cache checksum differs: {path}")
    return True
