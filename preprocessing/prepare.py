from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import contextlib
import io
import logging
import math
import os
import time
import earthaccess
import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform, transform_bounds, transform_geom, reproject, Resampling
from rasterio.windows import Window, from_bounds
from .common import (
    ROOT,
    BANDS,
    now,
    load,
    write,
    sha,
    config,
    fingerprint,
    profile,
    save_tif,
    qa_mask,
    aligned_window,
    binary,
    evidence,
    good_cache,
    merge_day,
    composite_daily,
)

HTTP_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_TIMEOUT="45",
    GDAL_HTTP_CONNECTTIMEOUT="15",
    GDAL_HTTP_MAX_RETRY="2",
    CPL_VSIL_CURL_USE_HEAD="NO",
)
DATA = ROOT / "data"
REPORT = ROOT / "reports/data"
AUTH = None


def event(stage, message, **extra):
    print(stage, message, extra or "", flush=True)


def authenticate():
    global AUTH
    if AUTH is None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            AUTH = earthaccess.login(strategy="netrc")
        if not AUTH.authenticated:
            raise RuntimeError("Earthdata authentication failed")
    return AUTH


def polygon(bounds):
    x0, y0, x1, y1 = bounds
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def candidates():
    cfg = config()
    path = ROOT / "data/candidates.json"
    if path.exists():
        result = load(path)
        if result["config_sha256"] != fingerprint():
            raise ValueError("Candidate config changed")
        return result
    all_areas = []
    grid = cfg["candidate_grid"]
    n = grid["aoi_pixels"]
    res = grid["resolution_m"]
    for region, spec in cfg["regions"].items():
        cx, cy = transform(
            "EPSG:4326", spec["crs"], [spec["center_lonlat"][0]], [spec["center_lonlat"][1]]
        )
        cx = round(cx[0] / res) * res
        cy = round((cy[0] - 10000000) / res) * res + 10000000
        for r in range(grid["rows"]):
            for c in range(grid["cols"]):
                left = cx + (c - (grid["cols"] - 1) / 2) * grid["step_pixels"] * res - n / 2 * res
                top = cy + ((grid["rows"] - 1) / 2 - r) * grid["step_pixels"] * res + n / 2 * res
                t = Affine(res, 0, left, 0, -res, top)
                bounds = array_bounds(n, n, t)
                geom = transform_geom(spec["crs"], "EPSG:4326", polygon(bounds))
                bbox = transform_bounds(spec["crs"], "EPSG:4326", *bounds, densify_pts=21)
                all_areas.append(
                    {
                        "candidate_id": f"{region}_C{r * grid['cols'] + c + 1:02d}",
                        "region": region,
                        "crs": spec["crs"],
                        "transform": list(t)[:6],
                        "pixels": n,
                        "bounds_utm": list(bounds),
                        "bbox_wgs84": list(bbox),
                        "geometry": geom,
                    }
                )
    result = {
        "created_utc": now(),
        "config_sha256": fingerprint(),
        "selection_before_model_results": True,
        "areas": all_areas,
    }
    write(path, result)
    write(
        ROOT / "data/candidates.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": a["geometry"],
                    "properties": {k: v for k, v in a.items() if k != "geometry"},
                }
                for a in all_areas
            ],
        },
    )
    return result


def catalog_one(region, year):
    cfg = config()
    areas = [a for a in candidates()["areas"] if a["region"] == region]
    path = DATA / "catalog" / f"{region}_{year}.json"
    if path.exists():
        m = load(path)
        if m["config_sha256"] != fingerprint():
            raise ValueError("Catalog config differs")
        return m
    boxes = [a["bbox_wgs84"] for a in areas]
    bbox = (
        min((b[0] for b in boxes)) - 0.015,
        min((b[1] for b in boxes)) - 0.015,
        max((b[2] for b in boxes)) + 0.015,
        max((b[3] for b in boxes)) + 0.015,
    )
    period = (
        f"{year}-{cfg['season']['start']}T00:00:00Z",
        f"{year}-{cfg['season']['end']}T23:59:59Z",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        results = earthaccess.search_data(
            short_name="HLSL30", version="2.0", bounding_box=bbox, temporal=period, count=-1
        )
    granules = []
    excluded = []
    zone = cfg["regions"][region]["crs"][-2:]
    for g in results:
        m = g["umm"]
        ident = m["GranuleUR"]
        platforms = [x["ShortName"] for x in m.get("Platforms", [])]
        if platforms != ["LANDSAT-8"]:
            excluded.append(
                {"id": ident, "reason": "not_exclusively_Landsat_8", "platforms": platforms}
            )
            continue
        attrs = {x["Name"]: x.get("Values", []) for x in m.get("AdditionalAttributes", [])}
        ids = attrs.get("LANDSAT_PRODUCT_ID", [])
        if not ids or any((not x.startswith("LC08_") for x in ids)):
            raise ValueError("Landsat 8 provenance unconfirmed: " + ident)
        if ident.split(".")[2][1:3] != zone:
            excluded.append(
                {"id": ident, "reason": "different_UTM_zone_avoid_reflectance_resampling"}
            )
            continue
        links = g.data_links()
        assets = {}
        for band in [*BANDS, "Fmask"]:
            matches = [u for u in links if u.startswith("https://") and u.endswith(f".{band}.tif")]
            if len(matches) != 1:
                raise ValueError("Missing/ambiguous asset: " + ident + " " + band)
            assets[band] = matches[0]
        temporal = m["TemporalExtent"]
        stamp = temporal.get("RangeDateTime", {}).get("BeginningDateTime") or temporal.get(
            "SingleDateTime"
        )
        if not stamp:
            raise ValueError("Missing acquisition date")
        granules.append(
            {
                "id": ident,
                "date": stamp[:10],
                "platforms": platforms,
                "landsat_product_ids": ids,
                "assets": assets,
                "umm": m,
            }
        )
    granules = sorted({g["id"]: g for g in granules}.values(), key=lambda g: (g["date"], g["id"]))
    with (ROOT / "configs/scenes.csv").open(encoding="utf-8", newline="") as stream:
        expected = {
            row["granule_id"]
            for row in csv.DictReader(stream)
            if row["region"] == region and int(row["year"]) == year
        }
    available = {g["id"] for g in granules}
    if not expected or not expected.issubset(available):
        raise ValueError("Some recorded source scenes are unavailable in the catalog")
    granules = [g for g in granules if g["id"] in expected]
    if not granules:
        raise ValueError("Empty Landsat 8 catalog")
    result = {
        "created_utc": now(),
        "config_sha256": fingerprint(),
        "region": region,
        "year": year,
        "query": {
            "short_name": "HLSL30",
            "version": "2.0",
            "bounding_box": bbox,
            "temporal": period,
        },
        "granules": granules,
        "excluded": excluded,
    }
    write(path, result)
    event("catalog", f"{region} {year}", granules=len(granules), excluded=len(excluded))
    return result


def download_qa(g):
    path = DATA / "source_qa" / f"{g['id']}.Fmask.tif"
    meta = path.with_suffix(".json")
    key = {"url": g["assets"]["Fmask"]}
    if good_cache(path, meta, key):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with authenticate().get_session() as session:
                with session.get(key["url"], stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    with path.with_suffix(".part").open("wb") as out:
                        for chunk in response.iter_content(1024 * 1024):
                            out.write(chunk)
                    headers = {
                        k: response.headers.get(k)
                        for k in ("ETag", "Last-Modified", "Content-Length")
                    }
            with rasterio.open(path.with_suffix(".part")) as src:
                if (
                    src.count != 1
                    or src.dtypes != ("uint8",)
                    or src.nodata != 255
                    or (src.res != (30, 30))
                ):
                    raise ValueError("Unexpected QA metadata: " + g["id"])
                src.read(1)
            os.replace(path.with_suffix(".part"), path)
            write(
                meta,
                {
                    "key": key,
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                    "retrieved_utc": now(),
                    "headers": headers,
                },
            )
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def mapbiomas(area, year, directory=None):
    cfg = config()
    directory = directory or DATA / "candidate_labels"
    ident = area.get("area_id", area["candidate_id"])
    path = directory / f"{ident}_{year}_classes.tif"
    meta = path.with_suffix(".json")
    grid = profile(area)
    url = cfg["labels"]["url_template"].format(year=year)
    key = {
        "url": url,
        "crs": area["crs"],
        "transform": area["transform"],
        "pixels": area["pixels"],
        "resampling": "nearest",
    }
    if good_cache(path, meta, key):
        with rasterio.open(path) as src:
            return src.read(1)
    bounds = array_bounds(grid["height"], grid["width"], grid["transform"])
    for attempt in range(3):
        try:
            with rasterio.Env(**HTTP_ENV):
                with rasterio.open(url) as src:
                    box = transform_bounds(grid["crs"], src.crs, *bounds, densify_pts=21)
                    crop = from_bounds(*box, src.transform)
                    c0 = max(0, math.floor(crop.col_off) - 2)
                    r0 = max(0, math.floor(crop.row_off) - 2)
                    c1 = min(src.width, math.ceil(crop.col_off + crop.width) + 2)
                    r1 = min(src.height, math.ceil(crop.row_off + crop.height) + 2)
                    window = Window(c0, r0, c1 - c0, r1 - r0)
                    raw = src.read(1, window=window)
                    out = np.full((grid["height"], grid["width"]), 255, dtype=np.uint8)
                    reproject(
                        raw,
                        out,
                        src_transform=src.window_transform(window),
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=grid["transform"],
                        dst_crs=grid["crs"],
                        dst_nodata=255,
                        resampling=Resampling.nearest,
                    )
                    source = {
                        "crs": str(src.crs),
                        "shape": list(src.shape),
                        "transform": list(src.transform)[:6],
                        "window": list(window.flatten()),
                        "nodata": src.nodata,
                        "dtype": src.dtypes[0],
                    }
            save_tif(path, out, grid, ["MapBiomas_class"], 255)
            write(meta, {"key": key, "sha256": sha(path), "retrieved_utc": now(), "source": source})
            return out
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def baseline_audit():
    cfg = config()
    rows = []
    for a in candidates()["areas"]:
        classes = mapbiomas(a, 2015)
        label = binary(classes, cfg)
        known = label != 255
        forest = float(np.count_nonzero(label == 1) / known.sum()) if known.any() else None
        reason = []
        if known.mean() < cfg["selection"]["min_known_label_fraction"]:
            reason.append("insufficient_known_MapBiomas_2015")
        lo, hi = cfg["selection"]["forest_fraction_range"]
        if forest is None or not lo <= forest <= hi:
            reason.append("forest_share_outside_declared_range")
        adjacent_known = known[:, 1:] & known[:, :-1]
        edge = float(
            np.count_nonzero((label[:, 1:] != label[:, :-1]) & adjacent_known)
            / max(1, adjacent_known.sum())
        )
        rows.append(
            {
                "candidate_id": a["candidate_id"],
                "region": a["region"],
                "known_fraction": float(known.mean()),
                "forest_fraction": forest,
                "horizontal_class_boundary_fraction": edge,
                "baseline_eligible": not reason,
                "reasons": reason,
            }
        )
        event("baseline", a["candidate_id"], forest_fraction=forest, eligible=not reason)
    write(
        REPORT / "candidate_baseline.json",
        {"created_utc": now(), "config_sha256": fingerprint(), "rows": rows},
    )
    return rows


def intersects(a, g):
    spatial = (
        g["umm"].get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {})
    )
    boxes = spatial.get("BoundingRectangles", [])
    if boxes:
        b = boxes[0]
        box = [
            b["WestBoundingCoordinate"],
            b["SouthBoundingCoordinate"],
            b["EastBoundingCoordinate"],
            b["NorthBoundingCoordinate"],
        ]
    else:
        points = [
            p
            for poly in spatial.get("GPolygons", [])
            for p in poly.get("Boundary", {}).get("Points", [])
        ]
        if not points:
            return True
        box = [
            min((p["Longitude"] for p in points)),
            min((p["Latitude"] for p in points)),
            max((p["Longitude"] for p in points)),
            max((p["Latitude"] for p in points)),
        ]
    b = a["bbox_wgs84"]
    pad = 0.02
    return not (
        box[2] < b[0] - pad or box[0] > b[2] + pad or box[3] < b[1] - pad or (box[1] > b[3] + pad)
    )


def qa_audit(workers):
    cfg = config()
    baselines = load(REPORT / "candidate_baseline.json")["rows"]
    eligible = {r["candidate_id"] for r in baselines if r["baseline_eligible"]}
    areas = [a for a in candidates()["areas"] if a["candidate_id"] in eligible]
    authenticate()
    reports = []
    for region in cfg["regions"]:
        for year in cfg["years"]:
            cat = catalog_one(region, year)
            relevant = [
                g
                for g in cat["granules"]
                if any((intersects(a, g) for a in areas if a["region"] == region))
            ]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(download_qa, g): g for g in relevant}
                for i, f in enumerate(as_completed(futures), 1):
                    f.result()
                    if i % 10 == 0 or i == len(futures):
                        event("qa_download", f"{region} {year}", done=i, total=len(futures))
            for a in [a for a in areas if a["region"] == region]:
                days = {}
                gs = []
                for g in relevant:
                    if not intersects(a, g):
                        continue
                    with rasterio.open(DATA / "source_qa" / f"{g['id']}.Fmask.tif") as src:
                        qa = src.read(
                            1,
                            window=aligned_window(src, profile(a)),
                            boundless=True,
                            fill_value=255,
                        )
                    valid = qa_mask(qa)
                    days[g["date"]] = days.get(g["date"], np.zeros(qa.shape, dtype=bool)) | valid
                    gs.append(
                        {"id": g["id"], "date": g["date"], "valid_fraction": float(valid.mean())}
                    )
                count = (
                    np.sum(np.stack(list(days.values())), axis=0, dtype=np.uint16)
                    if days
                    else np.zeros((a["pixels"], a["pixels"]), dtype=np.uint16)
                )
                coverage = float((count >= cfg["preprocessing"]["min_valid_days"]).mean())
                save_tif(
                    DATA / "candidate_quality" / f"{a['candidate_id']}_{year}_valid_days.tif",
                    count,
                    profile(a),
                    ["QA_valid_distinct_days"],
                    None,
                )
                reports.append(
                    {
                        "candidate_id": a["candidate_id"],
                        "year": year,
                        "qa_three_day_fraction": coverage,
                        "count_quantiles": np.quantile(count, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                        "granules": gs,
                    }
                )
                event("qa_coverage", f"{a['candidate_id']} {year}", fraction=coverage)
            write(
                REPORT / "candidate_quality.json",
                {
                    "created_utc": now(),
                    "config_sha256": fingerprint(),
                    "status": "QA_only_upper_bound_before_band_checks",
                    "rows": reports,
                },
            )


def select():
    cfg = config()
    base = load(REPORT / "candidate_baseline.json")["rows"]
    quality = load(REPORT / "candidate_quality.json")["rows"]
    areas = {a["candidate_id"]: a for a in candidates()["areas"]}
    eligible = []
    audit = []
    for b in base:
        reasons = list(b["reasons"])
        qs = [q for q in quality if q["candidate_id"] == b["candidate_id"]]
        if b["baseline_eligible"]:
            if {q["year"] for q in qs} != set(cfg["years"]):
                reasons.append("missing_QA_audit")
            elif any(
                (
                    q["qa_three_day_fraction"]
                    < cfg["selection"]["min_qa_three_day_fraction_each_year"]
                    for q in qs
                )
            ):
                reasons.append("insufficient_QA_three_day_support")
        row = {
            **b,
            "eligible": not reasons,
            "reasons": reasons,
            "qa_fraction_by_year": {str(q["year"]): q["qa_three_day_fraction"] for q in qs},
        }
        audit.append(row)
        if not reasons:
            eligible.append(row)
    write(
        REPORT / "candidate_selection_audit.json",
        {"config_sha256": fingerprint(), "created_utc": now(), "rows": audit},
    )
    selected = []
    for region_index, region in enumerate(cfg["regions"]):
        rows = sorted(
            [r for r in eligible if r["region"] == region],
            key=lambda r: (r["forest_fraction"], r["candidate_id"]),
        )
        if len(rows) < 6:
            raise ValueError(
                f"Only {len(rows)} eligible candidates in {region}; no silent threshold relaxation"
            )
        rng = np.random.default_rng(cfg["selection"]["seed"] + region_index)
        held = []
        for rank, group in enumerate(np.array_split(np.arange(len(rows)), 3), 1):
            chosen = rng.choice(group, 2, replace=False)
            for j, index in enumerate(chosen):
                a = {
                    **areas[rows[index]["candidate_id"]],
                    "baseline_forest_fraction": rows[index]["forest_fraction"],
                    "selection_rank_stratum": rank,
                }
                if j == 0:
                    a.update(area_id=f"{region}_T{rank:02d}", split="train", years=[2015, 2020])
                    selected.append(a)
                else:
                    held.append(a)
        held = [held[i] for i in rng.permutation(3)]
        for j, a in enumerate(held):
            a.update(
                area_id=f"{region}_V01" if j == 0 else f"{region}_E{j:02d}",
                split="validation" if j == 0 else "test",
                years=[2015, 2020] if j == 0 else [2015, 2020, 2025],
            )
            selected.append(a)
    distances = []
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            if a["region"] != b["region"]:
                continue
            x0, y0, x1, y1 = a["bounds_utm"]
            u0, v0, u1, v1 = b["bounds_utm"]
            d = math.hypot(max(x0 - u1, u0 - x1, 0), max(y0 - v1, v0 - y1, 0))
            if d < cfg["selection"]["min_boundary_distance_m"]:
                raise ValueError("AOI separation fails")
            distances.append({"a": a["area_id"], "b": b["area_id"], "boundary_distance_m": d})
    result = {
        "created_utc": now(),
        "config_sha256": fingerprint(),
        "status": "selected_before_training",
        "areas": sorted(selected, key=lambda a: a["area_id"]),
        "distances": distances,
    }
    path = ROOT / "configs/areas.json"
    if path.exists():
        existing = load(path)
        if (
            existing["areas"] != result["areas"]
            or existing["config_sha256"] != result["config_sha256"]
        ):
            raise ValueError("Refusing changed split")
        return existing
    write(path, result)
    write(
        ROOT / "data/areas.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": a["geometry"],
                    "properties": {k: v for k, v in a.items() if k != "geometry"},
                }
                for a in result["areas"]
            ],
        },
    )
    event("selection", "Selected 12 AOIs")
    return result


def read_band(g, band, area, context):
    grid = profile(area, context)
    path = DATA / "source_windows" / area["area_id"] / g["id"] / f"{band}.tif"
    meta = path.with_suffix(".json")
    key = {
        "url": g["assets"][band],
        "area": area["area_id"],
        "transform": list(grid["transform"])[:6],
        "shape": [grid["height"], grid["width"]],
        "resampling": "none",
        "config_sha256": fingerprint(),
    }
    if good_cache(path, meta, key):
        with rasterio.open(path) as src:
            return src.read(1)
    for attempt in range(3):
        try:
            with authenticate().get_session() as session:
                with session.get(
                    key["url"], headers={"Range": "bytes=0-15"}, stream=True, timeout=(15, 45)
                ) as r:
                    r.raise_for_status()
                    resolved = r.url
                    headers = {
                        k: r.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Range")
                    }
            with rasterio.Env(**HTTP_ENV):
                with rasterio.open(resolved) as src:
                    if src.count != 1 or src.dtypes != ("int16",) or src.nodata != -9999:
                        raise ValueError("Unexpected reflectance metadata: " + g["id"])
                    if (
                        not np.isclose(src.scales[0], 0.0001, atol=1e-10, rtol=0)
                        or src.offsets[0] != 0
                    ):
                        raise ValueError("Unexpected reflectance scale/offset")
                    win = aligned_window(src, grid)
                    arr = src.read(1, window=win, boundless=True, fill_value=-9999)
                    source = {
                        "crs": src.crs.to_wkt(),
                        "transform": list(src.transform)[:6],
                        "shape": list(src.shape),
                        "window": list(win.flatten()),
                        "scale": src.scales[0],
                        "offset": src.offsets[0],
                    }
            save_tif(path, arr, grid, [band], -9999)
            write(
                meta,
                {
                    "key": key,
                    "sha256": sha(path),
                    "retrieved_utc": now(),
                    "source": source,
                    "headers": headers,
                },
            )
            return arr
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def composite(area, year, workers):
    cfg = config()
    ident = f"{area['area_id']}_{year}"
    directory = DATA / "prepared" / area["split"] / ident
    manifest = directory / "manifest.json"
    if manifest.exists():
        m = load(manifest)
        if m["config_sha256"] != fingerprint():
            raise ValueError("Composite rules changed")
        for f in m["files"]:
            if sha(ROOT / f["path"]) != f["sha256"]:
                raise ValueError("Prepared file changed")
        event("composite_cached", ident)
        return m
    cat = catalog_one(area["region"], year)
    gs = [g for g in cat["granules"] if intersects(area, g)]
    ctx = cfg["preprocessing"]["context_pixels"]
    grid = profile(area, ctx)
    shape = (grid["height"], grid["width"])
    daily = {}
    provenance = []
    for j, g in enumerate(gs, 1):
        qa_path = download_qa(g)
        with rasterio.open(qa_path) as src:
            qa = src.read(1, window=aligned_window(src, grid), boundless=True, fill_value=255)
        valid = qa_mask(qa)
        entry = {
            "granule_id": g["id"],
            "date": g["date"],
            "qa_source": evidence(qa_path),
            "qa_valid_pixels_with_context": int(valid.sum()),
            "bands_downloaded": False,
        }
        needed = valid.copy()
        if g["date"] in daily:
            needed &= ~np.isfinite(daily[g["date"]][0])
        if valid.any() and (not needed.any()):
            entry["skip_reason"] = "same_day_already_complete"
        if needed.any():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                raw = np.stack(list(pool.map(lambda band: read_band(g, band, area, ctx), BANDS)))
            day = daily.setdefault(g["date"], np.full((6, *shape), np.nan, dtype=np.float32))
            valid, take = merge_day(day, raw, qa)
            entry.update(
                bands_downloaded=True,
                complete_valid_pixels_with_context=int(valid.sum()),
                source_windows=[
                    evidence(DATA / "source_windows" / area["area_id"] / g["id"] / f"{b}.tif")
                    for b in BANDS
                ],
            )
            entry["pixels_contributed_after_day_deduplication"] = int(take.sum())
        provenance.append(entry)
        event(
            "composite_sources",
            ident,
            done=j,
            total=len(gs),
            granule=g["id"],
            qa_valid_pixels=int(qa_mask(qa).sum()),
        )
    if not daily:
        raise ValueError("No valid daily vectors")
    days = sorted(daily)
    features, valid, count, first, last = composite_daily(
        daily,
        cfg["preprocessing"]["min_valid_days"],
        cfg["preprocessing"]["index_denominator_epsilon"],
    )
    core = (slice(ctx, ctx + area["pixels"]), slice(ctx, ctx + area["pixels"]))
    classes = mapbiomas(area, year, directory=DATA / "labels" / area["split"])
    labels = binary(classes, cfg)
    corevalid = valid[core]
    files = []
    outputs = [
        ("features_context.tif", features, grid, cfg["preprocessing"]["feature_order"], np.nan),
        ("valid_context.tif", valid.astype(np.uint8), grid, ["valid_features"], None),
        ("valid_days_context.tif", count, grid, ["complete_valid_distinct_days"], None),
        ("first_doy_context.tif", first, grid, ["first_valid_day_of_year"], 65535),
        ("last_doy_context.tif", last, grid, ["last_valid_day_of_year"], 65535),
        ("labels_mapbiomas.tif", labels, profile(area), ["forest_1_nonforest_0_unknown_255"], 255),
    ]
    for name, arr, p, names, nodata in outputs:
        path = directory / name
        save_tif(path, arr, p, names, nodata)
        files.append(evidence(path))
    m = {
        "created_utc": now(),
        "config_sha256": fingerprint(),
        "catalog_sha256": sha(DATA / "catalog" / f"{area['region']}_{year}.json"),
        "area_id": area["area_id"],
        "candidate_id": area["candidate_id"],
        "split": area["split"],
        "year": year,
        "context_pixels": ctx,
        "core_pixels": area["pixels"],
        "dates": days,
        "valid_fraction_core": float(corevalid.mean()),
        "valid_days_quantiles_core": np.quantile(count[core], [0, 0.01, 0.5, 0.99, 1]).tolist(),
        "eligible_label_counts": {
            str(c): int(np.count_nonzero(corevalid & (labels == c))) for c in (0, 1)
        },
        "files": files,
        "sources": provenance,
        "reference": "MapBiomas teacher labels, not independent ground truth",
    }
    write(manifest, m)
    event(
        "composite_complete",
        ident,
        valid_fraction=m["valid_fraction_core"],
        labels=m["eligible_label_counts"],
    )
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["candidates", "catalog", "baseline", "qa", "select", "composites", "all"]
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--area")
    p.add_argument("--year", type=int)
    args = p.parse_args()
    cfg = config()
    candidates()
    if args.stage == "candidates":
        return
    if args.stage in ("catalog", "all"):
        for region in cfg["regions"]:
            for year in cfg["years"]:
                catalog_one(region, year)
    if args.stage in ("baseline", "all"):
        baseline_audit()
    if args.stage in ("qa", "all"):
        qa_audit(args.workers)
    if args.stage in ("select", "all"):
        select()
    if args.stage in ("composites", "all"):
        authenticate()
        for area in load(ROOT / "configs/areas.json")["areas"]:
            if args.area and args.area != area["area_id"]:
                continue
            for year in area["years"]:
                if args.year and args.year != year:
                    continue
                composite(area, year, args.workers)


if __name__ == "__main__":
    logging.getLogger("earthaccess").setLevel(logging.ERROR)
    try:
        main()
    except Exception as exc:
        event("failed", str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        raise SystemExit(1)
