import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from .support import (
    read,
    write,
    utc,
    report,
    synchronize,
    prediction_timed,
    validation_metrics,
    predict,
    select_first,
    SparseDataset,
    load_test,
    save_predictions,
)
from .data import common_samples, load_images
from .models import UNet
from .unet import train_unet


def model_load(method, directory, device="cuda"):
    directory = Path(directory)
    if method == "ndvi":
        return read(directory / "threshold.json")
    if method in ("rf", "svm"):
        return joblib.load(directory / "model.joblib")
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(directory / "weights.pt", map_location=device, weights_only=False)["model"]
    )
    return model.eval()


def tabular_fit(method, params, seed, x, y, mean, std, validation, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = utc()
    if method == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=8, random_state=seed, **params
        )
        xx = x
    else:
        from cuml.svm import SVC

        model = SVC(
            kernel="rbf",
            tol=0.001,
            cache_size=4096,
            probability=False,
            output_type="numpy",
            **params,
        )
        xx = np.ascontiguousarray((x - mean) / std, dtype=np.float32)
    report(method + "_fit", trial=directory.name, samples=len(y), params=params, seed=seed)
    synchronize(method)
    begin = time.perf_counter()
    model.fit(xx, y)
    synchronize(method)
    fit = time.perf_counter() - begin
    extra = {}
    if method == "svm":
        extra = {
            "fit_status": int(model.fit_status_),
            "support_vectors": int(np.asarray(model.n_support_).sum()),
            "n_iter": np.asarray(model.n_iter_).tolist() if hasattr(model, "n_iter_") else None,
        }
        assert extra["fit_status"] == 0, "SVM did not converge"
    target = directory / "model.joblib"
    temporary = directory / "model.tmp"
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, target)
    reloaded = joblib.load(target)
    np.testing.assert_array_equal(
        np.asarray(model.predict(xx[:1024])), np.asarray(reloaded.predict(xx[:1024]))
    )
    outputs, pred_time = prediction_timed(method, reloaded, validation, mean, std)
    metrics = validation_metrics(validation, outputs)
    write(directory / "validation.json", metrics)
    result = {
        "status": "succeeded",
        "params": params,
        "seed": seed,
        "samples": len(y),
        "started_utc": started,
        "finished_utc": utc(),
        "fit_seconds": fit,
        "validation_seconds": pred_time,
        "selection_score": metrics["selection_score"],
        "model_file": "model.joblib",
        "reload_exact": True,
        **extra,
    }
    write(directory / "run.json", result)
    del model, reloaded
    return result


def ndvi_fit(validation, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    candidates = []
    start = int(round(config["start"] * 100))
    stop = int(round(config["stop"] * 100))
    assert config["step"] == 0.01
    for value in range(start, stop + 1):
        params = {"threshold": value / 100}
        predictions = predict("ndvi", params, validation, None, None)
        metrics = validation_metrics(validation, predictions)
        candidates.append(
            {**params, "selection_score": metrics["selection_score"], "metrics": metrics}
        )
    winner = select_first(candidates)
    write(directory / "candidates.json", candidates)
    write(
        directory / "threshold.json",
        {"threshold": winner["threshold"], "selection_score": winner["selection_score"]},
    )
    result = {
        "status": "succeeded",
        "params": {"threshold": winner["threshold"]},
        "seed": None,
        "selection_score": winner["selection_score"],
        "fit_seconds": time.perf_counter() - begin,
        "model_file": "threshold.json",
        "candidates": len(candidates),
    }
    write(directory / "validation.json", winner["metrics"])
    write(directory / "run.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description="Tune, train, freeze and predict one method family.")
    p.add_argument("--family", choices=["rf_ndvi", "unet", "svm"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/training.json"
    )
    p.add_argument("--resume-dir", type=Path)
    args = p.parse_args()
    bundle, out, family = (args.data, args.output, args.family)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory; use --resume-dir for a previous run")
    out.mkdir(parents=True, exist_ok=True)
    cfg = read(args.config)
    assert cfg["training_label_budget"] == 60000
    if family in ("svm", "unet"):
        assert torch.cuda.is_available(), "SVM and U-Net require CUDA for this experiment"
    torch.set_num_threads(8)
    write(out / "configuration.json", cfg)
    x, y, mean, std = common_samples(bundle)
    write(
        out / "normalization.json",
        {"mean": mean.tolist(), "std": std.tolist(), "fit_samples": len(y), "fit_split": "train"},
    )
    validation = load_images(bundle, "validation")
    train_index = read(bundle / "train/index.json")
    val_index = read(bundle / "validation/index.json")
    assert set((v["area_id"] for v in train_index["images"])).isdisjoint(
        (v["area_id"] for v in val_index["images"])
    )
    assert len(train_index["images"]) == 12 and len(validation) == 4
    final = []
    tuning = []
    selection = {"status": "selection_in_progress", "family": family, "started_utc": utc()}
    if family == "rf_ndvi":
        d = out / "final/ndvi"
        r = ndvi_fit(validation, cfg["methods"]["ndvi_threshold"], d)
        final.append(
            {"variant": "ndvi", "method": "ndvi", "directory": str(d.relative_to(out)), **r}
        )
        configurations = [
            {"max_depth": a, "min_samples_leaf": b}
            for a, b in itertools.product([20, None], [1, 5])
        ]
        method = "rf"
    elif family == "svm":
        configurations = [
            {"C": a, "gamma": b} for a, b in itertools.product([1, 10, 100], [0.01, 0.1])
        ]
        method = "svm"
    else:
        configurations = [{"learning_rate": v} for v in cfg["methods"]["unet"]["learning_rate"]]
        method = "unet"
        dataset = SparseDataset(load_images(bundle, "train"), mean, std)
        unet_config = copy.deepcopy(cfg["methods"]["unet"])
        unet_config["tie_tolerance"] = cfg["selection"]["tie_tolerance"]

    def fit(directory, params, seed):
        if args.resume_dir:
            prior = args.resume_dir / directory.relative_to(out)
            if (prior / "run.json").exists() and read(prior / "run.json")["status"] == "succeeded":
                shutil.copytree(prior, directory, dirs_exist_ok=True)
                return read(directory / "run.json")
            if prior.exists():
                shutil.copytree(prior, directory, dirs_exist_ok=True)
        if method == "unet":
            resume = (
                args.resume_dir / directory.relative_to(out) / "last.pt"
                if args.resume_dir
                else None
            )
            if resume and (not resume.exists()):
                resume = None
            return train_unet(
                directory,
                dataset,
                validation,
                mean,
                std,
                unet_config,
                seed,
                params["learning_rate"],
                resume=resume,
            )
        return tabular_fit(method, params, seed, x, y, mean, std, validation, directory)

    for i, params in enumerate(configurations):
        trial = f"{method}_candidate_{i:02}"
        result = fit(out / "tuning" / trial, params, 42)
        tuning.append({"trial": trial, **result})
        write(out / "tuning_summary.json", tuning)
    winner = select_first(tuning, cfg["selection"]["tie_tolerance"])
    selection.update(
        chosen_trial=winner["trial"],
        chosen_params=winner["params"],
        selection_score=winner["selection_score"],
    )
    for seed in [42] if method == "svm" else [42, 123, 2026]:
        variant = f"{method}_seed{seed}" if method != "svm" else "svm"
        directory = out / "final" / variant
        result = fit(directory, winner["params"], seed)
        final.append(
            {
                "variant": variant,
                "method": method,
                "directory": str(directory.relative_to(out)),
                **result,
            }
        )
    selection.update(status="selection_frozen_before_test", frozen_utc=utc(), final_models=final)
    write(out / "selection.json", selection)
    test = load_test(bundle, out / "selection.json")
    for item in final:
        variant, method = (item["variant"], item["method"])
        report("final_prediction", variant=variant, method=method)
        model = model_load(method, out / item["directory"])
        val_outputs, val_time = prediction_timed(method, model, validation, mean, std)
        save_predictions(out / "predictions/validation" / variant, validation, val_outputs, variant)
        write(
            out / item["directory"] / "validation.json", validation_metrics(validation, val_outputs)
        )
        outputs, elapsed = prediction_timed(method, model, test, mean, std)
        save_predictions(out / "predictions/test" / variant, test, outputs, variant)
        item.update(test_prediction_seconds=elapsed, validation_prediction_seconds=val_time)
        write(out / "final_summary.json", final)
        del model, outputs, val_outputs
        if method == "unet":
            torch.cuda.empty_cache()
    summary = {
        "status": "succeeded",
        "family": family,
        "dataset_id": cfg["dataset_id"],
        "campaign_id": cfg["campaign_id"],
        "training_samples": 60000,
        "tuning_fits": len(tuning),
        "final_variants": len(final),
        "test_state_maps": len(final) * 12,
        "test_change_maps": len(final) * 8,
        "test_reference_used_for_fit_or_selection": False,
        "started_utc": selection["started_utc"],
        "finished_utc": utc(),
    }
    write(out / "completion.json", summary)
    report("family_complete", 1, 1, family=family)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
