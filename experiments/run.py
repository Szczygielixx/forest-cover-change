import argparse

"One rented GPU, fixed experimental grid; no access to the final test set."
from pathlib import Path
import gc
import time
import numpy as np
import torch
from training.run import tabular_fit, model_load
from training.support import (
    read,
    write,
    utc,
    common_samples,
    load_images,
    SparseDataset,
    prediction_timed,
    validation_metrics,
    save_predictions,
    report,
)
from .unet import train_unet


def budget_samples(samples, budget):
    mask = samples["rank_in_image"] < budget // 12
    chosen = {k: v[mask] for k, v in samples.items()}
    assert len(chosen["y"]) == budget
    assert np.bincount(chosen["y"]).tolist() == [budget // 2] * 2
    return chosen


def supervised_images(images, chosen):
    result = []
    for i, original in enumerate(images):
        im = dict(original)
        labels = np.full_like(im["labels"], 255)
        take = chosen["image_index"] == i
        r, c, y = (chosen["row"][take], chosen["col"][take], chosen["y"][take])
        assert len(np.unique(r * labels.shape[1] + c)) == len(y)
        assert np.all(im["core_valid"][r, c])
        labels[r, c] = y
        o = int(im["core_offset"])
        np.testing.assert_array_equal(im["features"][:, r + o, c + o].T, chosen["X"][take])
        im["labels"] = labels
        result.append(im)
    return result


def main():
    p = argparse.ArgumentParser(description="Additional labels and U-Net epochs; validation only.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/controls.json"
    )
    p.add_argument("--method", choices=["rf", "svm", "unet", "all"], default="all")
    args = p.parse_args()
    cfg = read(args.config)
    if args.method != "all":
        cfg["fits"] = [f for f in cfg["fits"] if f["method"] == args.method]
    cfg["expected_fits"] = len(cfg["fits"])
    assert cfg["scope"] == "exploratory_validation_only"
    if any((f["method"] != "rf" for f in cfg["fits"])):
        assert torch.cuda.is_available(), "SVM and U-Net require CUDA for this experiment"
    out, bundle = (args.output, args.data)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    write(out / "experiment.json", cfg)
    with np.load(args.samples, allow_pickle=False) as z:
        samples = {k: z[k] for k in z.files}
    old_x, old_y, mean, std = common_samples(bundle)
    chosen = budget_samples(samples, 60000)
    np.testing.assert_array_equal(chosen["X"], old_x)
    np.testing.assert_array_equal(chosen["y"], old_y)
    np.testing.assert_array_equal(mean, np.array(cfg["normalization"]["mean"], np.float32))
    np.testing.assert_array_equal(std, np.array(cfg["normalization"]["std"], np.float32))
    images = load_images(bundle, "train")
    validation = load_images(bundle, "validation")
    assert set((str(im["area_id"]) for im in images)).isdisjoint(
        (str(im["area_id"]) for im in validation)
    )
    write(
        out / "data_verification.json",
        dict(
            base_prefix_exact=True,
            normalization_fixed=True,
            validation_areas=cfg["validation_areas"],
            test_reference_used=False,
        ),
    )
    results = []
    started = utc()
    for number, item in enumerate(cfg["fits"], 1):
        variant = item["variant"]
        directory = out / "fits" / variant
        report(
            "experiment_fit", number - 1, len(cfg["fits"]), variant=variant, budget=item["budget"]
        )
        if (directory / "complete.json").exists():
            results.append(read(directory / "complete.json"))
            continue
        chosen = budget_samples(samples, item["budget"])
        wall_start = time.perf_counter()
        if item["method"] == "unet":
            ims = supervised_images(images, chosen)
            dataset = SparseDataset(ims, mean, std, expected_count=item["budget"])
            checkpoint = directory / "last.pt"
            result = train_unet(
                directory,
                dataset,
                validation,
                mean,
                std,
                item["unet_config"],
                item["seed"],
                item["params"]["learning_rate"],
                resume=checkpoint if checkpoint.exists() else None,
            )
            del dataset, ims
        else:
            result = tabular_fit(
                item["method"],
                item["params"],
                item["seed"],
                chosen["X"],
                chosen["y"],
                mean,
                std,
                validation,
                directory,
            )
        model = model_load(item["method"], directory)
        predictions, elapsed = prediction_timed(item["method"], model, validation, mean, std)
        metrics = validation_metrics(validation, predictions)
        assert abs(result["selection_score"] - metrics["selection_score"]) < 1e-10
        save_predictions(out / "predictions/validation" / variant, validation, predictions, variant)
        write(directory / "validation.json", metrics)
        record = {
            **item,
            **result,
            "wall_seconds": time.perf_counter() - wall_start,
            "prediction_seconds": elapsed,
        }
        write(directory / "complete.json", record)
        results.append(record)
        write(out / "results.json", results)
        print(
            f"COMPLETED {number}/{len(cfg['fits'])} {variant}: validation_loss_F1={result['selection_score']:.6f}",
            flush=True,
        )
        del model, predictions, chosen
        gc.collect()
        torch.cuda.empty_cache()
    assert len(results) == cfg["expected_fits"]
    completion = dict(
        status="succeeded",
        campaign_id=cfg["campaign_id"],
        fits=len(results),
        scope=cfg["scope"],
        test_evaluated=False,
        started_utc=started,
        finished_utc=utc(),
    )
    write(out / "completion.json", completion)
    report("controls_complete", len(results), len(results))
    print(completion, flush=True)


if __name__ == "__main__":
    main()
