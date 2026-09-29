import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
from training.support import read, load_images, validation_metrics
from evaluation.run import csv_write


def replay_baseline_stopping(history, tolerance):
    best, bad, stop = (None, 0, None)
    for point in history:
        if point["epoch"] > 50:
            break
        if "selection_score" not in point:
            continue
        score = point["selection_score"]
        if best is None or score > best + tolerance:
            best, bad = (score, 0)
        else:
            bad += 1
        stop = point["epoch"]
        if bad >= 5:
            break
    return dict(epoch=stop, best_loss_f1=best)


def compact(metrics):
    result = {}
    for metric in ["precision", "recall", "f1"]:
        result["forest_" + metric] = statistics.mean((x[metric] for x in metrics["state"].values()))
        for target in ["loss", "gain"]:
            result[target + "_" + metric] = statistics.mean(
                (x[target][metric] for x in metrics["change"].values())
            )
    return result


def main():
    p = argparse.ArgumentParser(
        description="Summarize additional experiments using validation only."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    images = load_images(args.data, "validation")
    rows, stopping = ([], [])
    for root in args.runs:
        complete = read(root / "completion.json")
        assert complete["status"] == "succeeded" and (not complete["test_evaluated"])
        for spec in read(root / "experiment.json")["fits"]:
            variant = spec["variant"]
            directory = root / "fits" / variant
            record = read(directory / "complete.json")
            predictions = {}
            for im in images:
                with np.load(
                    root / "predictions/validation" / variant / "state" / f"{im['key']}.npz",
                    allow_pickle=False,
                ) as z:
                    values = {k: z[k] for k in ["prediction", "score", "valid"]}
                np.testing.assert_array_equal(values["valid"], im["core_valid"])
                predictions[im["key"]] = values
            metrics = validation_metrics(images, predictions)
            assert metrics == read(directory / "validation.json")
            row = {k: spec[k] for k in ["variant", "method", "budget", "regime", "seed"]}
            rows.append({**row, **compact(metrics)})
            if spec["method"] == "unet":
                history = read(directory / "history.json")
                stopping.append(
                    {
                        **row,
                        "best_epoch": record["best_epoch"],
                        "epochs_completed": record["epochs_completed"],
                        "best_loss_f1": record["selection_score"],
                        "last_loss_f1": history[-1]["selection_score"],
                        "replayed_stop_epoch": replay_baseline_stopping(history, 1e-06)["epoch"],
                        "replayed_best_loss_f1": replay_baseline_stopping(history, 1e-06)[
                            "best_loss_f1"
                        ],
                    }
                )
    assert len({r["variant"] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["budget"], row["regime"]].append(row)
    summary = []
    for (method, budget, regime), group in sorted(groups.items()):
        row = dict(method=method, budget=budget, regime=regime, runs=len(group))
        for metric in [
            "forest_precision",
            "forest_recall",
            "forest_f1",
            "loss_precision",
            "loss_recall",
            "loss_f1",
            "gain_precision",
            "gain_recall",
            "gain_f1",
        ]:
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(row)
    csv_write(args.output / "individual_results.csv", rows)
    csv_write(args.output / "method_comparison.csv", summary)
    csv_write(args.output / "unet_stopping.csv", stopping)
    print(f"Checked {len(rows)} validation runs.")


if __name__ == "__main__":
    main()
