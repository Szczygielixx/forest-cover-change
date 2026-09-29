import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from training.models import UNet, sparse_loss
from training.support import (
    prediction_timed,
    reset_seed,
    restore_rng,
    rng_state,
    torch_save,
    utc,
    validation_metrics,
    write,
    report,
)


def train_unet(
    directory,
    dataset,
    validation,
    mean,
    std,
    config,
    seed,
    lr,
    device="cuda",
    resume=None,
    stop_after_epoch=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reset_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0
    )
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler_config = config.get("scheduler")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        if scheduler_config
        else None
    )
    history, best_score, best_epoch, bad_checks = ([], None, None, 0)
    best_state, epoch_start, fit_seconds = (None, 1, 0.0)
    signature = {
        "seed": seed,
        "learning_rate": lr,
        "config": config,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "labels_per_epoch": dataset.label_count,
    }
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        assert checkpoint["signature"] == signature, "Resume configuration differs"
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"], generator)
        history = checkpoint["history"]
        best_score, best_epoch = (checkpoint["best_score"], checkpoint["best_epoch"])
        best_state, bad_checks = (checkpoint["best_model"], checkpoint["bad_checks"])
        epoch_start, fit_seconds = (checkpoint["epoch"] + 1, checkpoint["fit_seconds"])
    started = utc()
    stop_reason = "max_epochs"
    for epoch in range(epoch_start, config["max_epochs"] + 1):
        model.train()
        if device == "cuda":
            torch.cuda.synchronize()
        started_epoch = time.perf_counter()
        total_loss, labels_seen = (0.0, 0)
        for xx, yy in loader:
            xx, yy = (xx.to(device), yy.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = sparse_loss(model(xx), yy)
            assert torch.isfinite(loss), "Non-finite training loss"
            loss.backward()
            optimizer.step()
            count = int((yy != 255).sum().item())
            total_loss += loss.item() * count
            labels_seen += count
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_epoch
        fit_seconds += elapsed
        assert labels_seen == dataset.label_count
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mean_label_weighted_loss": total_loss / labels_seen,
            "labels_seen": labels_seen,
            "updates": len(loader),
            "fit_seconds": elapsed,
        }
        if epoch % config["validate_every_epochs"] == 0 or epoch == config["max_epochs"]:
            outputs, pred_seconds = prediction_timed("unet", model, validation, mean, std, device)
            metrics = validation_metrics(validation, outputs)
            score = metrics["selection_score"]
            write(directory / f"validation/epoch_{epoch:03}.json", metrics)
            record.update(selection_score=score, validation_seconds=pred_seconds)
            if best_score is None or score > best_score + config["tie_tolerance"]:
                best_score, best_epoch, bad_checks = (score, epoch, 0)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_checks += 1
            if scheduler:
                scheduler.step(score)
        record["next_learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(generator),
            "epoch": epoch,
            "signature": signature,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_model": best_state,
            "bad_checks": bad_checks,
            "fit_seconds": fit_seconds,
        }
        torch_save(directory / "last.pt", checkpoint)
        if best_epoch == epoch:
            torch_save(directory / "best.pt", checkpoint)
        write(directory / "history.json", history)
        report(
            "unet_training",
            epoch,
            config["max_epochs"],
            trial=directory.name,
            seed=seed,
            lr=lr,
            loss=record["mean_label_weighted_loss"],
            best_epoch=best_epoch,
            best_score=best_score,
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {"interrupted_for_test": True, "checkpoint": str(directory / "last.pt")}
        if bad_checks >= config["early_stopping_validation_checks"]:
            stop_reason = "early_stopping"
            break
    assert best_state is not None, "No validated checkpoint"
    model.load_state_dict(best_state)
    model.eval()
    torch_save(
        directory / "weights.pt",
        {"model": best_state, "signature": signature, "best_epoch": best_epoch},
    )
    restored = UNet().to(device)
    restored.load_state_dict(
        torch.load(directory / "weights.pt", map_location=device, weights_only=False)["model"]
    )
    restored.eval()
    probe = torch.zeros((1, 9, 128, 128), device=device)
    with torch.no_grad():
        torch.testing.assert_close(model(probe), restored(probe), rtol=0, atol=0)
    result = {
        "status": "succeeded",
        "seed": seed,
        "params": {"learning_rate": lr},
        "started_utc": started,
        "finished_utc": utc(),
        "epochs_completed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "fit_seconds": fit_seconds,
        "stop_reason": stop_reason,
        "samples": dataset.label_count,
        "patches_per_epoch": len(dataset),
        "reload_exact": True,
        "model_file": "weights.pt",
        "checkpoint_file": "last.pt",
    }
    write(directory / "run.json", result)
    return result
