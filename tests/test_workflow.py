import numpy as np
import pytest
import torch

from preprocessing.common import binary, composite_daily, config, indices, merge_day, qa_mask
from preprocessing.package import stable_seed, supervised_sample
from experiments.prepare import coordinates
from evaluation.run import context_labels
from training.models import sparse_loss
from training.run import model_load, ndvi_fit, tabular_fit
from training.support import SparseDataset, predict, read, transition, transition_metrics
from training.unet import train_unet


def test_quality_and_composite():
    flags = np.array([[0, 32, 2, 4, 8, 16, 192, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(
        qa_mask(flags), [[True, True, False, False, False, False, False, False]]
    )
    raw = np.full((6, 2, 2), 1000, dtype=np.int16)
    raw[3] = 3000
    raw[4, 1, 0] = -9999
    raw[4, 1, 1] = 12000
    day = np.full(raw.shape, np.nan, dtype=np.float32)
    qa = np.zeros((2, 2), dtype=np.uint8)
    valid, take = merge_day(day, raw, qa)
    np.testing.assert_array_equal(take, [[True, True], [False, False]])
    first = day.copy()
    merge_day(day, np.full(raw.shape, 4000, dtype=np.int16), qa)
    np.testing.assert_array_equal(day[:, 0], first[:, 0])
    daily = {f"2020-06-0{i}": day.copy() * i for i in [1, 2, 3]}
    daily["2020-06-03"][:, 1, 1] = np.nan
    features, valid, count, first_doy, last_doy = composite_daily(daily)
    np.testing.assert_allclose(features[:6, 0, 0], day[:, 0, 0] * 2)
    np.testing.assert_allclose(features[6:, 0, 0], indices(features[:6])[:, 0, 0])
    assert count[1, 1] == 2 and not valid[1, 1]
    assert np.isnan(features[:, 1, 1]).all()
    assert first_doy[0, 0] == 153 and last_doy[0, 0] == 155


def test_labels_and_nested_locations():
    codes = np.array([3, 5, 6, 49, 9, 15, 0, 27, 255, 254], dtype=np.uint8)
    np.testing.assert_array_equal(binary(codes, config()), [1, 1, 1, 1, 0, 0, 255, 255, 255, 255])
    labels = np.indices((20, 20)).sum(axis=0).astype(np.uint8) % 2
    valid = np.ones(labels.shape, dtype=bool)
    valid[0] = False
    seed = stable_seed("42:example_2015")
    small = supervised_sample(labels, valid, 10, seed)
    large = coordinates(labels, valid, 80, "example_2015")
    np.testing.assert_array_equal(small, large[:20])
    assert len(np.unique(large[:, 0] * 20 + large[:, 1])) == 80
    np.testing.assert_array_equal(np.bincount(labels[tuple(large.T)]), [40, 40])


def test_changes_and_context():
    before = np.array([0, 0, 1, 1, 255], dtype=np.uint8)
    after = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    truth = transition(before, after)
    np.testing.assert_array_equal(truth, [0, 1, 2, 3, 255])
    pred = np.array([2, 1, 2, 3, 0], dtype=np.uint8)
    metrics = transition_metrics(truth, pred, np.ones(5, bool))
    assert metrics["n"] == 4
    assert metrics["loss"]["precision"] == 0.5
    assert metrics["loss"]["recall"] == 1
    assert metrics["loss"]["f1"] == pytest.approx(2 / 3)
    labels = np.zeros((9, 9), np.uint8)
    labels[:, 4:] = 1
    contexts = context_labels(labels)
    np.testing.assert_array_equal(contexts[4], [0, 1, 1, 2, 2, 1, 1, 1, 0])


def images():
    result = []
    n, offset = 16, 32
    for area in ["first", "second"]:
        for year in [2015, 2020]:
            labels = np.zeros((n, n), np.uint8)
            labels[:, : 12 if year == 2015 else 6] = 1
            feature = np.pad(labels.astype(np.float32) * 0.6 + 0.1, offset, mode="edge")
            features = np.repeat(feature[None], 9, axis=0)
            result.append(
                dict(
                    area_id=area,
                    year=year,
                    key=f"{area}_{year}",
                    labels=labels,
                    core_size=n,
                    core_offset=offset,
                    features=features,
                    valid=np.ones(feature.shape, bool),
                    core_valid=np.ones(labels.shape, bool),
                )
            )
    return result


def test_ndvi_and_rf_fit_reload_and_predict(tmp_path):
    validation = images()
    ndvi = ndvi_fit(validation, dict(start=-0.2, stop=0.9, step=0.01), tmp_path / "ndvi")
    assert ndvi["selection_score"] == 1.0
    y = np.array([0, 1] * 64, dtype=np.uint8)
    x = np.repeat((y.astype(np.float32) * 0.6 + 0.1)[:, None], 9, axis=1)
    result = tabular_fit(
        "rf",
        dict(max_depth=20, min_samples_leaf=5),
        42,
        x,
        y,
        np.zeros(9),
        np.ones(9),
        validation,
        tmp_path / "rf",
    )
    assert result["reload_exact"] and result["selection_score"] == 1.0
    outputs = predict("rf", model_load("rf", tmp_path / "rf"), validation, None, None)
    for im in validation:
        np.testing.assert_array_equal(outputs[im["key"]]["prediction"], im["labels"])


def test_unet_sparse_loss_training_and_resume(tmp_path):
    torch.set_num_threads(2)
    logits = torch.zeros((1, 2, 2), requires_grad=True)
    labels = torch.tensor([[[0, 1], [255, 255]]], dtype=torch.uint8)
    sparse_loss(logits, labels).backward()
    assert logits.grad[0, 1].abs().sum() == 0
    validation = images()
    training = [dict(validation[0])]
    training[0]["labels"] = training[0]["labels"].copy()
    training[0]["labels"][1:] = 255
    mean, std = np.zeros(9, np.float32), np.ones(9, np.float32)
    dataset = SparseDataset(training, mean, std, expected_count=16)
    cfg = dict(
        batch_size=1,
        weight_decay=0.0001,
        max_epochs=2,
        validate_every_epochs=1,
        early_stopping_validation_checks=5,
        tie_tolerance=1e-6,
    )
    full = train_unet(
        tmp_path / "full", dataset, validation, mean, std, cfg, 42, 0.001, device="cpu"
    )
    train_unet(
        tmp_path / "resumed",
        dataset,
        validation,
        mean,
        std,
        cfg,
        42,
        0.001,
        device="cpu",
        stop_after_epoch=1,
    )
    resumed = train_unet(
        tmp_path / "resumed",
        dataset,
        validation,
        mean,
        std,
        cfg,
        42,
        0.001,
        device="cpu",
        resume=tmp_path / "resumed/last.pt",
    )
    assert full["selection_score"] == resumed["selection_score"]
    assert full["best_epoch"] == resumed["best_epoch"]
    a = model_load("unet", tmp_path / "full", device="cpu")
    b = model_load("unet", tmp_path / "resumed", device="cpu")
    for name, param in a.state_dict().items():
        torch.testing.assert_close(param, b.state_dict()[name], rtol=0, atol=0)
    assert all(h["labels_seen"] == 16 for h in read(tmp_path / "full/history.json"))
