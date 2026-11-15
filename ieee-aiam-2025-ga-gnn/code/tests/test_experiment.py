import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from aiam_gnn.cli import load_inputs, main, partition, result_directory
from aiam_gnn.graph_model import feature_graph


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(80, 4))
    x[0, 1] = np.nan
    frame = pd.DataFrame(x, columns=["a1", "a2", "b1", "b2"])
    frame["label"] = np.where(x[:, 0] + x[:, 2] > 0, "yes", "no")
    table = tmp_path / "input.csv"
    frame.to_csv(table, index=False)
    views = tmp_path / "views.json"
    views.write_text(json.dumps({"A": ["a1", "a2"], "B": ["b1", "b2"]}))
    return Namespace(input=table, modalities=views, label="label", positive="yes")


@pytest.mark.parametrize("encoder", ["gcn", "gat"])
def test_end_to_end(tmp_path, inputs, encoder):
    out = tmp_path / "result"
    main(
        [
            "--input",
            str(inputs.input),
            "--label",
            inputs.label,
            "--positive",
            inputs.positive,
            "--modalities",
            str(inputs.modalities),
            "--output",
            str(out),
            "--encoder",
            encoder,
            "--population",
            "3",
            "--generations",
            "4",
            "--surrogate-epochs",
            "2",
            "--epochs",
            "3",
            "--rescore",
            "2",
            "--explain",
        ]
    )
    run = json.loads((out / "run.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    rows = pd.read_csv(out / "test_predictions.csv")
    assert set(metrics) == {"qihga_gnn", "random_forest", "svm", "logistic_regression"}
    assert rows.row_index.tolist() == run["rows"]["test"]
    assert sum(run["fusion_weights"].values()) == pytest.approx(1)
    assert all(set(cols) & set(run["selected_features"]) for cols in run["modalities"].values())
    for model, score in metrics.items():
        positive = rows.positive_label_indicator == 1
        predicted = rows[model] == 1
        tp = (positive & predicted).sum()
        denominator = positive.sum() + predicted.sum()
        assert score["f1"] == pytest.approx(2 * tp / denominator if denominator else 0)
        assert score["tp"] == tp
    assert pd.read_csv(out / "sensitivity.csv").iloc[0].accuracy == pytest.approx(
        metrics["qihga_gnn"]["accuracy"]
    )
    assert (out / "weights.pt").is_file()
    assert len(pd.read_csv(out / "shap_summary.csv")) == 4


def test_only_training_rows_determine_preprocessing(inputs):
    x, y, *_ = load_inputs(inputs)
    base, train, val, test, means, scaler = partition(x, y, 7)
    assert len(set(train) | set(val) | set(test)) == len(train) + len(val) + len(test) == len(x)
    perturbed = x.copy()
    perturbed[np.r_[val, test]] = 5000
    changed, *_, other_means, other_scaler = partition(perturbed, y, 7)
    np.testing.assert_array_equal(changed[train], base[train])
    np.testing.assert_array_equal(other_means, means)
    np.testing.assert_array_equal(other_scaler.scale_, scaler.scale_)


@pytest.mark.parametrize(
    "schema, error",
    [
        ('{"a":["a1"], "b":["label"]}', "Exclude the label"),
        ('{"a":["a1"], "b":["a1"]}', "exactly once"),
        ('{"a":["a1"], "a":["a2"], "b":["b1"]}', "Duplicate modality"),
    ],
)
def test_invalid_modality_configuration(inputs, schema, error):
    inputs.modalities.write_text(schema)
    with pytest.raises(ValueError, match=error):
        load_inputs(inputs)


def test_single_feature_and_constant_nodes_have_self_loops():
    graph = feature_graph(np.ones((6, 1)), 5)
    np.testing.assert_array_equal(graph.numpy(), [[1]])


def test_interrupted_export_can_be_retried(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    with pytest.raises(RuntimeError), result_directory(out) as staged:
        (staged / "partial.txt").write_text("incomplete")
        raise RuntimeError("simulated write failure")
    assert list(out.iterdir()) == []
    with result_directory(out) as staged:
        (staged / "complete.txt").write_text("ok")
    assert (out / "complete.txt").read_text() == "ok"


def test_second_publisher_preserves_first_result(tmp_path):
    out = tmp_path / "results"
    with pytest.raises(OSError), result_directory(out) as second:
        (second / "metrics.json").write_text("second")
        with result_directory(out) as first:
            (first / "metrics.json").write_text("first")
    assert (out / "metrics.json").read_text() == "first"


def test_measured_elite_survives_across_generations(inputs, monkeypatch):
    from aiam_gnn.qihga import Chromosome, search

    observed = []
    original = Chromosome.measure

    def record(self, rng, groups):
        candidate = original(self, rng, groups)
        observed.append(candidate)
        return candidate

    monkeypatch.setattr(Chromosome, "measure", record)
    x, y, _, _, groups = load_inputs(inputs)
    x, tr, va, *_ = partition(x, y, 42)
    args = Namespace(
        seed=42,
        population=1,
        generations=3,
        surrogate_epochs=1,
        epochs=1,
        patience=1,
        neighbors=2,
        encoder="gcn",
        rescore=1,
    )
    config, _, trace = search(x[tr], y[tr], x[va], y[va], groups, args)
    assert len(observed) == 1
    assert config is observed[0]
    scores = [item["best_surrogate_fitness"] for item in trace if "generation" in item]
    assert scores == [scores[0]] * args.generations
