"""Train the QIHGA–GNN ensemble and three baseline models on a supplied table."""

import argparse
import json
import sys
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .graph_model import probabilities
from .qihga import search


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV, TSV or XLSX (first sheet)")
    parser.add_argument("--label", required=True, help="Binary target column name")
    parser.add_argument("--positive", required=True, help="Positive class value, matched as text")
    parser.add_argument(
        "--modalities", required=True, type=Path, help="JSON object mapping views to feature lists"
    )
    parser.add_argument("--output", required=True, type=Path, help="New or empty directory")
    parser.add_argument("--encoder", choices=("gcn", "gat"), default="gcn")
    parser.add_argument("--seed", type=int, default=42)
    for name, default in (
        ("neighbors", 5),
        ("population", 8),
        ("generations", 5),
        ("surrogate-epochs", 6),
        ("rescore", 3),
        ("epochs", 100),
        ("patience", 12),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument(
        "--explain", action="store_true", help="Estimate permutation SHAP and noise sensitivity"
    )
    args = parser.parse_args(argv)
    if any(
        getattr(args, name) < 1
        for name in (
            "neighbors",
            "population",
            "generations",
            "surrogate_epochs",
            "rescore",
            "epochs",
            "patience",
        )
    ):
        parser.error("Search and training counts must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("Use a seed between 0 and 4294967295.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must name a new or empty directory.")
    return args


def unique_modalities(pairs):
    result = {}
    for name, columns in pairs:
        if name in result:
            raise ValueError(f"Duplicate modality name: {name}")
        result[name] = columns
    return result


def load_inputs(args):
    ext = args.input.suffix.lower()
    if ext == ".xlsx":
        frame = pd.read_excel(args.input, dtype={args.label: str})
    elif ext in (".csv", ".tsv"):
        frame = pd.read_csv(args.input, sep="\t" if ext == ".tsv" else ",", dtype={args.label: str})
    else:
        raise ValueError("Use a .csv, .tsv or .xlsx table.")
    views = json.loads(
        args.modalities.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_modalities
    )
    if not isinstance(views, dict) or len(views) < 2:
        raise ValueError("Specify at least two named modalities.")
    names, groups = [], []
    for view, cols in views.items():
        if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
            raise ValueError(f"Modality {view!r} needs a nonempty list of feature column names.")
        groups.append(np.arange(len(names), len(names) + len(cols)))
        names.extend(cols)
    if len(set(names)) != len(names) or args.label in names:
        raise ValueError(
            "Each feature must occur exactly once. Exclude the label from the features."
        )
    for col in [*names, args.label]:
        if col not in frame:
            raise ValueError(f"Missing column: {col}")
    target = frame[args.label]
    if target.isna().any() or target.nunique() != 2 or args.positive not in set(target):
        raise ValueError("Supply a complete binary target and its exact positive class value.")
    features = frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(features).any():
        raise ValueError("Infinite feature values must be corrected before training.")
    return features, (target == args.positive).to_numpy(dtype=np.int64), views, names, groups


def partition(x, y, seed):
    indices = np.arange(len(y))
    pool, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(pool, test_size=0.2, stratify=y[pool], random_state=seed)
    if any(len(np.unique(y[part])) != 2 for part in (train, val, test)):
        raise ValueError(
            "Both classes must be present in the training, validation and test subsets."
        )
    if np.isnan(x[train]).all(axis=0).any():
        raise ValueError("A feature has no observed training values.")
    means = np.nanmean(x[train], axis=0)
    filled = np.where(np.isnan(x), means, x)
    scaler = StandardScaler().fit(filled[train])
    scaled = scaler.transform(filled).astype(np.float32)
    return scaled, train, val, test, means, scaler


def scores(y, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def explain(model, train, test, truth, names, config, output, seed):
    """Interventional permutation Shapley estimates with training-row backgrounds."""
    rng = np.random.default_rng(seed)
    subset = test[: min(32, len(test))]
    values = np.zeros((len(subset), len(names)))
    selected = np.flatnonzero(config.mask)
    for _ in range(32):
        current = train[rng.integers(len(train), size=len(subset))].copy()
        previous = probabilities(model, current)[:, 1]
        for j in rng.permutation(selected):
            current[:, j] = subset[:, j]
            updated = probabilities(model, current)[:, 1]
            values[:, j] += updated - previous
            previous = updated
    pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values / 32).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(output / "shap_summary.csv", index=False)
    records = []
    for level in np.linspace(0, 1, 11):
        noisy = test + rng.normal(0, level, test.shape).astype(np.float32)
        records.append(
            {
                "noise_sd": level,
                "accuracy": accuracy_score(truth, probabilities(model, noisy).argmax(1)),
            }
        )
    pd.DataFrame(records).to_csv(output / "sensitivity.csv", index=False)


@contextmanager
def result_directory(destination):
    """Publish a complete run by renaming a sibling directory on the same volume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".aiam-run-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "results"
        staged.mkdir()
        yield staged
        # rmdir only accepts an empty directory. Preserve files written by a
        # concurrent run or the user while this experiment was training.
        try:
            destination.rmdir()
        except FileNotFoundError:
            pass
        staged.rename(destination)


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw, y, views, names, groups = load_inputs(args)
    x, tr, va, te, means, scaler = partition(raw, y, args.seed)
    config, model, history = search(x[tr], y[tr], x[va], y[va], groups, args)
    results = {}
    predictions = pd.DataFrame({"row_index": te, "positive_label_indicator": y[te]})
    probabilities_test = probabilities(model, x[te])
    predictions["qihga_gnn_positive_probability"] = probabilities_test[:, 1]
    predictions["qihga_gnn"] = probabilities_test.argmax(axis=1)
    results["qihga_gnn"] = scores(y[te], predictions["qihga_gnn"])
    baselines = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=1),
        "svm": SVC(kernel="rbf"),
        "logistic_regression": LogisticRegression(max_iter=2000),
    }
    for name, estimator in baselines.items():
        estimator.fit(x[tr], y[tr])
        predictions[name] = estimator.predict(x[te])
        results[name] = scores(y[te], predictions[name])
    run = {
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {
            "python": sys.version.split()[0],
            **{
                pkg: version(pkg)
                for pkg in ("numpy", "pandas", "scikit-learn", "torch", "openpyxl")
            },
        },
        "modalities": views,
        "features_in_order": names,
        "selected_features": [n for n, keep in zip(names, config.mask) if keep],
        "hidden_width": config.hidden,
        "depth": config.depth,
        "learning_rate": config.lr,
        "fusion_weights": dict(zip(views, model.fusion_logits.detach().softmax(0).tolist())),
        "imputation_means": means.tolist(),
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scales": scaler.scale_.tolist(),
        "rows": {"train": tr.tolist(), "validation": va.tolist(), "test": te.tolist()},
        "search_history": history,
    }
    with result_directory(args.output) as out:
        for name, content in (("run.json", run), ("metrics.json", results)):
            (out / name).write_text(
                json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
        predictions.to_csv(out / "test_predictions.csv", index=False)
        torch.save(model.state_dict(), out / "weights.pt")
        if args.explain:
            explain(model, x[tr], x[te], y[te], names, config, out, args.seed)
    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
