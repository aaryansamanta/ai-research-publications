> Archived copy of the earlier code README. Links have been adjusted for its location under `docs/`. For current instructions, use [code/README.md](../code/README.md).

# Quantum-Inspired Hybrid Genetic Algorithm and Graph Neural Network Ensemble for Multimodal Classification

Research source code for the QIHGA–GNN feature-graph approach to binary classification. Each modality has its own graph encoder. A quantum-inspired evolutionary search chooses features and encoder settings, and the final model learns how to weight the modalities' predictions.

Aaryan Samanta, IEEE AUTEEE 2025, pp. 808–814. [DOI 10.1109/AUTEEE67053.2025.11322272](https://doi.org/10.1109/AUTEEE67053.2025.11322272)

## Quick start

From the repository root, use Python 3.10 or later to install the package in your active environment.

```sh
python -m pip install .
aiam-gnn --help
```

Prepare a CSV, TSV or XLSX table with one row per independent sample, numeric features, and a binary target. Excel input uses the first sheet. Copy [the modality template](../code/configs/modalities.example.json) to `modalities.json`, replace its placeholder names with your columns, and list each feature exactly once. Exclude the target, identifiers and target-derived features. Specify the positive class exactly as it appears in the table.

```sh
aiam-gnn --input measurements.csv --label target --positive 1 --modalities modalities.json --encoder gcn --seed 42 --output results
```

Choose a new or empty output directory. Select `--encoder gat` for graph attention and `--explain` for feature attribution and noise sensitivity. The command also works as `python -m aiam_gnn`. From a source checkout, `python run_experiment.py` accepts the same arguments.

## Repository guide

| Path | Purpose |
| --- | --- |
| [aiam_gnn/qihga.py](../code/aiam_gnn/qihga.py) | Evolutionary search, candidate scoring and elite retention |
| [aiam_gnn/graph_model.py](../code/aiam_gnn/graph_model.py) | Feature graphs, modality encoders, training and probability fusion |
| [aiam_gnn/cli.py](../code/aiam_gnn/cli.py) | Input validation, data splitting, baselines and evaluation |
| [experiment.ipynb](../code/experiment.ipynb) | Interactive experiment and results inspection |
| [tests/](../code/tests/) | Regression and end-to-end tests using temporary inputs |
| [.github/workflows/tests.yml](../.github/workflows/tests.yml) | Test workflow for pushes and pull requests |

To run the notebook, install the notebook extra with `python -m pip install ".[notebook]"`, launch `jupyter lab experiment.ipynb`, and set the input paths in its first code cell. To run the tests:

```sh
python -m pip install ".[test]"
python -m pytest -q
```

## Method and implementation settings

| Research method | Source code and configuration |
| --- | --- |
| III.A, IV.A · Preprocessing and feature graphs | The outer stratified split holds out 20% of rows for testing. An inner stratified split reserves 20% of the remaining rows for validation. Training rows determine column means, scaling and absolute feature correlations. Each modality has a top-k correlation graph with symmetric edges and self-loops. |
| III.B–D · QIHGA and fitness | Each feature has amplitudes given by the cosine and sine of an angle. Measurement samples a binary mask. The search performs tournament selection, amplitude crossover, guided rotation, mutation and elitism. At least one feature survives in each modality. Additional genes determine encoder width, depth and learning rate. |
| III.C, III.G · Candidate selection | A short training schedule scores candidates by validation accuracy with penalties for the retained feature fraction and the parameter-count proxy. The strongest candidates are retrained on a longer schedule and reranked on validation. The test set is reserved for final evaluation. |
| III.E · Graph encoders | Each modality uses either symmetrically normalized GCN propagation or masked, single-head GAT propagation. Trainable feature-node embeddings are pooled with each sample's selected standardized feature values. |
| III.F–G · Late fusion | A softmax over trainable modality weights defines a convex mixture of class probabilities. The network optimizes negative log likelihood with weight decay and dropout. Mask sparsity is applied during the discrete search. Validation loss controls early stopping. |
| IV.B–D · Evaluation | Random forest, RBF SVM and logistic regression use all listed features and the same training and test rows. Metrics include accuracy, positive-class precision, recall, F1 and confusion counts. |

The experiment section describes 1,000 synthetic samples and 50 features. Input size is configurable. Set modality membership from your data dictionary. Defaults such as 8 chromosomes, 5 generations, 5 neighbors, hidden widths of 8/16/32, the validation fraction and the seed are implementation settings. The notebook provides an interactive experiment workflow. New runs compute metrics from the selected input dataset.

## Run outputs

| File | Contents |
| --- | --- |
| `metrics.json` | Test metrics and confusion counts per model. |
| `test_predictions.csv` | Zero-based input row index, binary target, hybrid positive probability and the four model predictions. |
| `run.json` | Python and dependency versions, split indices, preprocessing statistics, feature order, selected features, hyperparameters, search trace and learned fusion weights. |
| `weights.pt` | The fitted ensemble's PyTorch state dictionary, including the feature-graph buffers. |
| `shap_summary.csv` | Optional mean absolute interventional permutation Shapley estimates for positive-class probability. Uses up to 32 test rows, 32 permutations and training-row backgrounds. |
| `sensitivity.csv` | Optional accuracy under independent Gaussian perturbations of standardized test features at noise standard deviations from 0 to 1. |

The split assumes independent rows. Studies with repeated observations per subject or time-dependent sampling require a subject- or time-aware split.
