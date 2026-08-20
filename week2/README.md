# MovieLens 100K baselines: SoftImpute + IGMC

This folder implements the two baselines for the MovieLens 100K experiment we discussed.

## Shared experimental protocol

- Dataset: MovieLens 100K.
- Representation: observed `(user, item, rating)` triples with zero-based user/item IDs.
- Split: one deterministic random **70% train / 20% validation / 10% test** rating-row split (`seed=42` by default).
- The exact same saved split files are reused by both baselines.
- Validation is used for model/hyperparameter selection.
- Test is touched only after selection.
- Primary metric: RMSE. MAE is reported as a secondary metric.
- Predictions are not clipped by default; pass `--clip` to both scripts if you want the same `[1,5]` clipping rule on both.

The scripts automatically download `ml-100k.zip` from GroupLens on first run and cache it under `data/`.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. SoftImpute

```bash
python softimpute_baseline.py
```

Default validation grid:

- truncated ranks: `20 50 100`
- shrinkage values: `0.05 0.10 0.20 0.40` times the training matrix's `lambda_max`
- up to 50 Soft-Impute iterations per candidate

The selected `(rank, lambda fraction)` is refit on train+validation, then evaluated once on test.

Outputs:

- `results/softimpute_results.json`
- `results/softimpute_test_predictions.csv`

## 3. IGMC

```bash
python igmc_baseline.py --device cuda
```

The default model follows the ICLR 2020 IGMC setup closely:

- 1-hop enclosing subgraphs
- target `(user,item)` edge removed before message passing
- IGMC structural node labels: target user `0`, target item `1`, 1-hop users `2`, 1-hop items `3`
- rating values `1..5` become five R-GCN relation types
- four relation-aware message-passing layers of 32 hidden units each
- basis decomposition with 4 bases
- concatenate all four layer states
- read out only the target user and target item representations
- MLP hidden size 128, dropout 0.5
- adjacency dropout 0.2
- adjacent-rating regularisation weight `0.001`
- Adam, learning rate `0.001`, decay by `0.1` after 50 epochs
- batch size 50
- maximum 200 sampled nodes per side of the 1-hop subgraph

Unlike the old reference implementation, this version uses plain modern PyTorch and does **not** require PyTorch Geometric.

The validation RMSE is checked every 5 epochs; the best epoch count is then used for a clean train+validation refit before final test evaluation.

Outputs:

- `results/igmc_results.json`
- `results/igmc_test_predictions.csv`
- `results/igmc_final.pt`

### Quick smoke test

A full IGMC run is much heavier than SoftImpute. To verify the pipeline on CPU first:

```bash
python igmc_baseline.py \
  --device cpu \
  --epochs 2 \
  --eval-every 1 \
  --train-limit 1000 \
  --val-limit 250 \
  --test-limit 250 \
  --final-train-limit 1250 \
  --max-nodes-per-hop 25
```

Those `--*-limit` flags are only for debugging; leave them unset for the actual baseline.

## Why the target edge deletion matters

For a training pair `(u,v,r)`, `r` is the label we are trying to predict. If the `(u,v)` edge carrying relation type `r` remains inside the input subgraph, the network can directly read the answer from the graph. This implementation removes that edge for every training subgraph. Validation/test edges never enter the graph in the first place.
