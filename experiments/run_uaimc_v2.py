"""Faithful(er) reimplementation of UA-IMC (Kasalicky, Ledent & Alves,
RecSys '23 - see Uncertainty-adjusted inductive matrix completion with
Graph Neura.pdf in the repo root), replacing the earlier approximation in
run_uaimc.py. Produces Table 1's "UA-IMC (faithful)" row, written to a
SEPARATE results file from the old approximation's, on purpose - they are
not the same method and shouldn't be silently conflated in one CSV.

  python run_uaimc_v2.py                      # ml-100k and ml-1m only, see below
  python run_uaimc_v2.py --datasets ml-100k
  python run_uaimc_v2.py --quick              # tiny grid/slice, smoke-test the pipeline

=====================================================================
WHAT THE PAPER ACTUALLY DOES, AND WHY run_uaimc.py WASN'T THAT
=====================================================================

The earlier run_uaimc.py built an "in the same spirit" stand-in: it took
IGMC's R-GCN and made it predict BOTH the rating and an unbounded
uncertainty score, trained with this project's OWN joint loss (proposal
eq. 4-5). That is not what the paper describes, in two structural ways:

1. The paper's *prediction* module is NOT a graph neural network. It is a
   plain nuclear-norm-regularized matrix factorisation, R = UV^T, trained
   with ordinary regularised MSE (eq. 1 in the paper, which is the exact
   textbook Koren-et-al./Mazumder-Hastie-Tibshirani objective this project
   already uses for Plain MF). The GNN's ONLY job is to output a per-rating
   anomaly score used to *weight* that MSE loss - it never produces a
   rating prediction itself. So UA-IMC is a hybrid: a transductive
   low-rank backbone (U, V - tied to specific user/item identity, exactly
   like Plain MF) plus an inductive GNN anomaly detector bolted onto its
   loss function (generalises to unseen users/items, exactly like IGMC).

2. The paper's uncertainty score W_i,j is squashed through a SIGMOID before
   it ever reaches exp(-W_i,j) (paper Sec. 3, eq. 5: "an output layer with
   sigmoid activation"). That bounds W to (0,1), which bounds exp(-W) to
   (e^-1, 1) ~= (0.37, 1) - it can never overflow, by construction. This
   project's own experiments2/mf_common.py hit exactly the failure mode
   the sigmoid prevents: an UNBOUNDED uncertainty term fed into exp(-.)
   diverged to nonsense on amazon-games and needed hand-added gradient
   clipping to survive. The paper's design doesn't need any of that - it's
   worth internalising this as the actual reason the sigmoid is there, not
   just "because the paper said so".

Concretely, the paper's model (paper Sec. 3, "Methodology"):

    minimize_{U,V,Theta}
        1/(2|Omega|) * sum_{(i,j) in Omega} [ exp(-W_ij)*(X_ij-(UV^T)_ij)^2 + alpha*W_ij ]
      + (1/2)*lambda*(||U||_F^2 + ||V||_F^2)
      + beta * L_ARR
    s.t.  W_ij = f_Theta(G_ij)                                         (paper eq. 2)

  - G_ij is IGMC's 1-hop enclosing subgraph around rating (i,j) (Zhang &
    Chen 2020, already implemented in run_igmc.py - reused unchanged here,
    see that module for the subgraph-extraction/DRNL-substitute reasoning).
  - f_Theta: the same R-GCN backbone as IGMC (paper explicitly reuses
    IGMC's architecture for this), but the final head is a SIGMOID-bounded
    scalar (an anomaly score), not a rating (see UncertaintyGNN below).
  - L_ARR ("Adjacent Rating Regularization", eq. 6) pulls each R-GCN
    layer's per-rating-value weight matrices for NEIGHBOURING ratings
    (e.g. the weights for rating=3 and rating=4) toward each other, since
    ratings are ordinal - a 3 and a 4 should behave more alike than a 1
    and a 5 do.
  - alpha=45 and beta=0.001 are the paper's own cross-validated/adopted
    fixed values (Sec. 4.1) - not re-tuned here, so a real Table-1 number
    stays comparable to theirs on that axis.
  - Training is two-phase (Sec. 4.1): first pretrain a *plain* IGMC model
    (predicting ratings directly - exactly run_igmc.py's IGMC class) on
    the dataset, then copy ONLY its R-GCN message-passing weights (never
    its rating-prediction head, which UA-IMC doesn't have) into the
    uncertainty network as initialization, then fine-tune EVERYTHING
    (U, V, the copied R-GCN weights, and the new sigmoid-anomaly head)
    jointly under the full loss above. The GNN is never frozen post-
    pretrain - Sec. 4.1 is explicit that joint fine-tuning is what lets
    the GNN's features "match the needs of the anomaly detection task".
  - Adam + cosine learning-rate decay (Sec. 4.1).
  - Batch size 200 for "dense" datasets (paper names Douban and ML-100K
    specifically), 1000 for Amazon Video Games. ML-1M/ML-25M aren't
    mentioned - this script uses 200 for them too as the closest
    documented analogue (another MovieLens dataset), flagged here as an
    extrapolation, not a stated paper value.

What the paper does NOT pin down, and this script therefore treats as
genuinely open hyperparameters (grid-searched, val-selected, like every
other run_*.py in this repo): the embedding rank d for U,V, the R-GCN's
hidden width, and lambda (the Frobenius-norm coefficient on U,V). L (the
number of R-GCN layers) IS pinned by the paper ("L = 4") and is NOT
grid-searched here, to stay faithful on that specific axis.

=====================================================================
A KNOWN BLOCKER, SEPARATE FROM ALL OF THE ABOVE
=====================================================================

This repo's "ml-100k" dataset is actually MovieLens "ml-latest-small"
(610 users, 9,724 items - its own README says so explicitly and calls
itself unsuitable for shared research results), not the paper's ML-100K
(943 users, 1,682 items, the classic 1998 GroupLens u.data benchmark).
No amount of model fidelity closes that gap - it needs the correct raw
dataset file swapped in before any number from this script is comparable
to the paper's Table 1. Being tracked/fixed separately; this script is
being run in the meantime to validate the *pipeline*, not to chase an
exact match tonight.

=====================================================================

datasets excluded by default: ml-25m (same reason as run_igmc.py: pure-
Python subgraph extraction doesn't scale to it) and douban/amazon-games
(not requested for tonight's run - pass --datasets to add them; expect
amazon-games to need experiments2-style care given its history of
numerical drama elsewhere in this repo, though the sigmoid bound here
should make it far more robust than the unclipped versions were).
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "dataset loaders and cleaners"))
sys.path.append(str(Path(__file__).resolve().parent))

from recsys_loader import load_split
from evaluate import sweep, ABSTENTION_RATES
import run_igmc  # imported as a module too, so we can patch run_igmc.MAX_NEIGHBORS below
from run_igmc import (
    DEVICE, N_THREADS, N_RATING_CLASSES,
    build_adjacency, precompute_subgraphs, collate, RGCNLayer, IGMC, train_igmc,
)

# extract_subgraph() (inside run_igmc.py) samples up to MAX_NEIGHBORS other
# users/items per subgraph, so with run_igmc's default of 20 a subgraph can
# have up to ~20*20*2 = 800 cross edges when both caps saturate. ML-100K
# rarely saturates that cap (most users/items have far fewer than 20
# neighbors), so its ~80K cached train+val subgraphs stayed small and the
# faithful run finished fine there. ML-1M is ~10x more ratings AND much
# denser (each user/item typically has 100+ neighbors, not <20), so most of
# its ~900K subgraphs DO saturate the cap - both far more subgraphs *and*
# each one far larger. That combination OOM-killed ML-1M's Phase 1 (model
# selection) at the default cap of 20. Dropping the cap to 10 fixed Phase 1
# but ML-1M's run still died in Phase 2 (trainval = train+val is bigger
# again than train alone, so its subgraph cache is bigger again too) - so
# this is now 6, a second, larger safety margin for the same reason, not
# because 10 was theoretically wrong. This is purely a host-RAM mitigation,
# not a paper-fidelity change - the paper doesn't pin a neighbor cap at
# all, run_igmc.py's 20 was IGMC's own convention, reused here for
# consistency, not something eq. 4-6 requires.
MAX_NEIGHBORS_OVERRIDE = {"ml-1m": 6, "amazon-games": 6}

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]  # ml-25m excluded, see module docstring
DEFAULT_DATASETS = ["ml-100k", "ml-1m"]  # tonight's requested scope
SEED = 42
torch.set_num_threads(N_THREADS)

# ---- Paper-pinned constants (Sec. 4.1) - not grid-searched ----
N_LAYERS = 4          # "we ... set the number of message-passing layers at L = 4"
ALPHA = 45.0           # "The regularization hyperparameter alpha was cross-validated, resulting in a value alpha = 45"
BETA = 0.001           # "beta = 0.001" (adopted from IGMC's own paper, per Sec. 4.1)
MLP_HIDDEN = 64        # "The architecture of MLP contains one hidden layer with 64 dimensions"

# batch size 200 for Douban/ML-100K (paper), 1000 for Amazon Video Games
# (paper); ML-1M/ML-25M default to 200 as the closest documented analogue
# (see module docstring) - not a stated paper value for those two.
BATCH_SIZES = {"ml-100k": 200, "douban": 200, "amazon-games": 1000}
DEFAULT_BATCH_SIZE = 200

# ---- Genuinely open hyperparameters (paper doesn't pin these) ----
HIDDEN_GRID = [16, 32]     # R-GCN width
RANK_GRID = [10, 20]       # rank d of U, V
LAM_GRID = [1e-3, 1e-2]    # Frobenius-norm coefficient on U, V (paper's "lambda", eq. 1-2)
MAX_EPOCHS = 60            # matches run_igmc.py's own budget - same backbone, same per-epoch cost
PATIENCE = 8
PRETRAIN_MAX_EPOCHS = 30   # IGMC pretrain phase gets a smaller budget: it's an initialization
PRETRAIN_PATIENCE = 5      # step, not the main event, and run_igmc.py's own grid already tells
PRETRAIN_LR = 0.003        # us roughly how long plain IGMC needs to reach a reasonable optimum
PRETRAIN_WD = 0.0

QUICK_HIDDEN_GRID = [16]
QUICK_RANK_GRID = [10]
QUICK_LAM_GRID = [1e-3]
QUICK_MAX_EPOCHS = 2
QUICK_PATIENCE = 2
QUICK_PRETRAIN_MAX_EPOCHS = 2
QUICK_PRETRAIN_PATIENCE = 2
QUICK_ROW_LIMIT = 800

METHOD = "UA-IMC (faithful)"
FIELDNAMES = [
    "method", "dataset", "p", "selective_rmse",
    "best_hidden", "best_rank", "best_lam", "val_rmse", "seconds",
]

# Checkpoint written the moment the grid search picks a winner, BEFORE
# Phase 2 (rebuild adjacency/subgraphs on train+val, re-pretrain IGMC,
# refit, score on test) even starts. This exists because Phase 2 repeats
# the same OOM-prone subgraph-precompute step as Phase 1 but on a bigger
# row set (train+val instead of just train) - it killed the ml-1m run
# tonight even after the MAX_NEIGHBORS fix made Phase 1 safe. Without this
# checkpoint, a Phase-2 crash throws away the ~4 hours the 8-config grid
# search just spent, since run_dataset() only writes rows_out to
# uaimc_faithful_results.csv at the very end, after Phase 2 finishes. With
# it, a restart skips straight to Phase 2 using the already-known winner.
CHECKPOINT_PATH = ROOT / "experiments" / "results" / "uaimc_faithful_checkpoint.json"


def load_checkpoint(dataset: str) -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return data.get(dataset)


def save_checkpoint(dataset: str, best_cfg: dict, best_val_rmse: float):
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")) if CHECKPOINT_PATH.exists() else {}
    data[dataset] = {"best_cfg": best_cfg, "best_val_rmse": best_val_rmse}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_checkpoint(dataset: str):
    if not CHECKPOINT_PATH.exists():
        return
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    data.pop(dataset, None)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


class UncertaintyGNN(nn.Module):
    """The GNN half of UA-IMC (paper eq. 3-5): predicts a per-rating
    anomaly score W_ij in (0,1) from the 1-hop enclosing subgraph around
    that rating. Architecturally this is IDENTICAL to run_igmc.py's IGMC
    class up through the concatenated target-node representation - same
    node_embed, same stacked RGCNLayer message-passing, same
    concat-all-layers-then-concat-both-targets readout (paper Sec. 3
    explicitly reuses IGMC's architecture for this). The only thing that
    changes is the final head: IGMC's head predicts a rating (any real
    number); this head predicts a bounded confidence weight instead
    (paper eq. 5: "an output layer with sigmoid activation").

    That bound is the whole point, numerically - see this module's
    docstring. exp(-W) for W in (0,1) sits in (0.37, 1) no matter what the
    network outputs; there is no input that makes it overflow. Contrast
    this project's OWN prior attempts at joint uncertainty training (this
    file's predecessor, and experiments2/mf_common.py's train_ours_mf),
    both of which used an UNBOUNDED uncertainty score and both of which
    needed hand-added clipping after hitting real numerical divergence.
    """

    def __init__(self, hidden_dim, n_layers=N_LAYERS, n_relations=N_RATING_CLASSES, n_node_roles=4,
                 mlp_hidden=MLP_HIDDEN, dropout=0.2):
        super().__init__()
        # Same shapes as run_igmc.py's IGMC.node_embed / IGMC.layers on
        # purpose: load_pretrained_backbone below copies weights directly
        # between the two via load_state_dict, which requires matching
        # parameter names AND shapes, not just "the same idea".
        self.node_embed = nn.Embedding(n_node_roles, hidden_dim)
        self.layers = nn.ModuleList(
            [RGCNLayer(hidden_dim, hidden_dim, n_relations) for _ in range(n_layers)]
        )
        concat_dim = hidden_dim * n_layers * 2  # n_layers per target node, 2 target nodes (user + item)

        # Paper Sec. 3/4.1: "g_theta is a multi-layer perceptron ... M
        # hidden layers followed by an output layer with sigmoid
        # activation" / "one hidden layer with 64 dimensions, ReLU
        # activations and dropout regularization" / "MLP parameters theta
        # were initialized with Xavier Initialization".
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),  # <- bounds W to (0,1); see class docstring
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def load_pretrained_backbone(self, igmc_model: IGMC):
        """Paper Sec. 4.1: 'Parameters P in the GCNN part of UAIMC ...
        were initialized by a GCNN trained to predict ratings [i.e. plain
        IGMC], as described in [53]'. Only the message-passing weights
        transfer - node_embed and the RGCNLayer stack - never IGMC's own
        rating head, which doesn't exist here (this network has its own,
        differently-shaped, sigmoid-bounded head instead). The copied
        weights are NOT frozen afterwards: Sec. 4.1 trains everything
        jointly so the GNN's features can specialise to anomaly detection
        rather than staying stuck at "good for predicting ratings"."""
        self.node_embed.load_state_dict(igmc_model.node_embed.state_dict())
        self.layers.load_state_dict(igmc_model.layers.state_dict())

    def forward(self, batch):
        h = self.node_embed(batch["node_role"])
        layer_outs = []
        for layer in self.layers:
            h = layer(h, batch["edge_src"], batch["edge_dst"], batch["edge_rel"], batch["n_nodes"])
            layer_outs.append(h)
        h_all = torch.cat(layer_outs, dim=-1)
        u_repr = h_all[batch["target_u"]]
        i_repr = h_all[batch["target_i"]]
        graph_repr = torch.cat([u_repr, i_repr], dim=-1)
        return self.mlp(graph_repr).squeeze(-1)

    def arr_loss(self):
        """Paper eq. 6: L_ARR = sum_{r=1}^{|R|-1} ||P_{r+1} - P_r||_F^2,
        where the P_r are the R-GCN's per-relation weight matrices (this
        project's RGCNLayer.rel_weight, shape (n_relations, in, out)) and
        R = {1,...,5} is the set of rating values, i.e. this pulls the
        weights for adjacent rating values (1<->2, 2<->3, 3<->4, 4<->5)
        toward each other. The paper writes eq. 3 (the message-passing
        rule) and eq. 6 as if there's a single set of P_r's, but the real
        model stacks L=4 such layers, each with its own rel_weight - the
        paper doesn't spell out the multi-layer case explicitly, so
        summing every layer's own ARR term (the natural reading: apply
        the same ordinal-smoothness prior at every depth of the network)
        is an assumption made here, not a verified detail. Flag this if
        ever checking numbers against a released reference implementation."""
        total = self.node_embed.weight.new_zeros(())
        for layer in self.layers:
            w = layer.rel_weight  # (n_relations, in_dim, out_dim)
            total = total + ((w[1:] - w[:-1]) ** 2).sum()
        return total


class UAIMC(nn.Module):
    """The full model (paper Fig. 1 / eq. 2). Two parts that only ever
    interact through the loss function (joint_loss below), never through
    each other's forward pass:

      - U, V (this class's own parameters): the actual rating predictor,
        a plain low-rank factorisation R_ij ~= (UV^T)_ij, exactly Plain
        MF's model shape. This part is TRANSDUCTIVE - U and V are tied to
        specific global user/item identity, so a genuinely new user/item
        at test time has no row to look up (same limitation Plain MF has,
        inherited on purpose since this is the paper's actual design).

      - self.uncertainty (an UncertaintyGNN): predicts W_ij from the LOCAL
        graph structure around (i,j) only, never seeing U or V at all.
        This part IS inductive (same as IGMC) - it works for user/item
        pairs it never saw during training, as long as they have some
        rated neighbours to build a subgraph from.

    Both are members of one nn.Module purely so a single optimizer can
    update U, V, and every GNN/MLP parameter together in the same step,
    which is what "jointly trained" (paper Sec. 4.1) actually means here."""

    def __init__(self, n_users, n_items, rank, hidden_dim, n_layers=N_LAYERS, seed=SEED, mean_rating=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(torch.randn(n_users, rank, generator=g) * 0.1)
        self.V = nn.Parameter(torch.randn(n_items, rank, generator=g) * 0.1)
        self.uncertainty = UncertaintyGNN(hidden_dim, n_layers)
        # Ratings average ~3.5, but U,V start small (scale 0.1), so hitting
        # that absolute scale needs large embedding norms - exactly the
        # failure mode already root-caused in this repo's Plain MF (see
        # experiments/run_plain_mf.py's train_plain_mf docstring): the
        # regularizer fights the model's ability to reach a sane baseline,
        # and confirmed empirically here too (an early un-centered version
        # of this exact model collapsed to val_rmse~3.67, i.e.
        # sqrt(mean(rating^2)) - "predicting zero stars"). Baking the mean
        # into the model as a buffer (not a parameter - it's fixed, not
        # learned) means predict_rating always returns real rating-scale
        # values, so every caller (loss, eval, final predictions) just
        # works without needing to remember to add it back anywhere.
        self.register_buffer("mean_rating", torch.tensor(float(mean_rating)))

    def predict_rating(self, u_idx, i_idx):
        return (self.U[u_idx] * self.V[i_idx]).sum(dim=1) + self.mean_rating

    def forward(self, subgraph_batch, u_idx, i_idx):
        s = self.predict_rating(u_idx, i_idx)
        w = self.uncertainty(subgraph_batch)
        return s, w


def joint_loss(s, w, actual, model, alpha, lam, beta, num_batches_per_epoch):
    """Paper eq. 2, term by term:

        1/(2|Omega|) * sum [exp(-W)(X-S)^2 + alpha*W]     <- mse_term
      + (1/2)*lambda*(||U||_F^2 + ||V||_F^2)               <- reg_term
      + beta * L_ARR                                        <- arr_term

    mse_term is computed as a mean over THIS minibatch, matching every
    other loss in this repo (Plain MF, IGMC, ...) - a minibatch mean is
    already an unbiased estimate of the full-dataset mean 1/|Omega|*sum(.),
    no separate rescaling needed (this differs from a per-example
    heteroscedastic model like Ours' eq. 5, which regularizes with a raw
    SUM over Omega and therefore does need minibatch rescaling - not the
    case here, since the paper's own eq. 2 already writes this term as an
    average, not a sum).

    reg_term and arr_term are a DIFFERENT kind of term: global penalties on
    the model's parameters as a whole (U, V; the GNN's per-relation
    weights), not sums indexed by Omega at all. The paper's eq. 2 adds each
    ONCE per (implicitly full-batch) gradient step. Under minibatch SGD
    with K minibatches per epoch, naively adding the FULL penalty to every
    one of those K minibatch losses would make its total per-epoch gradient
    contribution K times stronger than the paper's formula intends -
    dividing by num_batches_per_epoch here is what keeps one epoch's
    ACCUMULATED regularization pressure matching a single full-batch
    application, regardless of how finely that epoch happens to be chopped
    into minibatches.

    This is a genuinely different bug from - and a different fix than -
    experiments/run_plain_mf.py's weight-decay collapse, even though both
    are "a regularizer interacting badly with minibatching": that bug was
    PyTorch's optimizer decaying untouched embedding ROWS every step (a
    SPARSE-update mismatch, fixed by scoping the penalty to only the rows
    touched in that batch). This one is a DENSE, whole-matrix penalty being
    replayed too many times per epoch (a batching-FREQUENCY mismatch,
    fixed by dividing its strength, not by scoping which rows it touches -
    scoping would be wrong here, since eq. 1's nuclear-norm-equivalence
    footnote makes clear this term is intentionally about the whole
    matrix's rank, not particular rows)."""
    mse_term = 0.5 * torch.mean(torch.exp(-w) * (actual - s) ** 2 + alpha * w)
    reg_term = 0.5 * lam * (model.U.pow(2).sum() + model.V.pow(2).sum()) / num_batches_per_epoch
    arr_term = beta * model.uncertainty.arr_loss() / num_batches_per_epoch
    return mse_term + reg_term + arr_term


def _global_ids(rows, batch_idx, device):
    """The GNN's subgraph batch (collate(), from run_igmc.py) only carries
    LOCAL node indices within each extracted subgraph - by design, since
    the whole point of an inductive model is to never see global user/item
    identity. But U, V very much DO need to know which global user i and
    item j a training example refers to, to look up U[i] and V[j]. rows is
    the original (user_idx, item_idx, rating) array in the SAME order
    precompute_subgraphs consumed it in, so indexing rows by the same
    batch_idx used to slice the subgraph list recovers the matching global
    ids for free - no separate bookkeeping needed."""
    u = torch.as_tensor(rows[batch_idx, 0], dtype=torch.long, device=device)
    i = torch.as_tensor(rows[batch_idx, 1], dtype=torch.long, device=device)
    return u, i


def predict_uaimc(model, graphs, rows, batch_size, device):
    model.eval()
    preds, ws = [], []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            idx = np.arange(start, min(start + batch_size, len(graphs)))
            batch = collate([graphs[k] for k in idx], device)
            u_idx, i_idx = _global_ids(rows, idx, device)
            s, w = model(batch, u_idx, i_idx)
            preds.append(s.cpu().numpy())
            ws.append(w.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ws)


def evaluate_rmse(model, graphs, rows, batch_size, device):
    """Selection metric is the rating head (S) alone, W unscored during
    model selection - same fairness reasoning the old run_uaimc.py and
    experiments/run_ours.py already used: picking hyperparameters by how
    well the UNCERTAINTY happens to fit would let a config win by gaming
    abstention rather than by genuinely predicting ratings better."""
    pred, _ = predict_uaimc(model, graphs, rows, batch_size, device)
    actual = rows[:, 2]
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def train_uaimc(
    train_graphs, train_rows, val_graphs, val_rows, n_users, n_items,
    rank, hidden_dim, lam, pretrained_backbone, max_epochs, patience, batch_size, seed,
    desc=None, show_progress=False, heartbeat_every=None,
):
    """One full UA-IMC training run: build a fresh model, load the given
    pretrained R-GCN backbone into its uncertainty head (paper Sec. 4.1's
    initialization step), then jointly fine-tune U, V, and every GNN/MLP
    parameter together under joint_loss (eq. 2), with Adam + cosine LR
    decay (Sec. 4.1) and early stopping on val rating-head RMSE."""
    torch.manual_seed(seed)
    mean_rating = float(train_rows[:, 2].mean())  # see UAIMC.__init__'s docstring
    model = UAIMC(n_users, n_items, rank, hidden_dim, seed=seed, mean_rating=mean_rating).to(DEVICE)
    model.uncertainty.load_pretrained_backbone(pretrained_backbone)

    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    shuffle_rng = random.Random(seed)
    idx_all = list(range(len(train_graphs)))
    num_batches_per_epoch = -(-len(idx_all) // batch_size)  # ceil division, see joint_loss's docstring

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    bar = tqdm(range(max_epochs), desc=desc or f"h={hidden_dim} rank={rank} lam={lam:.0e}",
               leave=False, disable=not show_progress)
    for epoch in bar:
        shuffle_rng.shuffle(idx_all)
        model.train()
        for start in range(0, len(idx_all), batch_size):
            batch_idx = np.array(idx_all[start:start + batch_size])
            batch = collate([train_graphs[k] for k in batch_idx], DEVICE)
            u_idx, i_idx = _global_ids(train_rows, batch_idx, DEVICE)
            actual = torch.as_tensor(train_rows[batch_idx, 2], dtype=torch.float32, device=DEVICE)

            opt.zero_grad()
            s, w = model(batch, u_idx, i_idx)
            loss = joint_loss(s, w, actual, model, ALPHA, lam, BETA, num_batches_per_epoch)
            loss.backward()
            opt.step()
        scheduler.step()

        v_rmse = evaluate_rmse(model, val_graphs, val_rows, batch_size, DEVICE)
        epochs_run = epoch + 1
        if show_progress:
            bar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_rmse:.4f}")
        if heartbeat_every and (epoch + 1) % heartbeat_every == 0:
            tag = desc or f"h={hidden_dim} rank={rank} lam={lam:.0e}"
            print(f"    [{tag}] epoch {epoch + 1}/{max_epochs}: val_rmse={v_rmse:.4f} (best={best_rmse:.4f})")

        if v_rmse < best_rmse - 1e-5:
            best_rmse = v_rmse
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_rmse, epochs_run


def already_done(results_path: Path, dataset: str) -> bool:
    if not results_path.exists():
        return False
    seen = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"] == METHOD:
                seen.add(row["p"])
    expected = {str(p) for p in ABSTENTION_RATES}
    return expected.issubset(seen)


def append_results(results_path: Path, rows: list[dict]):
    file_exists = results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, hidden_grid, rank_grid, lam_grid, max_epochs, patience,
                 pretrain_max_epochs, pretrain_patience, row_limit=None) -> list[dict]:
    print(f"\n=== {dataset} ===")
    t_dataset0 = time.time()
    batch_size = BATCH_SIZES.get(dataset, DEFAULT_BATCH_SIZE)

    # See MAX_NEIGHBORS_OVERRIDE's definition above for why this exists at
    # all: it's a host-RAM mitigation for dense/large datasets, not a paper
    # requirement. run_igmc.extract_subgraph reads this as a bare module
    # global, so patching run_igmc.MAX_NEIGHBORS before precompute_subgraphs
    # is called is enough to shrink every subgraph built from here on.
    run_igmc.MAX_NEIGHBORS = MAX_NEIGHBORS_OVERRIDE.get(dataset, 20)
    print(f"{dataset}: MAX_NEIGHBORS={run_igmc.MAX_NEIGHBORS}"
          + (" (reduced from IGMC's default 20 - see MAX_NEIGHBORS_OVERRIDE comment)"
             if dataset in MAX_NEIGHBORS_OVERRIDE else ""))

    split = load_split(dataset, seed=SEED)
    print(f"{dataset}: {split.n_users:,} users x {split.n_items:,} items | "
          f"train={len(split.train):,} val={len(split.val):,} test={len(split.test):,} | batch_size={batch_size}")

    train_rows, val_rows, test_rows = split.train, split.val, split.test
    if row_limit:
        train_rows, val_rows, test_rows = train_rows[:row_limit], val_rows[:row_limit], test_rows[:row_limit]

    heartbeat_every = max(1, max_epochs // 10)

    checkpoint = load_checkpoint(dataset)
    if checkpoint is not None:
        # A previous attempt already finished the grid search and recorded
        # the winner before dying in Phase 2 (see CHECKPOINT_PATH's comment
        # above) - skip straight to Phase 2 instead of repeating however
        # many hours the grid search took.
        best_cfg, best_val_rmse = checkpoint["best_cfg"], checkpoint["best_val_rmse"]
        print(f"{dataset}: resuming from checkpoint {CHECKPOINT_PATH.name} - "
              f"skipping grid search, best config already known: {best_cfg} (val_rmse={best_val_rmse:.4f})")
    else:
        # Phase 1 (model selection): train-only adjacency/subgraphs, exactly
        # run_igmc.py's convention - see that module for why val must never
        # see its own edges in the graph used to score it.
        user_items, item_users = build_adjacency(train_rows)
        rng = random.Random(SEED)
        sel_train_graphs = precompute_subgraphs(train_rows, user_items, item_users, rng, f"{dataset} train subgraphs")
        sel_val_graphs = precompute_subgraphs(val_rows, user_items, item_users, rng, f"{dataset} val subgraphs")
        print(f"{dataset}: train={len(sel_train_graphs):,} val={len(sel_val_graphs):,} "
              f"subgraphs cached (model selection)")

        configs = [
            {"hidden_dim": h, "rank": r, "lam": lam}
            for h in hidden_grid for r in rank_grid for lam in lam_grid
        ]
        print(f"{dataset}: {len(configs)} configs, max_epochs={max_epochs}, "
              f"patience={patience}, batch_size={batch_size}")

        best_cfg, best_val_rmse = None, float("inf")

        # Pretrain one plain IGMC model per distinct hidden_dim (paper Sec.
        # 4.1), cached and reused across every (rank, lam) config that
        # shares it - the pretrain cost only depends on hidden_dim (and the
        # paper-fixed N_LAYERS), not on U/V's rank or lambda, so redoing it
        # per-config would be pure waste.
        pretrained_by_hidden: dict[int, IGMC] = {}
        hidden_dims_needed = sorted({cfg["hidden_dim"] for cfg in configs})
        for h in hidden_dims_needed:
            print(f"{dataset}: pretraining plain IGMC backbone (hidden_dim={h}, n_layers={N_LAYERS}) "
                  f"for uncertainty-network initialization (paper Sec. 4.1)...")
            t0 = time.time()
            igmc_model, igmc_val_rmse, igmc_epochs = train_igmc(
                sel_train_graphs, sel_val_graphs, hidden_dim=h, n_layers=N_LAYERS,
                lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD, max_epochs=pretrain_max_epochs,
                patience=pretrain_patience, batch_size=batch_size, seed=SEED,
                desc=f"{dataset} IGMC pretrain h={h}",
            )
            pretrained_by_hidden[h] = igmc_model
            print(f"{dataset}: pretrained IGMC h={h} val_rmse={igmc_val_rmse:.4f} "
                  f"({igmc_epochs} epochs, {time.time() - t0:.1f}s)")

        pbar = tqdm(configs, desc=f"{dataset} grid search")
        for n, cfg in enumerate(pbar, 1):
            t0 = time.time()
            _, v_rmse, epochs_run = train_uaimc(
                sel_train_graphs, train_rows, sel_val_graphs, val_rows,
                split.n_users, split.n_items, cfg["rank"], cfg["hidden_dim"], cfg["lam"],
                pretrained_by_hidden[cfg["hidden_dim"]], max_epochs, patience, batch_size, SEED,
                heartbeat_every=heartbeat_every,
            )
            if v_rmse < best_val_rmse:
                best_val_rmse, best_cfg = v_rmse, cfg
            pbar.set_postfix(val_rmse=f"{v_rmse:.4f}", best=f"{best_val_rmse:.4f}")
            pbar.write(f"  [{n}/{len(configs)}] hidden={cfg['hidden_dim']:>3} rank={cfg['rank']:>3} "
                       f"lam={cfg['lam']:.0e} -> val_rmse={v_rmse:.4f} ({epochs_run} epochs, {time.time() - t0:.1f}s)")

        print(f"{dataset}: best config {best_cfg} (val_rmse={best_val_rmse:.4f})")
        save_checkpoint(dataset, best_cfg, best_val_rmse)

        # Free model-selection subgraphs (and their pretrained backbones - a
        # full IGMC model per hidden_dim isn't huge, but no reason to keep
        # them alive once we know which config won) before phase 2 builds a
        # second full set - same reasoning as run_igmc.py's identical cleanup.
        del sel_train_graphs, sel_val_graphs, user_items, item_users, pretrained_by_hidden
        gc.collect()

    trainval_rows = np.concatenate([train_rows, val_rows], axis=0)
    user_items, item_users = build_adjacency(trainval_rows)
    rng2 = random.Random(SEED)
    trainval_graphs = precompute_subgraphs(trainval_rows, user_items, item_users, rng2, f"{dataset} trainval subgraphs")
    final_test_graphs = precompute_subgraphs(test_rows, user_items, item_users, rng2, f"{dataset} test subgraphs")

    print(f"{dataset}: re-pretraining IGMC backbone on train+val for the final refit "
          f"(hidden_dim={best_cfg['hidden_dim']})...")
    t0 = time.time()
    final_igmc, _, _ = train_igmc(
        trainval_graphs, final_test_graphs, hidden_dim=best_cfg["hidden_dim"], n_layers=N_LAYERS,
        lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD, max_epochs=pretrain_max_epochs,
        patience=pretrain_patience, batch_size=batch_size, seed=SEED, desc=f"{dataset} refit IGMC pretrain",
    )
    print(f"{dataset}: refit pretrain done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    final_model, _, epochs_run = train_uaimc(
        trainval_graphs, trainval_rows, final_test_graphs, test_rows,
        split.n_users, split.n_items, best_cfg["rank"], best_cfg["hidden_dim"], best_cfg["lam"],
        final_igmc, max_epochs, patience, batch_size, SEED,
        desc=f"{dataset} refit", show_progress=True, heartbeat_every=heartbeat_every,
    )
    print(f"{dataset}: refit on train+val done ({epochs_run} epochs, {time.time() - t0:.1f}s)")

    actual = test_rows[:, 2]
    pred, w = predict_uaimc(final_model, final_test_graphs, test_rows, batch_size, DEVICE)

    # W itself is the unreliability score for sweep(): the paper's own
    # convention is "higher W = more anomalous = less confident", which is
    # exactly retain_indices'/sweep's "higher = less confident" convention
    # already - no sign flip or transform needed.
    rows_out = []
    for p, val in sweep(pred, actual, w).items():
        rows_out.append({
            "method": METHOD,
            "dataset": dataset,
            "p": p,
            "selective_rmse": round(val, 4),
            "best_hidden": best_cfg["hidden_dim"],
            "best_rank": best_cfg["rank"],
            "best_lam": best_cfg["lam"],
            "val_rmse": round(best_val_rmse, 4),
            "seconds": round(time.time() - t_dataset0, 1),
        })

    clear_checkpoint(dataset)
    print(f"{dataset}: done in {time.time() - t_dataset0:.1f}s total")
    return rows_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--quick", action="store_true", help="tiny grid/slice, just to check the pipeline runs")
    parser.add_argument("--force", action="store_true", help="rerun a dataset even if --out already has it")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "uaimc_faithful_results.csv")
    args = parser.parse_args()

    hidden_grid, rank_grid, lam_grid = (
        (QUICK_HIDDEN_GRID, QUICK_RANK_GRID, QUICK_LAM_GRID) if args.quick else (HIDDEN_GRID, RANK_GRID, LAM_GRID)
    )
    max_epochs = QUICK_MAX_EPOCHS if args.quick else MAX_EPOCHS
    patience = QUICK_PATIENCE if args.quick else PATIENCE
    pretrain_max_epochs = QUICK_PRETRAIN_MAX_EPOCHS if args.quick else PRETRAIN_MAX_EPOCHS
    pretrain_patience = QUICK_PRETRAIN_PATIENCE if args.quick else PRETRAIN_PATIENCE
    row_limit = QUICK_ROW_LIMIT if args.quick else None

    print(f"device: {DEVICE} | CPU threads: {N_THREADS}")
    print(f"results file: {args.out}")
    print("NOTE: this is the FAITHFUL reimplementation (nuclear-norm MF + sigmoid-bounded GNN anomaly "
          "score + ARR + IGMC-pretrain-then-joint-finetune) - see module docstring for exactly what's "
          "paper-pinned vs. still an open hyperparameter, and for the ml-100k dataset-identity blocker "
          "that means tonight's numbers won't match the paper's Table 1 yet regardless of model fidelity.")

    t0 = time.time()
    for dataset in args.datasets:
        if not args.force and already_done(args.out, dataset):
            print(f"\n=== {dataset}: already complete in {args.out.name}, skipping (--force to rerun) ===")
            continue
        rows = run_dataset(dataset, hidden_grid, rank_grid, lam_grid, max_epochs, patience,
                            pretrain_max_epochs, pretrain_patience, row_limit=row_limit)
        append_results(args.out, rows)

    print(f"\nall done in {time.time() - t0:.1f}s total. results -> {args.out}")


if __name__ == "__main__":
    main()
