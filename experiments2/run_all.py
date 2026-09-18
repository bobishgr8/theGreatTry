"""Runs every experiment back to back, across every dataset each script's
own DATASETS default covers (see each script's module docstring for its own
exclusions - not every script attempts every dataset).

  python run_all.py                    # the real, full overnight run
  python run_all.py --datasets ml-100k # just one dataset, every experiment
  python run_all.py --quick            # tiny grid, smoke-test every pipeline
  python run_all.py --force            # rerun everything even if already in results/

Most arguments pass straight through to every script unchanged (they all
share --datasets/--quick/--force), so this is mostly a thin sequencer - one
process at a time, so the experiments don't fight each other for CPU/RAM,
matching how they were run manually before this script existed. Each script
is still independently resumable on its own (see their docstrings), so
killing this halfway through and rerunning `run_all.py` picks up wherever it
left off. EXCEPTION: --workers is only understood by the four plain-numpy
scripts (1-4, ProcessPoolExecutor-based); passing it while also running
5/6/7/8 (all PyTorch, single-process/GPU) would fail those with an
"unrecognized arguments" error - pass --workers via a separate,
scripts-1-4-only invocation if you need it, rather than through this runner.

Ordered cheapest/fastest to most expensive, not by experiment number, so a
run that gets interrupted overnight still banks the fast wins first:
plain MF / lasso MF / ours / rank-ablation (numpy, minutes each) -> SoftImpute
/ AutoRec (dense-matrix, fast on GPU) -> IGMC (subgraph GNN, hours) -> UA-IMC
(subgraph GNN + IGMC pretrain + local hyperparameter search, the slowest by
far - see run_experiment5_uaimc.py's module docstring).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "run_experiment1_plain.py",
    "run_experiment2_lasso.py",
    "run_experiment3_ours.py",
    "run_experiment4_rank_ablation.py",
    "run_experiment7_softimpute.py",
    "run_experiment6_autorec.py",
    "run_experiment8_igmc.py",
    "run_experiment5_uaimc.py",
]


def main():
    extra_args = sys.argv[1:]
    for script in SCRIPTS:
        print(f"\n##### {script} #####", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script), *extra_args])
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode} - stopping "
                  f"(rerun run_all.py to resume once it's fixed; earlier scripts' "
                  f"results are already saved).", flush=True)
            sys.exit(result.returncode)

    print(f"\n##### all {len(SCRIPTS)} experiments done #####", flush=True)


if __name__ == "__main__":
    main()
