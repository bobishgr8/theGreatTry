import csv
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent

DATASETS = ["ml-100k", "ml-1m", "douban", "amazon-games"]
DATASET_LABEL = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "douban": "Douban", "amazon-games": "Am. Games"}
PS = ["0.0", "0.05", "0.1", "0.2"]

# value[(row_label, dataset, p)] = selective_rmse (string)
value = {}


def load(path, method_col_transform=None):
    p = HERE / path
    if not p.exists():
        return
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method_col_transform:
                method = method_col_transform(method)
            ds = row["dataset"]
            pp = row["p"]
            value[(method, ds, pp)] = row["selective_rmse"]


# Ours (U-hat) and UAIMC (U-hat) - single native-uncertainty curve, no rand/val/supp split
load("experiment3_ours_abstention_final.csv", lambda m: "Ours ($\\hat{U}$)")
load("experiment5_uaimc_best_ml100k_recheck.csv", lambda m: "UA-IMC ($\\hat{U}$)")
load("experiment5_uaimc_best_extra.csv", lambda m: "UA-IMC ($\\hat{U}$)")

# IGMC / SoftImpute / AutoRec / Plain MF - each already comes as "Method (rule)" in the method column
RULE_MAP = {"rand": "rand", "val": "val", "supp": "supp"}


def igmc_label(m):
    # "IGMC (rand)" -> "IGMC (rand)"
    return m


load("experiment8_igmc_best_ml100k_recheck.csv", igmc_label)
load("experiment8_igmc_best_extra.csv", igmc_label)
load("experiment7_softimpute_best_final.csv", igmc_label)
load("experiment6_autorec_best_final.csv", igmc_label)
load("experiment1_plain_abstention_final.csv", lambda m: m.replace("Plain MF (numpy, r-only) ", "Plain MF "))

ROW_ORDER = [
    "Ours ($\\hat{U}$)",
    "UA-IMC ($\\hat{U}$)",
    "IGMC (rand)", "IGMC (val)", "IGMC (supp)",
    "SoftImpute (rand)", "SoftImpute (val)", "SoftImpute (supp)",
    "AutoRec (rand)", "AutoRec (val)", "AutoRec (supp)",
    "Plain MF (rand)", "Plain MF (val)", "Plain MF (supp)",
]

# --- console preview ---
print("Coverage check (rows with at least one filled cell):")
for r in ROW_ORDER:
    filled = sum(1 for ds in DATASETS for p in PS if (r, ds, p) in value)
    print(f"  {r:22s} {filled:2d}/16 cells")

# --- LaTeX table ---
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Selective RMSE by method and abstention rate $p$, across datasets}")
lines.append(r"\label{tab:full-abstention}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{l" + "cccc" * len(DATASETS) + "}")
lines.append(r"\toprule")
header1 = " & " + " & ".join(
    r"\multicolumn{4}{c}{" + DATASET_LABEL[ds] + "}" for ds in DATASETS
) + r" \\"
lines.append(header1)
cmid = " ".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(len(DATASETS)))
lines.append(cmid)
header2 = "Method (rule) & " + " & ".join(["0", ".05", ".10", ".20"] * len(DATASETS)) + r" \\"
lines.append(header2)
lines.append(r"\midrule")

prev_group = None
for r in ROW_ORDER:
    group = r.split(" (")[0]
    if prev_group is not None and group != prev_group and prev_group not in r:
        lines.append(r"\addlinespace")
    prev_group = group
    cells = []
    for ds in DATASETS:
        for p in PS:
            v = value.get((r, ds, p))
            cells.append(f"{float(v):.4f}" if v is not None else "--")
    lines.append(f"{r} & " + " & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}%")
lines.append(r"}")
lines.append(r"\end{table}")

out = HERE / "table4_4_full.tex"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {out}")
