#!/usr/bin/env python3
"""Replication runner for the BFT LiNGAM paper.

Every table and figure is computed from the original bug-report data in
`data/raw/{core,product}/summary.csv`. Nothing is copied from the manuscript.
The values of the first manuscript version are used only in step 08
(`compare_with_original_manuscript`); `check_manuscript.py` checks the current one.

Pipeline (mirrors the original code in exec/lingam_s.py and exec/bn_t.py):

  01  data provenance: input checksums and the filtering funnel
  02  preprocessing (lingam_s.process + check_train filters) and Table I
  03  LiNGAM bootstrap total causal effects -> Figures 5 and 6 (heatmaps)
      (stored original bootstrap outputs by default; `--retrain-lingam`
      re-runs the 100-sample bootstrap from the raw data)
  04  Table III: strength/direction codes from the Figure 5/6 values
  05  RQ3: RF / LR / BN / LiNGAM-BN on 4 BFT classes -> Tables IV-VI
  06  RQ4: binary classification at the median BFT (7 days) -> Table VII
  07  RQ4: sensitivity to the binary threshold and the outlier filter -> Table VIII
  08  original protocol (exec/bn_t.py) behind the first manuscript version, and a
      comparison of its outputs with that version's values

Steps 05-07 train every model, including the LiNGAM structure used by LiNGAM-BN,
on the training split only and do not use working_time as a predictor.

Usage:
  python replicate.py all [--retrain-lingam] [--n-sampling 100]
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STORED_LINGAM_DIR = DATA_DIR / "lingam"
# LiNGAM bootstrap learned on the training split only (prediction steps 05-07)
TRAIN_LINGAM_DIR = DATA_DIR / "lingam_train"
PAPER_HEATMAP_DIR = DATA_DIR / "paper_result"
DEFAULT_OUTPUT = PACKAGE_DIR / "output"

DATA_TYPES = ["core", "product"]
SCOPE = {"core": "Core", "product": "Product"}
CLASSES = [1, 2, 3, 4]
CLASS_LABELS = {1: "Class 1 (<=1d)", 2: "Class 2 (2-7d)", 3: "Class 3 (8-30d)", 4: "Class 4 (>=31d)"}

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 100
# RQ4 binary task: short <= 7 days (the median BFT of both scopes, and the boundary
# between BFT Classes 2 and 3), long > 7 days.
BINARY_THRESHOLD = 7
# Thresholds of the first manuscript version (exec/lingam_s.py process(data_type=...));
# used only by the original protocol (step 08) and the stored 2-class LiNGAM outputs.
BINARY_THRESHOLDS_ORIGINAL = {"core": 9, "product": 4}
# Step 07: binary thresholds (days) of the sensitivity analysis
SENSITIVITY_THRESHOLDS = [1, 2, 3, 4, 5, 7, 9, 14, 30]
# exec/bn_t.py get_lingam_edge(): LiNGAM edges with bootstrap probability >= 0.8
LINGAM_PROB_THRESHOLD = 0.8
MIN_EFFECT = 0.01
BDEU_ESS = 5.0
N_FOLDS = 10
N_PERMUTATION = 1000

NAME_4CLASS = "user_not_newcomer_filter_wtf_fixed"
NAME_2CLASS = "user_not_newcomer_filter_2_wtf_fixed"

# exec/lingam_s.py `columns` (order matters for DirectLiNGAM)
LINGAM_COLUMNS = [
    "severity", "priority", "system_workloads", "developer_workloads",
    "num_blocks", "num_depends", "num_duplicates", "reporter_reputations",
    "num_creator_previous_works", "assigned_days", "working_time", "fixing_time",
    "discussion_days", "comment_count", "number_of_attachments", "position",
    "discussion_days_class", "fixing_time_class", "assigned_days_class", "working_time_class",
]

# Columns of the training-split LiNGAM used for prediction: BFT enters as the continuous
# fixing_time, so the learned structure does not depend on how BFT is binned.
PRED_LINGAM_COLUMNS = [c for c in LINGAM_COLUMNS if c != "fixing_time_class"]

# Predictors for RQ3/RQ4. working_time (creation -> resolution) always contains
# fixing_time, so it and its class are excluded to avoid target leakage.
EXCLUDE_WORKING_TIME = True
PREDICTORS = [
    "severity", "priority", "system_workloads", "developer_workloads",
    "num_blocks", "num_depends", "num_duplicates", "reporter_reputations",
    "num_creator_previous_works", "assigned_days", "working_time",
    "discussion_days", "comment_count", "number_of_attachments", "position",
]
LEAKY = {"working_time", "working_time_class", "fixing_time", "fixing_time_class"}

# Row labels used in the paper's Figures 5/6 (data/paper_result/heatmap_*.csv)
FACTOR_LABELS = {
    "assigned_days": "Assigned Days",
    "assigned_days_class": "Assigned Days(Class)",
    "comment_count": "Comments",
    "developer_workloads": "Developer Workloads",
    "discussion_days": "Discussion Days",
    "discussion_days_class": "Discussion Days(Class)",
    "fixing_time_class": "BFT(Class)",
    "num_blocks": "Num Blocks",
    "num_creator_previous_works": "Num Developer Prev. Works",
    "num_depends": "Num Depends",
    "num_duplicates": "Num Duplicates",
    "number_of_attachments": "Number of Attachment",
    "position": "Position",
    "priority": "Priority",
    "reporter_reputations": "Reporter Reputations",
    "severity": "Severity",
    "system_workloads": "System Workloads",
    "working_time": "Total Resolution Time",
    "working_time_class": "Total Resolution Time(Class)",
}

# Order of Table III (the first element is not used; the direction is read from the edges)
TABLE3_FACTORS = [
    ("Short Resolution", "comment_count", "Comments"),
    ("Short Resolution", "discussion_days", "Discussion Days"),
    ("Short Resolution", "number_of_attachments", "Number of Attachments"),
    ("Short Resolution", "position", "Position"),
    ("Long Resolution", "assigned_days", "Assigned Days"),
    ("Long Resolution", "num_blocks", "Num Blocks"),
    ("Long Resolution", "num_depends", "Num Depends"),
    ("Long Resolution", "num_duplicates", "Num Duplicates"),
    ("Long Resolution", "priority", "Priority"),
    ("Long Resolution", "reporter_reputations", "Reporter Reputations"),
    ("Long Resolution", "severity", "Severity"),
]

MODELS_4 = ["RF", "LR", "BN", "LiNGAM-BN", "Refined LiNGAM-BN"]
MODELS_2 = ["LR", "RF", "BN", "LiNGAM+LR", "LiNGAM+RF", "LiNGAM+BN", "Refined LiNGAM-BN"]
REFINED_MODELS = {"Refined LiNGAM-BN"}
# models of the first manuscript version (original protocol, step 08)
MODELS_4_ORIGINAL = ["RF", "LR", "BN", "LiNGAM-BN"]
MODELS_2_ORIGINAL = ["LR", "RF", "BN", "LiNGAM+LR", "LiNGAM+RF", "LiNGAM+BN"]

# Values printed in the first manuscript version (as revised on 2026-10-01, before the evaluation
# was corrected). Used ONLY by step_08 to show that the original protocol reproduces them;
# no output table is built from them. check_manuscript.py checks the current manuscript.
PAPER = {
    "table1_total": {("Product", "Firefox"): 1073, ("Product", "Firefox for iOS"): 106,
                     ("Product", "SeaMonkey"): 1860, ("Product", "Thunderbird"): 417,
                     ("Product", "Total"): 3456, ("Core", "Total"): 9590, ("Total", ""): 13046},
    "table3": {
        "Comments": ["S+", "W+", "W+", "W-"], "Discussion Days": ["M-/S-", "M-", "W-", "W-"],
        "Number of Attachments": ["W+", "N/W+", "W+", "N"], "Position": ["W+", "W+/N", "N", "N"],
        "Assigned Days": ["N", "N", "N/W+", "W+"], "Num Depends": ["W-/N", "W+", "W+", "W-/M+"],
        "Num Duplicates": ["N", "W+/N", "W+/N", "M+/W+"], "Priority": ["W+/N", "N", "W-/W+", "S-/W-"],
        "Reporter Reputations": ["W-/W+", "W+/W-", "M-/W+", "S-/M-"], "Severity": ["W-/N", "N", "W-", "S-"],
    },
    "f1": {
        "product": {'RF': [0.79, 0.66, 0.72, 0.86], 'LR': [0.78, 0.65, 0.77, 0.88], 'BN': [0.8, 0.7, 0.76, 0.87], 'LiNGAM-BN': [0.84, 0.74, 0.79, 0.88]},
        "core": {'RF': [0.76, 0.69, 0.74, 0.83], 'LR': [0.75, 0.72, 0.74, 0.85], 'BN': [0.77, 0.75, 0.77, 0.83], 'LiNGAM-BN': [0.84, 0.8, 0.84, 0.89]},
    },
    "precision": {
        "product": {'RF': [0.87, 0.67, 0.69, 0.81], 'LR': [0.88, 0.64, 0.8, 0.78], 'BN': [1.0, 0.65, 0.81, 0.78], 'LiNGAM-BN': [0.99, 0.76, 0.78, 0.78]},
        "core": {'RF': [0.83, 0.69, 0.74, 0.77], 'LR': [0.94, 0.67, 0.72, 0.81], 'BN': [1.0, 0.78, 0.74, 0.71], 'LiNGAM-BN': [0.94, 0.79, 0.81, 0.85]},
    },
    "recall": {
        "product": {'RF': [0.73, 0.64, 0.76, 0.91], 'LR': [0.69, 0.67, 0.73, 0.99], 'BN': [0.67, 0.76, 0.72, 1.0], 'LiNGAM-BN': [0.73, 0.73, 0.81, 1.0]},
        "core": {'RF': [0.7, 0.69, 0.74, 0.91], 'LR': [0.62, 0.78, 0.77, 0.9], 'BN': [0.62, 0.73, 0.8, 1.0], 'LiNGAM-BN': [0.76, 0.82, 0.88, 0.92]},
    },
    "macro_f1": {"core": {"RF": 0.754, "LR": 0.766, "BN": 0.778, "LiNGAM-BN": 0.842}, "product": {"RF": 0.757, "LR": 0.768, "BN": 0.785, "LiNGAM-BN": 0.812}},
    "table8": {"product": {"Count": 144, "Mean": 0.424, "Max": 2.0}, "core": {"Count": 385, "Mean": 1.564, "Max": 112.0}},
    "table9": {"core": {"LR": 0.861, "RF": 0.845, "BN": 0.861, "LiNGAM+LR": 0.861, "LiNGAM+RF": 0.837, "LiNGAM+BN": 0.912, "LiNGAM+BN(Multi-class)": 0.84}, "product": {"LR": 0.879, "RF": 0.879, "BN": 0.882, "LiNGAM+LR": 0.876, "LiNGAM+RF": 0.873, "LiNGAM+BN": 0.936, "LiNGAM+BN(Multi-class)": 0.816}},
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def ensure_dirs(output: Path) -> tuple[Path, Path]:
    tables, figures = Path(output) / "tables", Path(output) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    str_df = df.map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(h) for h in str_df.columns]
    rows = str_df.values.tolist()
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    fmt = lambda vals: "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"
    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"] + [fmt(r) for r in rows])


def write_table(df: pd.DataFrame, tables: Path, name: str) -> pd.DataFrame:
    df.to_csv(tables / f"{name}.csv", index=False)
    (tables / f"{name}.md").write_text(dataframe_to_markdown(df) + "\n")
    print(f"[table] {tables / name}.csv")
    return df


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 02 Preprocessing (exec/lingam_s.py process(filter=True) + check_train)
# ---------------------------------------------------------------------------
def load_raw(data_type: str) -> pd.DataFrame:
    path = RAW_DIR / data_type / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing original data: {path}")
    return pd.read_csv(path, low_memory=False)


def _class4(s: pd.Series) -> np.ndarray:
    return np.select([s <= 1, s <= 7, s <= 30, s > 30], [1, 2, 3, 4])


def preprocess(data_type: str, n_classes: int = 4, return_funnel: bool = False,
               threshold: float | None = None, remove_outliers: bool = True):
    """Reproduces lingam_s.process(df, data_type, filter=True) followed by the
    check_train filters (FIXED, not self-reported, not newcomer developer).

    n_classes=2 gives the binary task, short <= `threshold` days (default
    BINARY_THRESHOLD). `remove_outliers=False` skips the IQR filter (step 07)."""
    df = load_raw(data_type)
    funnel = [("summary.csv (all reports)", len(df))]
    df["priority"] = df["priority"].map({"P1": 2, "P2": 1, "P3": 1, "P4": 0, "P5": 0, "--": 0})
    df["severity"] = df["severity"].map(
        {"critical": 2, "blocker": 2, "major": 1, "trivial": 0, "minor": 0, "--": 0})
    if n_classes == 4:
        df["fixing_time_class"] = _class4(df["fixing_time"])
    else:
        cut = BINARY_THRESHOLD if threshold is None else threshold
        df["fixing_time_class"] = np.where(df["fixing_time"] <= cut, 1, 2)
    # 4-class BFT, used to stratify the train/test split so that every task uses the same split
    df["bft_class4"] = _class4(df["fixing_time"])
    for col in ["fixing_time", "discussion_days", "working_time", "position", "assigned_days"]:
        df[col] = df[col].round(2)
    for col in ["discussion_days", "assigned_days", "working_time"]:
        df[f"{col}_class"] = _class4(df[col])

    if remove_outliers:
        q1, q3 = df["working_time"].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df["working_time"] >= q1 - 1.5 * iqr) & (df["working_time"] <= q3 + 1.5 * iqr)]
        funnel.append(("IQR filter on working_time", len(df)))
    df = df[df["severity_priority_changed"] == 1]
    funnel.append(("severity/priority changed", len(df)))
    df = df[df["resolution"] == "FIXED"]
    funnel.append(("resolution FIXED", len(df)))
    df = df[(df["is_self_reported"] == 0) & (df["is_newcomer_developer"] == 0)]
    funnel.append(("not self-reported, not newcomer developer", len(df)))
    df = df.reset_index(drop=True)
    return (df, funnel) if return_funnel else df


def step_01_data_collection(tables: Path) -> pd.DataFrame:
    rows = []
    for data_type in DATA_TYPES:
        path = RAW_DIR / data_type / "summary.csv"
        _, funnel = preprocess(data_type, return_funnel=True)
        for stage, n in funnel:
            rows.append({"Scope": SCOPE[data_type], "Stage": stage, "Reports": n})
        rows.append({"Scope": SCOPE[data_type], "Stage": f"sha256 {path.relative_to(PACKAGE_DIR)}",
                     "Reports": sha256(path)[:16]})
    write_table(pd.DataFrame(rows), tables, "01_data_collection")
    step_01_figure2(tables)
    return pd.DataFrame(rows)


def step_01_figure2(tables: Path) -> pd.DataFrame:
    """Counts of Fig. 2 and the Data section: extracted reports per project and the
    reports left after the FIXED filter alone (before the analysis filters)."""
    rows = []
    for data_type in DATA_TYPES:
        raw = load_raw(data_type)
        for project, g in raw.groupby("product"):
            rows.append({"Scope": SCOPE[data_type], "Project": project, "Extracted": len(g),
                         "FIXED": int((g["resolution"] == "FIXED").sum())})
        rows.append({"Scope": SCOPE[data_type], "Project": "Total", "Extracted": len(raw),
                     "FIXED": int((raw["resolution"] == "FIXED").sum())})
    tot = pd.DataFrame([r for r in rows if r["Project"] == "Total"])
    rows.append({"Scope": "All", "Project": "Total", "Extracted": int(tot["Extracted"].sum()),
                 "FIXED": int(tot["FIXED"].sum())})
    return write_table(pd.DataFrame(rows), tables, "figure_2_data_overview")


def step_02_table1(tables: Path) -> pd.DataFrame:
    frames = {d: preprocess(d) for d in DATA_TYPES}
    groups = [("Product", p, g) for p, g in frames["product"].groupby("product")]
    groups += [("Product", "Total", frames["product"]), ("Core", "Total", frames["core"]),
               ("Total", "", pd.concat(frames.values()))]
    rows = []
    for bug_type, project, g in groups:
        counts = g["fixing_time_class"].value_counts().reindex(CLASSES, fill_value=0)
        row = {"Bug Type": bug_type, "Project": project}
        for c in CLASSES:
            row[CLASS_LABELS[c]] = f"{counts[c]:,} ({100 * counts[c] / len(g):.2f}%)"
        row["Total"] = len(g)
        rows.append(row)
    return write_table(pd.DataFrame(rows), tables, "table_I_bft_distribution")


# ---------------------------------------------------------------------------
# 03 LiNGAM (exec/lingam_s.py bootstrap_train / evaluate_tce)
# ---------------------------------------------------------------------------
def bootstrap_lingam(X: pd.DataFrame, n_sampling: int = N_BOOTSTRAP,
                     columns: list[str] = LINGAM_COLUMNS) -> pd.DataFrame:
    import lingam

    model = lingam.DirectLiNGAM(random_state=RANDOM_SEED)
    result = model.bootstrap(X[columns].astype(float), n_sampling=n_sampling)
    df = pd.DataFrame(result.get_total_causal_effects())
    df["from"] = df["from"].map(lambda i: columns[int(i)])
    df["to"] = df["to"].map(lambda i: columns[int(i)])
    best = {}
    for _, row in df.iterrows():  # keep the more probable direction of each pair
        pair = tuple(sorted([row["from"], row["to"]]))
        if pair not in best or row["probability"] > best[pair]["probability"]:
            best[pair] = row
    return pd.DataFrame(best.values()).reset_index(drop=True)


def retrain_lingam(out_dir: Path, n_sampling: int = N_BOOTSTRAP) -> Path:
    """Re-run the bootstrap LiNGAM of exec/lingam_s.py check_train (4-class and 2-class)."""
    for data_type in DATA_TYPES:
        target = Path(out_dir) / data_type
        target.mkdir(parents=True, exist_ok=True)
        for n_classes, name in [(4, NAME_4CLASS), (2, NAME_2CLASS)]:
            df = preprocess(data_type, n_classes, threshold=BINARY_THRESHOLDS_ORIGINAL[data_type])
            print(f"[lingam] {data_type} {n_classes}-class all (n={len(df):,}, n_sampling={n_sampling})")
            bootstrap_lingam(df, n_sampling).to_csv(target / f"lingam_tce_{name}.csv", index=False)
            for cls in sorted(df["fixing_time_class"].unique()):
                sub = df[df["fixing_time_class"] == cls]
                print(f"[lingam] {data_type} {n_classes}-class class={cls} (n={len(sub):,})")
                bootstrap_lingam(sub, n_sampling).to_csv(target / f"lingam_tce_{name}_class={cls}.csv", index=False)
    return Path(out_dir)


def train_split_lingam(data_type: str, remove_outliers: bool = True, retrain: bool = False,
                       n_sampling: int = N_BOOTSTRAP) -> pd.DataFrame:
    """LiNGAM bootstrap on the training split only, for the prediction steps 05-07.

    BFT enters as the continuous fixing_time (PRED_LINGAM_COLUMNS), so the same
    structure serves the 4-class task and every binary threshold. Stored in
    data/lingam_train/; recomputed when missing or with `retrain`."""
    path = TRAIN_LINGAM_DIR / f"lingam_tce_train_{data_type}_{'iqr' if remove_outliers else 'no_iqr'}.csv"
    if retrain or not path.exists():
        train, _ = split(preprocess(data_type, 4, remove_outliers=remove_outliers))
        print(f"[lingam] {data_type} training split (outliers {'removed' if remove_outliers else 'kept'}, "
              f"n={len(train):,}, n_sampling={n_sampling})")
        path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_lingam(train, n_sampling, PRED_LINGAM_COLUMNS).to_csv(path, index=False)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Refined LiNGAM-BN: direct LiNGAM edges with temporal prior knowledge
# ---------------------------------------------------------------------------
REFINED_LINGAM_DIR = DATA_DIR / "lingam_refined"
# total resolution time contains BFT, so it is left out of the refined causal model
REFINED_LINGAM_COLUMNS = [c for c in PRED_LINGAM_COLUMNS if c not in ("working_time", "working_time_class")]
# variables fixed before the fix starts: BFT cannot cause them
PRE_FIX_VARIABLES = ["assigned_days", "assigned_days_class", "discussion_days", "discussion_days_class",
                     "reporter_reputations", "num_creator_previous_works", "developer_workloads",
                     "system_workloads"]
# Configuration chosen by refine_lingam_bn.py: highest mean 5-fold CV macro-F1 on the training
# splits over the four scope/task combinations; among tied configurations, the sparsest graph
# (p = 0.8) and the ESS of the standard BN.
REFINED_P, REFINED_CAP, REFINED_ESS = 0.8, 3, BDEU_ESS


def bootstrap_direct_lingam(X: pd.DataFrame, n_sampling: int = N_BOOTSTRAP, prior: bool = True,
                            columns: list[str] | None = None) -> pd.DataFrame:
    """DirectLiNGAM bootstrap returning DIRECT edges: probability that the edge appears and
    the median non-zero coefficient. With `prior`, fixing_time cannot cause PRE_FIX_VARIABLES."""
    import lingam

    cols = columns or REFINED_LINGAM_COLUMNS
    pk = None
    if prior:
        pk = -np.ones((len(cols), len(cols)), dtype=int)
        ft = cols.index("fixing_time")
        for v in PRE_FIX_VARIABLES:
            if v in cols:
                pk[cols.index(v), ft] = 0
    res = lingam.DirectLiNGAM(random_state=RANDOM_SEED, prior_knowledge=pk).bootstrap(
        X[cols].astype(float), n_sampling=n_sampling)
    prob = res.get_probabilities(min_causal_effect=MIN_EFFECT)  # prob[i, j]: edge j -> i
    adj = np.array(res.adjacency_matrices_)
    rows = []
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and prob[i, j] > 0:
                vals = adj[:, i, j][np.abs(adj[:, i, j]) > 0]
                rows.append({"from": cols[j], "to": cols[i], "probability": float(prob[i, j]),
                             "effect": float(np.median(vals)) if len(vals) else 0.0})
    return pd.DataFrame(rows)


def refined_lingam(key: str, train: pd.DataFrame | None = None, retrain: bool = False,
                   n_sampling: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Stored direct-edge LiNGAM (data/lingam_refined/<key>.csv); computed from `train` when
    missing or with `retrain`."""
    path = REFINED_LINGAM_DIR / f"{key}.csv"
    if retrain or not path.exists():
        if train is None:
            raise FileNotFoundError(path)
        print(f"[lingam] refined {key} (n={len(train):,}, n_sampling={n_sampling})")
        path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_direct_lingam(train, n_sampling).to_csv(path, index=False)
    return pd.read_csv(path)


def refined_edges(tab: pd.DataFrame, nodes: list[str], p: float = REFINED_P,
                  cap: int | None = REFINED_CAP) -> list[tuple[str, str]]:
    """Refined LiNGAM-BN structure: direct edges with probability >= p (the more probable
    direction of each pair), cycles broken at the least probable edge, and at most `cap`
    parents per node (the most probable are kept)."""
    to_node = lambda v: BN_TARGET if v in ("fixing_time", "fixing_time_class") else v
    best = {}
    for _, e in tab[(tab["probability"] >= p) & (tab["effect"].abs() >= MIN_EFFECT)].iterrows():
        a, b = to_node(e["from"]), to_node(e["to"])
        if a == b or a not in nodes or b not in nodes:
            continue
        key = tuple(sorted((a, b)))
        if key not in best or e["probability"] > best[key][2]:
            best[key] = (a, b, float(e["probability"]), abs(float(e["effect"])))
    g = nx.DiGraph()
    for a, b, pr, w in best.values():
        g.add_edge(a, b, p=pr, w=w)
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        g.remove_edge(*min(cycle, key=lambda x: (g.edges[x[0], x[1]]["p"], g.edges[x[0], x[1]]["w"]))[:2])
    if cap:
        for n in list(g.nodes):
            ps = sorted(g.predecessors(n), key=lambda u: (-g.edges[u, n]["p"], -g.edges[u, n]["w"]))
            for u in ps[cap:]:
                g.remove_edge(u, n)
    return sorted(g.edges())


def temporal_split(df: pd.DataFrame, share: float = 1 - TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest `share` of the reports (by creation time) for training, the newest for testing."""
    order = pd.to_datetime(df["creation_time"]).sort_values(kind="mergesort").index
    cut = int(round(len(order) * share))
    return df.loc[order[:cut]], df.loc[order[cut:]]


def refined_lingam_inputs() -> dict:
    """Training data behind every stored refined LiNGAM (data/lingam_refined/)."""
    out = {}
    for d in DATA_TYPES:
        for remove in (True, False):
            out[f"train_{d}_{'iqr' if remove else 'no_iqr'}"] = lambda d=d, r=remove: split(preprocess(d, 4, remove_outliers=r))[0]
        out[f"all_{d}"] = lambda d=d: preprocess(d, 4)
        out[f"temporal_{d}"] = lambda d=d: temporal_split(preprocess(d, 4))[0]
    return out


def load_tce(lingam_dir: Path, data_type: str, name: str, cls: int | None = None) -> pd.DataFrame:
    suffix = f"_class={cls}" if cls is not None else ""
    return pd.read_csv(Path(lingam_dir) / data_type / f"lingam_tce_{name}{suffix}.csv")


def effects_on_bft(tce: pd.DataFrame, target: str = "fixing_time") -> pd.Series:
    """lingam_s.evaluate_tce: signed effect of each factor on fixing_time
    (edge factor->fixing_time, or fixing_time->factor), |effect| >= 0.01."""
    rows = tce[((tce["to"] == target) & (tce["from"] != target)) |
               ((tce["from"] == target) & (tce["to"] != target))]
    rows = rows[rows["effect"].abs() >= MIN_EFFECT]
    factor = np.where(rows["to"] == target, rows["from"], rows["to"])
    return pd.Series(rows["effect"].values, index=factor)


def effect_edges(tce: pd.DataFrame, target: str = "fixing_time") -> pd.DataFrame:
    """The edges behind effects_on_bft, with their direction and bootstrap probability.
    'Factor -> BFT' is the effect of the factor on fixing_time; 'BFT -> Factor' is the
    effect of fixing_time on the factor (LiNGAM oriented the edge away from BFT)."""
    rows = tce[((tce["to"] == target) & (tce["from"] != target)) |
               ((tce["from"] == target) & (tce["to"] != target))]
    rows = rows[rows["effect"].abs() >= MIN_EFFECT]
    to_bft = rows["to"] == target
    return pd.DataFrame({"Factor": np.where(to_bft, rows["from"], rows["to"]),
                         "Direction": np.where(to_bft, "Factor -> BFT", "BFT -> Factor"),
                         "Effect": rows["effect"].values, "Probability": rows["probability"].values})


def causal_edge_table(lingam_dir: Path, data_type: str) -> pd.DataFrame:
    parts = []
    for cls in [None] + CLASSES:
        e = effect_edges(load_tce(lingam_dir, data_type, NAME_4CLASS, cls))
        e.insert(0, "Class", "all" if cls is None else str(cls))
        parts.append(e)
    out = pd.concat(parts, ignore_index=True)
    out["Factor"] = out["Factor"].map(lambda v: FACTOR_LABELS.get(v, v))
    return out


def causal_strength_matrix(lingam_dir: Path, data_type: str) -> pd.DataFrame:
    cols = {"all": effects_on_bft(load_tce(lingam_dir, data_type, NAME_4CLASS))}
    for cls in CLASSES:
        cols[str(cls)] = effects_on_bft(load_tce(lingam_dir, data_type, NAME_4CLASS, cls))
    mat = pd.DataFrame(cols).sort_index()
    mat.index = [FACTOR_LABELS.get(i, i) for i in mat.index]
    return mat


def step_03_heatmaps(tables: Path, figures: Path, lingam_dir: Path = STORED_LINGAM_DIR) -> dict:
    out = {}
    for data_type, fig in [("core", "figure_5_causal_strength_core"),
                           ("product", "figure_6_causal_strength_product")]:
        mat = causal_strength_matrix(lingam_dir, data_type)
        out[data_type] = mat
        write_table(mat.round(4).reset_index(names="Factor"), tables, f"{fig}_data")
        edges = causal_edge_table(lingam_dir, data_type)
        write_table(edges.round({"Effect": 4, "Probability": 2}), tables, f"{fig}_edges")
        fig_, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt=".2g", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Causal Effect of Factors on BFT ({data_type})")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Factors")
        fig_.tight_layout()
        fig_.savefig(figures / f"{fig}.png", dpi=100)
        plt.close(fig_)
        print(f"[figure] {figures / fig}.png")
    return out


# ---------------------------------------------------------------------------
# 04 Table III
# ---------------------------------------------------------------------------
def strength_code(value) -> str:
    """Paper rule: S |v|>5, M 1<=|v|<=5, W 0<|v|<1, N none; sign = direction."""
    if value is None or not np.isfinite(value) or value == 0:
        return "N"
    level = "S" if abs(value) > 5 else ("M" if abs(value) >= 1 else "W")
    return level + ("+" if value > 0 else "-")


LOW_PROBABILITY = 0.5  # Table III marks effects present in fewer than half of the bootstrap samples


def step_04_table3(tables: Path, lingam_dir: Path = STORED_LINGAM_DIR) -> pd.DataFrame:
    """Strength codes per factor and class; a dagger marks bootstrap probability < 0.5.
    'Direction' is the orientation of the factor's edges in Classes 1-4."""
    edges = {d: causal_edge_table(lingam_dir, d) for d in DATA_TYPES}
    rows = []
    for related, var, label in TABLE3_FACTORS:
        lab = FACTOR_LABELS[var]
        dirs = set(pd.concat([e[(e["Factor"] == lab) & (e["Class"] != "all")]["Direction"] for e in edges.values()]))
        row = {"Direction": "/".join(sorted(dirs)), "Factor": label}
        for cls in CLASSES:
            codes = {}
            for d in DATA_TYPES:
                hit = edges[d][(edges[d]["Factor"] == lab) & (edges[d]["Class"] == str(cls))]
                if hit.empty:
                    codes[d] = "N"
                else:
                    codes[d] = strength_code(hit["Effect"].iloc[0]) + \
                        ("\u2020" if hit["Probability"].iloc[0] < LOW_PROBABILITY else "")
            row[CLASS_LABELS[cls]] = (codes["product"] if codes["product"] == codes["core"]
                                      else f"{codes['product']}/{codes['core']}")
        rows.append(row)
    return write_table(pd.DataFrame(rows), tables, "table_III_strength_direction")


# ---------------------------------------------------------------------------
# Discrete Bayesian network (exec/bn_t.py: bnlearn hc/BDeu, LiNGAM prior DAG)
# ---------------------------------------------------------------------------
BN_TARGET = "bft_class"


def discretize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """exec/bn_t.py process() mappings where defined; train-set quantile bins otherwise."""
    fixed = {
        "reporter_reputations": lambda s: np.select([s <= 0.3, s <= 0.7], [0, 1], 2),
        "comment_count": lambda s: (s > 20).astype(int),
        "number_of_attachments": lambda s: (s > 0).astype(int),
        "num_blocks": lambda s: (s > 0).astype(int),
        "num_depends": lambda s: (s > 0).astype(int),
        "num_duplicates": lambda s: (s > 0).astype(int),
        "system_workloads": lambda s: np.select([s <= 30, s <= 1000], [0, 1], 2),
        "developer_workloads": lambda s: np.select([s <= 10, s <= 20], [0, 1], 2),
        "assigned_days": lambda s: (s >= 1).astype(int),
        "discussion_days": lambda s: (s > 7).astype(int),
        "working_time": lambda s: _class4(s) - 1,
        "severity": lambda s: s.fillna(0).astype(int),
        "priority": lambda s: s.fillna(0).astype(int),
    }
    tr, te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for c in cols:
        if c in fixed:
            tr[c], te[c] = fixed[c](train[c]), fixed[c](test[c])
        elif c.endswith("_class"):
            tr[c], te[c] = train[c].astype(int), test[c].astype(int)
        else:
            inner = np.unique(np.quantile(train[c], [1 / 3, 2 / 3]))
            tr[c], te[c] = np.digitize(train[c], inner), np.digitize(test[c], inner)
    return tr.astype(int), te.astype(int)


class DiscreteBN:
    """Discrete BN with BDeu-smoothed CPTs. Prediction is exact: with every
    other node observed, P(target | evidence) is proportional to
    P(target | parents) * prod over children of P(child | its parents)."""

    def __init__(self, edges: list[tuple[str, str]], target: str = BN_TARGET, ess: float = BDEU_ESS):
        self.graph = nx.DiGraph(edges)
        self.graph.add_node(target)
        self.target, self.ess = target, ess

    def fit(self, data: pd.DataFrame) -> "DiscreteBN":
        self.states = {c: np.sort(data[c].unique()) for c in self.graph.nodes}
        self.train = data[list(self.graph.nodes)]
        return self

    def _log_prob(self, node: str, frame: pd.DataFrame) -> np.ndarray:
        parents = list(self.graph.predecessors(node))
        r = len(self.states[node])
        q = int(np.prod([len(self.states[p]) for p in parents])) if parents else 1
        a_x, a_pa = self.ess / (r * q), self.ess / q
        if parents:
            joint = self.train.groupby(parents + [node]).size()
            marg = self.train.groupby(parents).size()
            n_x = joint.reindex(pd.MultiIndex.from_frame(frame[parents + [node]])).fillna(0).values
            n_pa = marg.reindex(pd.MultiIndex.from_frame(frame[parents]) if len(parents) > 1
                                else frame[parents[0]]).fillna(0).values
        else:
            counts = self.train[node].value_counts()
            n_x = counts.reindex(frame[node]).fillna(0).values
            n_pa = np.full(len(frame), len(self.train))
        return np.log((n_x + a_x) / (n_pa + a_pa))

    def predict(self, evidence: pd.DataFrame) -> np.ndarray:
        blanket = [self.target] + list(self.graph.successors(self.target))
        scores = []
        for t in self.states[self.target]:
            frame = evidence.copy()
            frame[self.target] = t
            scores.append(sum(self._log_prob(n, frame) for n in blanket))
        return self.states[self.target][np.argmax(np.vstack(scores), axis=0)]


def _hill_climb_edges(columns: list[str], values: np.ndarray) -> list[tuple[str, str]]:
    from pgmpy.estimators import BDeuScore, HillClimbSearch

    data = pd.DataFrame(np.asarray(values, dtype=np.int64), columns=list(columns))

    dag = HillClimbSearch(data).estimate(scoring_method=BDeuScore(data, equivalent_sample_size=BDEU_ESS),
                                         show_progress=False)
    return sorted(dag.edges())


# Hill-climbing structures of the standard BN in steps 05-07, learned once and stored
BN_STRUCTURE_DIR = DATA_DIR / "bn_structures"
RELEARN_BN = False


def learn_structure(data: pd.DataFrame, key: str | None = None) -> list[tuple[str, str]]:
    """Standard BN baseline: hill-climbing structure search with BDeu (bn_t.py: hc + bdeu).

    pgmpy breaks ties between equally scored moves in an order that depends on the
    Python process (string-hash seed and runtime environment), so the same data can
    give slightly different structures in different runs. Like the original BNs in
    data/models/, the structures are therefore learned once (in a child process with
    PYTHONHASHSEED=0) and stored in data/bn_structures/<key>.csv; `--relearn-bn`
    learns them again."""
    path = BN_STRUCTURE_DIR / f"{key}.csv" if key else None
    if path is not None and path.exists() and not RELEARN_BN:
        e = pd.read_csv(path)
        return list(zip(e["from"], e["to"]))
    previous = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            edges = pool.submit(_hill_climb_edges, list(data.columns), data.to_numpy(np.int64)).result()
    finally:
        if previous is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(edges, columns=["from", "to"]).to_csv(path, index=False)
    return edges


def ensure_fixed_hash_seed() -> None:
    """Re-run the current script with PYTHONHASHSEED=0 (used by the command-line entry points)."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})


def lingam_edges(tce: pd.DataFrame, nodes: list[str]) -> list[tuple[str, str]]:
    """LiNGAM-BN structure: LiNGAM edges with probability >= 0.8 (bn_t.get_lingam_edge);
    fixing_time / fixing_time_class become the BFT class node. Cycles are broken by
    dropping the least probable edge."""
    to_node = lambda v: BN_TARGET if v in ("fixing_time", "fixing_time_class") else v
    df = tce[(tce["probability"] >= LINGAM_PROB_THRESHOLD) & (tce["effect"].abs() >= MIN_EFFECT)]
    g = nx.DiGraph()
    for _, r in df.iterrows():
        a, b = to_node(r["from"]), to_node(r["to"])
        if a != b and a in nodes and b in nodes:
            g.add_edge(a, b, p=float(r["probability"]))
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        g.remove_edge(*min(cycle, key=lambda e: g.edges[e[0], e[1]]["p"])[:2])
    return list(g.edges())


def lingam_features(tce: pd.DataFrame, features: list[str]) -> list[str]:
    """Factors with a LiNGAM path to/from fixing_time with probability >= 0.8."""
    df = tce[(tce["probability"] >= LINGAM_PROB_THRESHOLD) & (tce["effect"].abs() >= MIN_EFFECT)]
    df = df[(df["to"].isin(["fixing_time", "fixing_time_class"])) | (df["from"].isin(["fixing_time", "fixing_time_class"]))]
    chosen = set(df["from"]) | set(df["to"])
    return [f for f in features if f in chosen] or features


# ---------------------------------------------------------------------------
# 05 RQ3: 4-class prediction
# ---------------------------------------------------------------------------
def predictors() -> list[str]:
    return [f for f in PREDICTORS if not (EXCLUDE_WORKING_TIME and f == "working_time")]


def split(df: pd.DataFrame):
    """Stratified 80/20 split on the 4 BFT classes, identical for the 4-class and binary tasks."""
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["bft_class4"], random_state=RANDOM_SEED)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, tce: pd.DataFrame, models: list[str],
                structure_key: str | None = None, refined_tab: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """Train each model on `train` and predict `test`. `tce`: LiNGAM total effects (LiNGAM-BN,
    LiNGAM+X feature selection); `refined_tab`: direct LiNGAM edges (refined LiNGAM-BN)."""
    feats = predictors()
    y_tr = train["fixing_time_class"].astype(int).values
    scaler = StandardScaler().fit(train[feats])
    Xs_tr = pd.DataFrame(scaler.transform(train[feats]), columns=feats)
    Xs_te = pd.DataFrame(scaler.transform(test[feats]), columns=feats)
    lfeats = lingam_features(tce, feats) if tce is not None else feats

    bn_nodes = feats + [c for c in ["discussion_days_class", "assigned_days_class"] if c not in LEAKY]
    d_tr, d_te = discretize(train, test, bn_nodes)
    d_tr[BN_TARGET] = y_tr

    def bn_predict(edges, ess=BDEU_ESS):
        nodes = sorted({n for e in edges for n in e} | {BN_TARGET})
        bn = DiscreteBN(edges, ess=ess).fit(d_tr[nodes])
        return bn.predict(d_te[[n for n in nodes if n != BN_TARGET]].reset_index(drop=True))

    preds = {}
    for m in models:
        cols = lfeats if m in ("LiNGAM+LR", "LiNGAM+RF") else feats
        if m.endswith("RF"):
            preds[m] = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1) \
                .fit(Xs_tr[cols], y_tr).predict(Xs_te[cols])
        elif m.endswith("LR"):
            preds[m] = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED) \
                .fit(Xs_tr[cols], y_tr).predict(Xs_te[cols])
        elif m == "BN":
            key = structure_key and structure_key + ("" if EXCLUDE_WORKING_TIME else "_with_working_time")
            preds[m] = bn_predict(learn_structure(d_tr[bn_nodes + [BN_TARGET]], key))
        elif m in REFINED_MODELS:
            preds[m] = bn_predict(refined_edges(refined_tab, bn_nodes + [BN_TARGET]), REFINED_ESS)
        else:  # LiNGAM-BN / LiNGAM+BN: total-effect edges
            preds[m] = bn_predict(lingam_edges(tce, bn_nodes + [BN_TARGET]))
    return preds, {"lingam_features": lfeats}


def run_models(df: pd.DataFrame, tce: pd.DataFrame, models: list[str],
               structure_key: str | None = None, refined_tab: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    train, test = split(df)
    preds, info = fit_predict(train, test, tce, models, structure_key, refined_tab)
    out = pd.DataFrame({"id": test["bug_id"].values, "fixing_time": test["fixing_time"].values,
                        "y_true": test["fixing_time_class"].astype(int).values,
                        **{f"pred_{m}": p for m, p in preds.items()}})
    return out, {**info, "n_train": len(train), "n_test": len(test)}


def _f1_batch(y_true, y_pred, cls) -> np.ndarray:
    t, p = y_true == cls, y_pred == cls
    tp, fp, fn = (t & p).sum(1), (~t & p).sum(1), (t & ~p).sum(1)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    b = np.sort(b)
    greater = np.searchsorted(b, a, side="left").sum()
    less = (len(b) - np.searchsorted(b, a, side="right")).sum()
    return float((greater - less) / (len(a) * len(b)))


def effect_vs_rf(y_true, y_rf, y_m, cls) -> tuple[float, float]:
    """Cliff's delta between per-fold class-F1 of the model and RF (10 stratified
    folds of the test set) and a paired permutation p-value (1,000 permutations)."""
    rng = np.random.default_rng(RANDOM_SEED)
    fm, fr = [], []
    for _, idx in StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(y_true, y_true):
        fm.append(_f1_batch(y_true[idx][None], y_m[idx][None], cls)[0])
        fr.append(_f1_batch(y_true[idx][None], y_rf[idx][None], cls)[0])
    obs = _f1_batch(y_true[None], y_m[None], cls)[0] - _f1_batch(y_true[None], y_rf[None], cls)[0]
    swap = rng.random((N_PERMUTATION, len(y_true))) < 0.5
    yt = np.broadcast_to(y_true, swap.shape)
    diff = _f1_batch(yt, np.where(swap, y_rf, y_m), cls) - _f1_batch(yt, np.where(swap, y_m, y_rf), cls)
    p = (np.sum(np.abs(diff) >= abs(obs) - 1e-12) + 1) / (N_PERMUTATION + 1)
    return cliffs_delta(np.array(fm), np.array(fr)), float(p)


def step_05_prediction_tables(tables: Path) -> dict:
    results = {}
    for data_type in DATA_TYPES:
        df = preprocess(data_type, 4)
        preds, info = run_models(df, train_split_lingam(data_type), MODELS_4, f"{data_type}_4class_iqr",
                                 refined_lingam(f"train_{data_type}_iqr"))
        preds.to_csv(tables / f"predictions_4class_{data_type}.csv", index=False)
        print(f"[rq3] {data_type}: train={info['n_train']:,} test={info['n_test']:,}")
        results[data_type] = preds

    for data_type, name in [("product", "table_IV_performance_product"), ("core", "table_IV_performance_core")]:
        p, rows = results[data_type], []
        for m in MODELS_4:
            rep = classification_report(p["y_true"], p[f"pred_{m}"], labels=CLASSES, output_dict=True, zero_division=0)
            for metric, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
                rows.append({"Model": m, "Metric": metric, **{CLASS_LABELS[c]: round(rep[str(c)][key], 3) for c in CLASSES}})
        write_table(pd.DataFrame(rows), tables, name)

    write_table(pd.DataFrame([{
        "Model": m,
        **{SCOPE[d]: round(f1_score(results[d]["y_true"], results[d][f"pred_{m}"], labels=CLASSES,
                                    average="macro", zero_division=0), 3) for d in DATA_TYPES},
    } for m in MODELS_4]), tables, "table_V_macro_f1")

    rows = []
    for cls in CLASSES:
        for m in MODELS_4:
            row = {"Class": CLASS_LABELS[cls], "Model": m}
            for d in DATA_TYPES:
                p = results[d]
                yt, yr, ym = (p["y_true"].values, p["pred_RF"].values, p[f"pred_{m}"].values)
                row[f"{SCOPE[d]} F1"] = round(float(_f1_batch(yt[None], ym[None], cls)[0]), 3)
                if m == "RF":
                    row[f"{SCOPE[d]} d/p"] = "--"
                else:
                    delta, pv = effect_vs_rf(yt, yr, ym, cls)
                    row[f"{SCOPE[d]} d/p"] = f"{delta:.2f}/{pv:.3f}"
            rows.append(row)
    write_table(pd.DataFrame(rows), tables, "table_VI_effect_size_f1")
    return results


# ---------------------------------------------------------------------------
# 06 RQ4: binary classification at the median BFT
# ---------------------------------------------------------------------------
def _binary_rows(preds: pd.DataFrame, models: list[str]) -> dict:
    row = {}
    for m in models:
        row[f"{m} accuracy"] = accuracy_score(preds["y_true"], preds[f"pred_{m}"])
        row[f"{m} macro-F1"] = f1_score(preds["y_true"], preds[f"pred_{m}"], average="macro")
    return row


def step_06_binary_tables(tables: Path) -> pd.DataFrame:
    """Short (<= BINARY_THRESHOLD days) vs long; same split and LiNGAM structure as step 05."""
    out = []
    for d in DATA_TYPES:
        df = preprocess(d, 2)
        tce = train_split_lingam(d)
        preds, info = run_models(df, tce, MODELS_2, f"{d}_binary{BINARY_THRESHOLD}_iqr", refined_lingam(f"train_{d}_iqr"))
        preds.to_csv(tables / f"predictions_2class_{d}.csv", index=False)
        print(f"[rq4] {d}: threshold <= {BINARY_THRESHOLD}d, LiNGAM features={info['lingam_features']}")
        r = _binary_rows(preds, MODELS_2)
        share = float((preds["y_true"] == 1).mean())
        for m in MODELS_2:
            out.append({"Model": m, "Scope": SCOPE[d], "Accuracy": r[f"{m} accuracy"], "Macro-F1": r[f"{m} macro-F1"]})
        out.append({"Model": "Majority baseline", "Scope": SCOPE[d], "Accuracy": max(share, 1 - share),
                    "Macro-F1": np.nan})
    long = pd.DataFrame(out)
    wide = long.pivot(index="Model", columns="Scope", values=["Accuracy", "Macro-F1"])
    wide.columns = [f"{scope} {metric}" for metric, scope in wide.columns]
    wide = wide.reindex(MODELS_2 + ["Majority baseline"])[
        ["Core Accuracy", "Core Macro-F1", "Product Accuracy", "Product Macro-F1"]].round(3)
    return write_table(wide.reset_index(), tables, "table_VII_binary_classification")


# ---------------------------------------------------------------------------
# 07 RQ4: sensitivity to the binary threshold and the outlier filter
# ---------------------------------------------------------------------------
def step_07_sensitivity(tables: Path) -> pd.DataFrame:
    """Binary task for every threshold in SENSITIVITY_THRESHOLDS, with and without the
    IQR outlier filter. For each scope and filter setting, all thresholds share one
    split and one training-split LiNGAM structure."""
    rows = []
    for d in DATA_TYPES:
        for remove in (True, False):
            df = preprocess(d, 4, remove_outliers=remove)
            tce = train_split_lingam(d, remove_outliers=remove)
            for thr in SENSITIVITY_THRESHOLDS:
                df_t = df.copy()
                df_t["fixing_time_class"] = np.where(df_t["fixing_time"] <= thr, 1, 2)
                preds, info = run_models(df_t, tce, MODELS_4, f"{d}_binary{thr}_{'iqr' if remove else 'no_iqr'}",
                                         refined_lingam(f"train_{d}_{'iqr' if remove else 'no_iqr'}"))
                share = float((preds["y_true"] == 1).mean())
                r = _binary_rows(preds, MODELS_4)
                rows.append({"Scope": SCOPE[d], "Outliers": "removed (IQR)" if remove else "kept",
                             "Threshold (days)": thr, "Reports": info["n_train"] + info["n_test"],
                             "Short share (test)": round(share, 3),
                             "Majority accuracy": round(max(share, 1 - share), 3),
                             **{k: round(v, 3) for k, v in r.items()}})
            print(f"[sensitivity] {d}: outliers {'removed' if remove else 'kept'} done")
    return write_table(pd.DataFrame(rows), tables, "table_VIII_threshold_sensitivity")


# ---------------------------------------------------------------------------
# 09 Transfer experiment (TRANSFER_PROTOCOL.md)
# ---------------------------------------------------------------------------
TRANSFER_SETTINGS = [("Core -> Product", "core", "product"), ("Product -> Core", "product", "core"),
                     ("Core, temporal", "core", "core"), ("Product, temporal", "product", "product")]
TRANSFER_TASKS = ["4class", f"binary{BINARY_THRESHOLD}"]


def transfer_data(setting: str, train_scope: str, test_scope: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if train_scope != test_scope:
        return preprocess(train_scope, 4), preprocess(test_scope, 4), f"all_{train_scope}"
    tr, te = temporal_split(preprocess(train_scope, 4))
    return tr, te, f"temporal_{train_scope}"


def transfer_total_lingam(key: str, train: pd.DataFrame, retrain: bool = False,
                          n_sampling: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Total-effect LiNGAM (as train_split_lingam) on a transfer training set."""
    path = TRAIN_LINGAM_DIR / f"lingam_tce_{key}.csv"
    if retrain or not path.exists():
        print(f"[lingam] total effects {key} (n={len(train):,})")
        path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_lingam(train, n_sampling, PRED_LINGAM_COLUMNS).to_csv(path, index=False)
    return pd.read_csv(path)


def _label(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = df.copy()
    if task != "4class":
        df["fixing_time_class"] = np.where(df["fixing_time"] <= BINARY_THRESHOLD, 1, 2)
    return df


def _in_distribution(tables: Path, scope: str, task: str, model: str) -> float:
    if task == "4class":
        p = pd.read_csv(tables / f"predictions_4class_{scope}.csv")
        col = f"pred_{model}"
    else:
        p = pd.read_csv(tables / f"predictions_2class_{scope}.csv")
        col = f"pred_{'LiNGAM+BN' if model == 'LiNGAM-BN' else model}"
    return f1_score(p["y_true"], p[col], average="macro")


def step_09_transfer(tables: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows, summary = [], []
    for setting, train_scope, test_scope in TRANSFER_SETTINGS:
        tr0, te0, key = transfer_data(setting, train_scope, test_scope)
        tce = transfer_total_lingam(key, tr0)
        refined = refined_lingam(key, tr0)
        for task in TRANSFER_TASKS:
            tr, te = _label(tr0, task), _label(te0, task)
            preds, _ = fit_predict(tr, te, tce, MODELS_4, f"transfer_{key}_{task}", refined)
            y = te["fixing_time_class"].astype(int).values
            for m in MODELS_4:
                ind = _in_distribution(tables, test_scope, task, m)
                tf = f1_score(y, preds[m], average="macro")
                rows.append({"Setting": setting, "Task": task, "Model": m, "In-distribution macro-F1": round(ind, 3),
                             "Transfer macro-F1": round(tf, 3), "Drop": round(ind - tf, 3),
                             "Transfer accuracy": round(accuracy_score(y, preds[m]), 3)})
            # 95% bootstrap CI of the macro-F1 difference refined LiNGAM-BN - RF on the transfer test data
            a, b = preds["Refined LiNGAM-BN"], preds["RF"]
            diffs = []
            for _ in range(N_PERMUTATION):
                i = rng.integers(0, len(y), len(y))
                diffs.append(f1_score(y[i], a[i], average="macro") - f1_score(y[i], b[i], average="macro"))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            r_ref = [x for x in rows if x["Setting"] == setting and x["Task"] == task]
            drop = {x["Model"]: x["Drop"] for x in r_ref}
            summary.append({"Setting": setting, "Task": task,
                            "Drop refined LiNGAM-BN": drop["Refined LiNGAM-BN"], "Drop RF": drop["RF"],
                            "H1: smaller drop than RF": drop["Refined LiNGAM-BN"] < drop["RF"],
                            "Transfer macro-F1 difference (refined LiNGAM-BN - RF)": round(
                                f1_score(y, a, average="macro") - f1_score(y, b, average="macro"), 3),
                            "95% CI low": round(lo, 3), "95% CI high": round(hi, 3),
                            "H2: CI above 0": lo > 0})
            print(f"[transfer] {setting} {task} done")
    write_table(pd.DataFrame(summary), tables, "table_IX_transfer_summary")
    return write_table(pd.DataFrame(rows), tables, "table_IX_transfer")


# ---------------------------------------------------------------------------
# 08 Original protocol: comparison with the first manuscript version
# ---------------------------------------------------------------------------
def compare_with_original_manuscript(tables: Path) -> pd.DataFrame:
    """Put each value of the first manuscript version (PAPER) next to the value computed
    by the original protocol (step 08)."""
    rows = []

    def add(artifact, item, paper, computed):
        if isinstance(paper, str):
            ok = paper == computed
        else:
            tol = 0.5 if float(paper).is_integer() and abs(paper) >= 1 else 0.0051
            ok = computed is not None and abs(float(paper) - float(computed)) <= tol
        rows.append({"Artifact": artifact, "Item": item, "Paper": paper, "Computed": computed, "Match": bool(ok)})

    have = lambda name: (tables / f"{name}.csv").exists()

    if have("table_I_bft_distribution"):
        t1 = pd.read_csv(tables / "table_I_bft_distribution.csv").fillna("")
        for (bt, proj), v in PAPER["table1_total"].items():
            hit = t1[(t1["Bug Type"] == bt) & (t1["Project"] == proj)]
            add("Table I total", f"{bt} {proj}".strip(), v, int(hit["Total"].iloc[0]) if len(hit) else None)

    # Figures 5/6: computed LiNGAM effects vs the values plotted in the paper
    for d, csv in [("core", "heatmap_core.csv"), ("product", "heatmap_product.csv")]:
        fig = f"figure_{5 if d == 'core' else 6}_causal_strength_{d}_data"
        if not (PAPER_HEATMAP_DIR / csv).exists() or not have(fig):
            continue
        paper_mat = pd.read_csv(PAPER_HEATMAP_DIR / csv).set_index("Factor")
        comp = pd.read_csv(tables / f"{fig}.csv").set_index("Factor")
        for factor in paper_mat.index:
            for pc, cc in zip(["All", "Class 1", "Class 2", "Class 3", "Class 4"], ["all", "1", "2", "3", "4"]):
                pv = paper_mat.loc[factor, pc]
                cv = comp.loc[factor, cc] if factor in comp.index else np.nan
                if pd.isna(pv) and pd.isna(cv):
                    continue
                ok = (not pd.isna(pv) and not pd.isna(cv)
                      and abs(pv - cv) <= max(0.051 * abs(pv), 0.0051))
                rows.append({"Artifact": f"Figure {5 if d == 'core' else 6} ({SCOPE[d]})",
                             "Item": f"{factor} {pc}", "Paper": pv, "Computed": round(cv, 4) if not pd.isna(cv) else cv,
                             "Match": bool(ok)})

    if have("table_III_strength_direction"):
        t3 = pd.read_csv(tables / "table_III_strength_direction.csv").set_index("Factor")
        for factor, codes in PAPER["table3"].items():
            for cls, code in zip(CLASSES, codes):
                add("Table III", f"{factor} class {cls}", code, t3.loc[factor, CLASS_LABELS[cls]])

    for d, name in [("product", "table_IV_performance_product"), ("core", "table_IV_performance_core")]:
        if not have(name):
            continue
        t = pd.read_csv(tables / f"{name}.csv")
        for key, metric in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
            for m, vals in PAPER[key][d].items():
                hit = t[(t["Model"] == m) & (t["Metric"] == metric)].iloc[0]
                for cls, v in zip(CLASSES, vals):
                    add(f"Table IV ({SCOPE[d]})", f"{m} {metric} class {cls}", v,
                        hit[CLASS_LABELS[cls]])

    if have("table_V_macro_f1"):
        t6 = pd.read_csv(tables / "table_V_macro_f1.csv").set_index("Model")
        for d, models in PAPER["macro_f1"].items():
            for m, v in models.items():
                if f"{SCOPE[d]} macro-F1" in t6.columns:
                    add("Table V (vs macro-F1)", f"{SCOPE[d]} {m}", v, t6.loc[m, f"{SCOPE[d]} macro-F1"])
                    add("Table V (vs accuracy)", f"{SCOPE[d]} {m}", v, t6.loc[m, f"{SCOPE[d]} accuracy"])
                else:
                    add("Table V (vs macro-F1)", f"{SCOPE[d]} {m}", v, t6.loc[m, SCOPE[d]])

    if have("table_VII_short_duration_bft_stats"):
        t8 = pd.read_csv(tables / "table_VII_short_duration_bft_stats.csv").set_index("Metric")
        for d, items in PAPER["table8"].items():
            for k, v in items.items():
                add("Table VII", f"{SCOPE[d]} {k}", v, t8.loc[k, SCOPE[d]])

    if have("table_VIII_binary_accuracy"):
        t9 = pd.read_csv(tables / "table_VIII_binary_accuracy.csv").set_index("Model")
        for d, models in PAPER["table9"].items():
            for m, v in models.items():
                add("Table VIII", f"{SCOPE[d]} {m}", v, t9.loc[m, f"{SCOPE[d]} Accuracy"])

    out = pd.DataFrame(rows)
    write_table(out, tables, "07_paper_vs_computed")
    summary = out.groupby("Artifact", sort=False)["Match"].agg(Matched="sum", Total="count").reset_index()
    write_table(summary, tables, "07_paper_vs_computed_summary")
    return summary


# ---------------------------------------------------------------------------
# 08 Original protocol (exec/bn_t.py): the evaluation behind the first manuscript version
# ---------------------------------------------------------------------------
STORED_BN_DIR = DATA_DIR / "models"


def _stored_bn(data_type: str, two_class: bool) -> Path | None:
    path = STORED_BN_DIR / data_type / f"bn_model_wtf{'_2' if two_class else ''}_hc_bdeu_none.pkl"
    return path if path.exists() else None


def _rf_lr_on_bnt_inputs(raw: pd.DataFrame, thresholds, lingam_tce: pd.DataFrame | None = None) -> dict:
    """RF / LR on the same inputs as the BNs (bn_t.py node values, incl. working_time),
    trained on the training split. No RF/LR code survives in exec/, so this is the
    closest reconstruction, not the original run."""
    import paper_protocol as pp

    df = pp.preprocess_bnt(raw, thresholds)
    X = pp.encoded_features(df)
    y = df["fixing_time"].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    sets = {"": list(X.columns)}
    if lingam_tce is not None:
        linked = lingam_tce[(lingam_tce["probability"] >= LINGAM_PROB_THRESHOLD)
                            & ((lingam_tce["to"] == "fixing_time") | (lingam_tce["from"] == "fixing_time"))]
        chosen = (set(linked["from"]) | set(linked["to"])) & set(X.columns)
        sets["LiNGAM+"] = [c for c in X.columns if c in chosen] or list(X.columns)
    out = {}
    for prefix, cols in sets.items():
        Xs = StandardScaler().fit(X.iloc[idx_tr][cols]).transform(X[cols])
        out[f"{prefix}RF"] = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1) \
            .fit(X.iloc[idx_tr][cols], y[idx_tr]).predict(X.iloc[idx_te][cols])
        out[f"{prefix}LR"] = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED) \
            .fit(Xs[idx_tr], y[idx_tr]).predict(Xs[idx_te])
    return out


def step_08_original_protocol(output: Path, lingam_dir: Path = STORED_LINGAM_DIR, diagnostics: bool = True) -> pd.DataFrame:
    """Tables IV-VIII under the manuscript's evaluation protocol, plus a
    diagnostic table showing what each hidden condition contributes."""
    import paper_protocol as pp

    tables, _ = ensure_dirs(Path(output) / "original_protocol")
    preds4, preds2 = {}, {}
    for d in DATA_TYPES:
        raw = load_raw(d)
        p = pp.run_bn_models(raw, load_tce(lingam_dir, d, NAME_4CLASS), stored_bn=_stored_bn(d, False))
        for m, v in _rf_lr_on_bnt_inputs(raw, (1, 7, 30)).items():
            p[f"pred_{m}"] = v
        preds4[d] = p
        p.to_csv(tables / f"predictions_4class_{d}.csv", index=False)
        cut = (BINARY_THRESHOLDS_ORIGINAL[d],)
        p2 = pp.run_bn_models(raw, load_tce(lingam_dir, d, NAME_2CLASS, 2), cut, stored_bn=_stored_bn(d, True))
        p2 = p2.rename(columns={"pred_LiNGAM-BN": "pred_LiNGAM+BN"})
        for m, v in _rf_lr_on_bnt_inputs(raw, cut, load_tce(lingam_dir, d, NAME_2CLASS, 2)).items():
            p2[f"pred_{m}"] = v
        preds2[d] = p2
        p2.to_csv(tables / f"predictions_2class_{d}.csv", index=False)
        print(f"[paper protocol] {d}: done")

    for d, name in [("product", "table_IV_performance_product"), ("core", "table_IV_performance_core")]:
        rows = []
        for m in MODELS_4_ORIGINAL:
            rep = classification_report(preds4[d]["y_true"], preds4[d][f"pred_{m}"], labels=CLASSES,
                                        output_dict=True, zero_division=0)
            for metric, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
                rows.append({"Model": m, "Metric": metric, **{CLASS_LABELS[c]: round(rep[str(c)][key], 3) for c in CLASSES}})
        write_table(pd.DataFrame(rows), tables, name)

    write_table(pd.DataFrame([{
        "Model": m,
        **{f"{SCOPE[d]} macro-F1": round(f1_score(preds4[d]["y_true"], preds4[d][f"pred_{m}"], average="macro"), 3)
           for d in DATA_TYPES},
        **{f"{SCOPE[d]} accuracy": round(accuracy_score(preds4[d]["y_true"], preds4[d][f"pred_{m}"]), 4)
           for d in DATA_TYPES},
    } for m in MODELS_4_ORIGINAL]), tables, "table_V_macro_f1")

    rows = []
    for cls in CLASSES:
        for m in MODELS_4_ORIGINAL:
            row = {"Class": CLASS_LABELS[cls], "Model": m}
            for d in DATA_TYPES:
                p = preds4[d]
                yt, yr, ym = p["y_true"].values, p["pred_RF"].values, p[f"pred_{m}"].values
                row[f"{SCOPE[d]} F1"] = round(float(_f1_batch(yt[None], ym[None], cls)[0]), 3)
                row[f"{SCOPE[d]} d/p"] = "--" if m == "RF" else "{:.2f}/{:.3f}".format(*effect_vs_rf(yt, yr, ym, cls))
            rows.append(row)
    write_table(pd.DataFrame(rows), tables, "table_VI_effect_size_f1")

    # Table VII: BFT (days) of the test reports that this protocol's LiNGAM-BN predicts as class 1
    stats = {}
    for d in ["product", "core"]:
        s = preds4[d].loc[preds4[d]["pred_LiNGAM-BN"] == 1, "fixing_time"].describe()
        stats[SCOPE[d]] = [int(s["count"]), s["mean"], s["std"], s["min"], s["25%"], s["50%"], s["75%"], s["max"]]
    t8 = pd.DataFrame(stats, index=["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"]).round(3)
    write_table(t8.reset_index(names="Metric"), tables, "table_VII_short_duration_bft_stats")

    acc2 = {d: [round(accuracy_score(preds2[d]["y_true"], preds2[d][f"pred_{m}"]), 4) for m in MODELS_2_ORIGINAL]
                + [round(accuracy_score(preds4[d]["y_true"], preds4[d]["pred_LiNGAM-BN"]), 4)] for d in DATA_TYPES}
    write_table(pd.DataFrame({
        "Model": MODELS_2_ORIGINAL + ["LiNGAM+BN(Multi-class)"],
        "Core Accuracy": acc2["core"],
        "Product Accuracy": acc2["product"],
    }), tables, "table_VIII_binary_accuracy")

    if diagnostics:
        rows = []
        for d in DATA_TYPES:
            raw = load_raw(d)
            for fit_on in ["all", "train"]:
                for use_wt in [True, False]:
                    p = pp.run_bn_models(raw, load_tce(lingam_dir, d, NAME_4CLASS), fit_on=fit_on, use_working_time=use_wt,
                                         stored_bn=_stored_bn(d, False) if (fit_on == "all" and use_wt) else None)
                    for m in ["BN", "LiNGAM-BN"]:
                        rows.append({"Scope": SCOPE[d], "Model": m,
                                     "BN fitted on": "all reports (incl. test)" if fit_on == "all" else "training split",
                                     "working_time evidence": "yes" if use_wt else "no",
                                     "Accuracy": round(accuracy_score(p["y_true"], p[f"pred_{m}"]), 4),
                                     "Macro-F1": round(f1_score(p["y_true"], p[f"pred_{m}"], average="macro"), 3)})
        write_table(pd.DataFrame(rows), tables, "08_protocol_diagnostics")
    return compare_with_original_manuscript(tables)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def set_include_working_time(include: bool) -> None:
    """Sensitivity option: also use working_time (which contains fixing_time) as a predictor."""
    global EXCLUDE_WORKING_TIME, LEAKY
    EXCLUDE_WORKING_TIME = not include
    LEAKY = {"fixing_time", "fixing_time_class"} if include else \
        {"working_time", "working_time_class", "fixing_time", "fixing_time_class"}


def run_all(output: Path = DEFAULT_OUTPUT, retrain: bool = False, n_sampling: int = N_BOOTSTRAP,
            include_working_time: bool = False, original_protocol: bool = True) -> pd.DataFrame:
    set_include_working_time(include_working_time)
    tables, figures = ensure_dirs(output)
    lingam_dir = retrain_lingam(Path(output) / "lingam", n_sampling) if retrain else STORED_LINGAM_DIR
    print(f"[lingam] causal analysis (steps 03-04) uses {lingam_dir}")
    if retrain:
        for d in DATA_TYPES:
            for remove in (True, False):
                train_split_lingam(d, remove, retrain=True, n_sampling=n_sampling)
    step_01_data_collection(tables)
    step_02_table1(tables)
    step_03_heatmaps(tables, figures, lingam_dir)
    step_04_table3(tables, lingam_dir)
    step_05_prediction_tables(tables)
    step_06_binary_tables(tables)
    step_07_sensitivity(tables)
    step_09_transfer(tables)
    if original_protocol:
        return step_08_original_protocol(output, lingam_dir)
    return pd.DataFrame()


def main() -> None:
    ensure_fixed_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="all", choices=["all"])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--retrain-lingam", action="store_true",
                    help="re-run the LiNGAM bootstraps from data/raw instead of using data/lingam "
                         "and data/lingam_train")
    ap.add_argument("--n-sampling", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--include-working-time", action="store_true",
                    help="sensitivity run: use working_time as a predictor (it contains fixing_time)")
    ap.add_argument("--relearn-bn", action="store_true",
                    help="learn the standard BN structures again instead of using data/bn_structures")
    args = ap.parse_args()
    global RELEARN_BN
    RELEARN_BN = args.relearn_bn
    print(run_all(args.output, args.retrain_lingam, args.n_sampling, args.include_working_time).to_string(index=False))


if __name__ == "__main__":
    main()
