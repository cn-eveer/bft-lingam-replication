#!/usr/bin/env python3
"""Choice of the refined LiNGAM-BN configuration (replicate.REFINED_*).

48 LiNGAM-BN variants are scored by 5-fold cross-validation on the training split of
each scope and task (4 classes, binary <= 7 days); the test split is not used:

  edges  : total (total effects, as the original LiNGAM-BN) | direct (bootstrap adjacency)
  lingam : base (all PRED_LINGAM_COLUMNS) | prior (no total resolution time, temporal prior)
  p      : 0.5 | 0.8      bootstrap probability threshold
  cap    : none | 3       maximum number of parents per node
  ess    : 1 | 5 | 10     BDeu equivalent sample size

Two hybrids (hill climbing started from / constrained to the direct-prior LiNGAM graph)
and RF are scored on the same folds for reference. The hybrids re-run pgmpy's search,
whose tie-breaking depends on the Python process, so their scores can vary slightly.

Writes output/tables/refinement_cv_variants.csv.
"""

import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import replicate as r

GRID_DIR = r.REFINED_LINGAM_DIR / "grid"
T = r.BN_TARGET
NODES = r.predictors() + ["discussion_days_class", "assigned_days_class"]


def edge_table(d: str, edges: str, lingam: str) -> pd.DataFrame:
    if edges == "total" and lingam == "base":
        return r.train_split_lingam(d)
    if edges == "direct" and lingam == "prior":
        return r.refined_lingam(f"train_{d}_iqr")
    return pd.read_csv(GRID_DIR / f"{edges}_{d}_{lingam}.csv")


def predict(edges, d_tr, d_te, ess):
    nodes = sorted({n for e in edges for n in e} | {T})
    return r.DiscreteBN(edges, ess=ess).fit(d_tr[nodes]).predict(
        d_te[[n for n in nodes if n != T]].reset_index(drop=True))


def _hc(columns, values, lingam_edges, mode):
    from pgmpy.base import DAG
    from pgmpy.estimators import BDeuScore, HillClimbSearch

    data = pd.DataFrame(np.asarray(values, dtype=np.int64), columns=columns)
    kw = {}
    if mode == "start":
        dag = DAG()
        dag.add_nodes_from(columns)
        dag.add_edges_from(lingam_edges)
        kw["start_dag"] = dag
    else:
        kw["fixed_edges"] = set(lingam_edges)
    return list(HillClimbSearch(data).estimate(
        scoring_method=BDeuScore(data, equivalent_sample_size=r.BDEU_ESS), show_progress=False, **kw).edges())


def hybrid(d_tr, lingam_edges, mode):
    os.environ["PYTHONHASHSEED"] = "0"
    data = d_tr[NODES + [T]]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_hc, list(data.columns), data.to_numpy(np.int64), lingam_edges, mode).result()


def evaluate(d: str, task: str) -> pd.DataFrame:
    df = r.preprocess(d, 4)
    if task != "4class":
        df["fixing_time_class"] = np.where(df["fixing_time"] <= r.BINARY_THRESHOLD, 1, 2)
    tr, _ = r.split(df)
    y = tr["fixing_time_class"].astype(int).values
    folds = list(StratifiedKFold(5, shuffle=True, random_state=r.RANDOM_SEED).split(tr, tr["bft_class4"]))
    frames = []
    for a, b in folds:
        f_tr, f_te = r.discretize(tr.iloc[a], tr.iloc[b], NODES)
        f_tr[T] = y[a]
        frames.append((f_tr, f_te, y[b]))
    tabs = {(e, l): edge_table(d, e, l) for e in ("total", "direct") for l in ("base", "prior")}

    def score(edges, ess):
        return float(np.mean([f1_score(yv, predict(edges, a, b, ess), average="macro") for a, b, yv in frames]))

    def variant(cfg):
        e, l, p, cap, ess = cfg
        edges = r.refined_edges(tabs[(e, l)], NODES + [T], p, cap)
        return {"Model": "LiNGAM-BN", "Edges": e, "LiNGAM": l, "p": p, "Max parents": cap or "none",
                "ESS": ess, "Edges in BN": len(edges), "CV macro-F1": round(score(edges, ess), 4)}

    grid = itertools.product(("total", "direct"), ("base", "prior"), (0.5, 0.8), (None, 3), (1.0, 5.0, 10.0))
    rows = Parallel(n_jobs=8)(delayed(variant)(c) for c in grid)
    base = r.refined_edges(tabs[("direct", "prior")], NODES + [T], 0.5, None)
    for mode in ("start", "fixed"):
        s = float(np.mean([f1_score(yv, predict(hybrid(a, base, mode), a, b, r.BDEU_ESS), average="macro")
                           for a, b, yv in frames]))
        rows.append({"Model": f"Hill climbing ({mode} from LiNGAM graph)", "Edges": "direct", "LiNGAM": "prior",
                     "p": 0.5, "Max parents": "none", "ESS": r.BDEU_ESS, "CV macro-F1": round(s, 4)})
    s = []
    for (a, b), (_, _, yv) in zip(folds, frames):
        rf = RandomForestClassifier(n_estimators=100, random_state=r.RANDOM_SEED, n_jobs=-1)
        s.append(f1_score(yv, rf.fit(tr.iloc[a][r.predictors()], y[a]).predict(tr.iloc[b][r.predictors()]),
                          average="macro"))
    rows.append({"Model": "RF (reference)", "CV macro-F1": round(float(np.mean(s)), 4)})
    out = pd.DataFrame(rows)
    out.insert(0, "Task", task)
    out.insert(0, "Scope", r.SCOPE[d])
    return out


if __name__ == "__main__":
    tables, _ = r.ensure_dirs(r.DEFAULT_OUTPUT)
    res = pd.concat([evaluate(d, t) for d in r.DATA_TYPES for t in ("4class", f"binary{r.BINARY_THRESHOLD}")])
    r.write_table(res, tables, "refinement_cv_variants")
    # selection rule: highest mean CV macro-F1 over the four scope/task combinations; ties are
    # broken by the sparsest graph (higher p), then by the ESS of the standard BN
    cfg = ["Edges", "LiNGAM", "p", "Max parents", "ESS"]
    lb = res[res["Model"] == "LiNGAM-BN"].copy()
    mean = lb.groupby(cfg, dropna=False)["CV macro-F1"].mean().round(4).rename("Mean CV macro-F1").reset_index()
    mean["ess_default"] = (mean["ESS"] == r.BDEU_ESS)
    mean = mean.sort_values(["Mean CV macro-F1", "p", "ess_default"], ascending=False)
    r.write_table(mean.drop(columns="ess_default"), tables, "refinement_cv_ranking")
    print(mean.drop(columns="ess_default").head(8).to_string(index=False))
