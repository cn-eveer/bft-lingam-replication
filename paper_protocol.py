"""Paper protocol: the evaluation behind the manuscript's BN / LiNGAM-BN results.

Re-implements exec/bn_t.py (Feb 2025), which produced the saved models
`result/{core,product}/model/bn_model_wtf_hc_bdeu_none.pkl` and
`bn_model-lingam_wtf_prior_bayes_bdeu_none.pkl`:

  * preprocessing = bn_t.process(filter=True): categorical mappings, IQR on
    working_time, severity/priority changed, then FIXED / not self-reported /
    not newcomer developer;
  * BN        = bnlearn hill climbing (BDeu) + parameter learning;
  * LiNGAM-BN = bnlearn DAG built from the LiNGAM edges with bootstrap
    probability >= 0.8 (bn_t.get_lingam_edge), Bayesian parameters (BDeu);
  * evaluation = stratified 20 % split (seed 42), evidence = every BN node
    except fixing_time.

Two properties of this protocol explain the manuscript's high scores and are
reported explicitly:
  1. the BNs are fitted on ALL reports, so the test reports are also in the
     training data (`fit_on="all"`); `fit_on="train"` is the held-out variant;
  2. working_time (creation -> resolution, binned at the same 1/7/30-day cut
     points as BFT) is part of the evidence; `use_working_time=False` removes it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# exec/bn_t.py `columns`
BNT_COLUMNS = [
    "product", "component", "severity", "priority", "is_self_reported", "resolution",
    "system_workloads", "developer_workloads", "num_blocks", "num_depends", "num_duplicates",
    "is_newcomer_developer", "reporter_reputations", "assigned_days", "working_time",
    "fixing_time", "discussion_days", "comment_count", "number_of_attachments", "position",
    "is_fixed",
]

# exec/bn_t.py / exec/lingam_s.py `grouping["core"]`
GROUPING = {
    "Core Browser Technology": [
        "CSS Parsing and Computation", "CSS Transitions and Animations", "DOM", "Graphics",
        "Layout", "Web Painting", "SVG", "JavaScript Engine", "XBL", "XML", "XPCOM",
        "XPConnect", "XSLT", "XUL", "mozglue", "MathML", "JavaScript: Standard Library",
        "JavaScript: GC", "JavaScript: Internationalization API", "JavaScript: WebAssembly",
        "MFBT", "JavaScript",
    ],
    "User Interface & Accessibility": [
        "Widget", "Window Management", "Disability Access APIs", "Print Preview", "Printing",
        "Preferences", "Internationalization", "General", "Web Audio",
    ],
    "Security & Networking": [
        "Audio/Video", "WebVR", "Web Speech", "WebRTC", "Networking", "IPC",
        "Hardware Abstraction Layer (HAL)", "Memory Allocator", "Performance",
        "Permission Manager", "Privacy", "Security", "Spelling checker", "Storage", "Fuzzing",
        "Gecko Profiler", "Find Backend", "Panning and Zooming", "DMD", "Sanitizers",
        "DLL Services", "js-ctypes", "AutoConfig (Mission Control Desktop)",
    ],
}


def _three(value, low, mid, labels=("low", "middle", "high")):
    return labels[0] if value <= low else (labels[1] if value <= mid else labels[2])


def preprocess_bnt(raw: pd.DataFrame, thresholds: tuple[float, ...] = (1, 7, 30)) -> pd.DataFrame:
    """exec/bn_t.py process(df, data_type, filter=True) + the FIXED / self / newcomer filter.

    `thresholds` are the BFT class cut points: (1, 7, 30) gives 4 classes,
    a single value (e.g. (9,)) gives the 2-class split used for Table VIII."""
    df = raw.copy()
    df["raw_fixing_time"] = df["fixing_time"]
    df["priority"] = df["priority"].map({"P1": "high", "P2": "medium", "P3": "medium",
                                         "P4": "low", "P5": "low", "--": "low"})
    df["severity"] = df["severity"].map({"critical": "high", "blocker": "high", "major": "medium",
                                         "trivial": "low", "minor": "low", "--": "low"})
    df["reporter_reputations"] = df["reporter_reputations"].apply(lambda v: _three(v, 0.3, 0.7))
    df["comment_count"] = df["comment_count"].apply(lambda v: "low" if v <= 20 else "high")
    exist = lambda v: "exist" if v else "none"
    df["number_of_attachments"] = df["number_of_attachments"].apply(exist)
    df["system_workloads"] = df["system_workloads"].apply(lambda v: _three(v, 30, 1000))
    df["developer_workloads"] = df["developer_workloads"].apply(lambda v: _three(v, 10, 20))
    for col in ["num_blocks", "num_depends", "num_duplicates"]:
        df[col] = df[col].apply(exist)
    df["assigned_days"] = df["assigned_days"].apply(lambda v: int(v >= 1))

    # IQR filter on working_time. bn_t.py indexes with the Python set of kept
    # row labels, so the row order (and therefore the split) follows set order.
    q1, q3 = df["working_time"].quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = set(df[(df["working_time"] >= q1 - 1.5 * iqr) & (df["working_time"] <= q3 + 1.5 * iqr)].index)
    df = df.loc[list(keep)]

    cuts = list(thresholds)
    df["fixing_time"] = np.select([df["fixing_time"] <= c for c in cuts] + [df["fixing_time"] > cuts[-1]],
                                  list(range(1, len(cuts) + 2)))
    df["discussion_days"] = np.select([df["discussion_days"] <= 7, df["discussion_days"] > 7], [0, 1])
    df["working_time"] = np.select([df["working_time"] <= 1, df["working_time"] <= 7,
                                    df["working_time"] <= 30, df["working_time"] > 30], [0, 1, 2, 3])
    df["is_fixed"] = df["resolution"].apply(lambda v: int(v == "FIXED"))
    df = df[df["severity_priority_changed"] == 1]
    df["component"] = df["component"].apply(lambda v: v.split(":")[0])
    df["group"] = df["component"].apply(
        lambda v: next((k for k, vals in GROUPING.items() if v in vals), "Other"))
    df = df[BNT_COLUMNS + ["group", "raw_fixing_time"]]
    return df[(df["is_self_reported"] == 0) & (df["is_newcomer_developer"] == 0) & (df["resolution"] == "FIXED")]


def lingam_edges_bnt(tce: pd.DataFrame, prob: float = 0.8) -> list[tuple[str, str]]:
    """exec/bn_t.py get_lingam_edge(): every LiNGAM edge with probability >= 0.8,
    and each source also points to is_fixed and resolution."""
    df = tce.sort_values("from")
    df = df[df["probability"] >= prob]
    rel: dict[str, list[str]] = {}
    for src, dst in zip(df["from"], df["to"]):
        rel.setdefault(src, ["is_fixed", "resolution"]).append(dst)
    return [(p, c) for p, children in rel.items() if p in BNT_COLUMNS for c in children if c in BNT_COLUMNS]


def _predict(model, test: pd.DataFrame) -> np.ndarray:
    from pgmpy.inference import VariableElimination

    graph = model["model"]
    infer = VariableElimination(graph)
    nodes = [n for n in graph.nodes() if n != "fixing_time"]
    fallback = graph.get_cpds("fixing_time").state_names["fixing_time"][0]
    out = []
    for row in test[nodes].to_dict(orient="records"):
        try:
            q = infer.query(["fixing_time"], evidence=row, show_progress=False)
            out.append(q.state_names["fixing_time"][int(np.argmax(q.values))])
        except Exception:  # evidence state unseen in training (held-out fit only)
            out.append(fallback)
    return np.array(out)


def run_bn_models(raw: pd.DataFrame, tce: pd.DataFrame, thresholds=(1, 7, 30),
                  fit_on: str = "all", use_working_time: bool = True,
                  stored_bn: str | None = None) -> pd.DataFrame:
    """Train BN and LiNGAM-BN with the bn_t.py protocol and predict the test split.

    Hill-climbing structure search is not exactly deterministic, so `stored_bn`
    (the original saved bnlearn BN) can be evaluated instead of a re-learned BN.
    LiNGAM-BN has a fixed structure and is always re-fitted."""
    import bnlearn as bn

    df = preprocess_bnt(raw, thresholds)
    train, test = train_test_split(df, test_size=0.2, stratify=df["fixing_time"], random_state=42)
    fit_df = (df if fit_on == "all" else train).drop(columns=["raw_fixing_time"])
    if not use_working_time:
        fit_df = fit_df.drop(columns=["working_time"])

    if stored_bn is not None:
        hc = bn.load(str(stored_bn), verbose=0)
    else:
        hc = bn.structure_learning.fit(fit_df, methodtype="hc", scoretype="bdeu", verbose=0)
        hc = bn.parameter_learning.fit(hc, fit_df[list(set(hc["model"].nodes()) & set(fit_df.columns))], verbose=0)

    edges = [e for e in lingam_edges_bnt(tce) if use_working_time or "working_time" not in e]
    nodes = sorted({n for e in edges for n in e})
    lbn = bn.parameter_learning.fit(bn.make_DAG(edges, verbose=0), fit_df[nodes],
                                    methodtype="bayes", scoretype="bdeu", verbose=0)
    return pd.DataFrame({"id": test["raw_fixing_time"].index, "fixing_time": test["raw_fixing_time"].values,
                         "y_true": test["fixing_time"].values,
                         "pred_BN": _predict(hc, test), "pred_LiNGAM-BN": _predict(lbn, test)})


def encoded_features(df: pd.DataFrame, use_working_time: bool = True) -> pd.DataFrame:
    """bn_t.py node values as integer codes (for RF / LR on the same inputs as the BNs)."""
    drop = {"fixing_time", "raw_fixing_time", "resolution", "is_fixed", "is_self_reported",
            "is_newcomer_developer"} | (set() if use_working_time else {"working_time"})
    return df[[c for c in df.columns if c not in drop]].apply(lambda s: s.astype("category").cat.codes)
