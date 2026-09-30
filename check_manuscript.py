#!/usr/bin/env python3
"""Check that the replication outputs reproduce the tables printed in the manuscript.

Every value is read directly from the LaTeX source (default: ../section), so the
check always reflects the current manuscript. A value matches when the computed
number rounds to what the paper prints (tolerance = half a unit of the last
printed digit). Run `python analyze.py` first.

    python check_manuscript.py [--section ../section]

Writes output/manuscript_check.csv (every value) and prints a summary.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HELD = HERE / "output" / "tables"
PAPER = HERE / "output" / "paper_protocol" / "tables"
CLASS_COLS = ["Class 1 (<=1d)", "Class 2 (2-7d)", "Class 3 (8-30d)", "Class 4 (>=31d)"]


# ---------------------------------------------------------------------------
# LaTeX table parsing
# ---------------------------------------------------------------------------
def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def table_rows(tex: str, label: str) -> list[list[str]]:
    """Body rows of the tabular in the table environment that has `label`."""
    tex = _strip_comments(tex)
    pos = tex.index(f"\\label{{{label}}}")
    start = tex.rfind("\\begin{table", 0, pos)
    end = tex.index("\\end{table", pos)
    body = tex[start:end]
    body = body[body.index("\\begin{tabular}"):body.index("\\end{tabular}")]
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", body)
    rows = []
    for raw in re.split(r"\\\\", body):
        raw = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", raw)
        raw = re.sub(r"\\cmidrule\{[^}]*\}", "", raw).strip()
        if raw:
            rows.append([clean(c) for c in raw.split("&")])
    return rows


def clean(cell: str) -> str:
    cell = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", cell)
    for cmd in ["textbf", "underline", "textit", "emph", "makecell"]:
        cell = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", r"\1", cell)
    cell = cell.replace("\\%", "%").replace("{", "").replace("}", "")
    return " ".join(cell.split())


def num(text: str):
    """(value, decimals) of the first number in a cell, or None."""
    m = re.search(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    s = m.group(0).replace(",", "")
    return float(s), (len(s.split(".")[1]) if "." in s else 0)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
RESULTS: list[dict] = []


def compare(table: str, item: str, printed: str, computed, source: str) -> None:
    parsed = num(printed)
    if isinstance(computed, str) or parsed is None:
        ok = str(printed).strip() == str(computed).strip()
    elif computed is None or (isinstance(computed, float) and np.isnan(computed)):
        ok = False
    else:
        value, dec = parsed
        ok = abs(value - float(computed)) <= 0.5 * 10 ** (-dec) + 1e-9
    RESULTS.append({"Table": table, "Item": item, "Paper": printed, "Computed": computed,
                    "Source": source, "Match": bool(ok)})


def check_table1(section: Path) -> None:
    rows = table_rows((section / "data.tex").read_text(), "tab:bug_report_stats")
    comp = pd.read_csv(HELD / "table_1_bug_fixing_time_distribution.csv").fillna("")
    group = ""
    for r in rows:
        if len(r) < 7 or not r[2] or r[2].startswith("BFT Class") or r[0] == "Bug Type":
            continue
        group = r[0] or group
        hit = comp[(comp["Bug Type"] == group) & (comp["Project"].str.lower() == r[1].lower())]
        if hit.empty:
            compare("Table 1", f"{group} {r[1]}", r[6], None, "held-out Table 1")
            continue
        h = hit.iloc[0]
        for c, col in enumerate(CLASS_COLS):
            n, pct = re.match(r"\s*([\d,]+)\s*\(\s*([\d.]+)%\)", r[2 + c]).groups()
            cn, cpct = re.match(r"\s*([\d,]+)\s*\(\s*([\d.]+)%\)", str(h[col])).groups()
            compare("Table 1", f"{group} {r[1]} class {c + 1} count", n, float(cn.replace(",", "")), "output/tables")
            compare("Table 1", f"{group} {r[1]} class {c + 1} %", pct, float(cpct), "output/tables")
        compare("Table 1", f"{group} {r[1]} total", r[6], float(h["Total"]), "output/tables")


def check_heatmaps() -> None:
    for d, fig in [("core", "figure_5_causal_strength_core"), ("product", "figure_6_causal_strength_product")]:
        paper = pd.read_csv(HERE / "data" / "paper_result" / f"heatmap_{d}.csv").set_index("Factor")
        comp = pd.read_csv(HELD / f"{fig}_data.csv").set_index("Factor")
        for f in paper.index:
            for pc, cc in zip(paper.columns, ["all", "1", "2", "3", "4"]):
                pv, cv = paper.loc[f, pc], comp.loc[f, cc] if f in comp.index else np.nan
                if pd.isna(pv) and pd.isna(cv):
                    continue
                # heatmap annotations use 2 significant digits
                ok = (not pd.isna(pv) and not pd.isna(cv) and float(f"{cv:.2g}") == float(f"{pv:.2g}"))
                RESULTS.append({"Table": f"Figure heat_{d}", "Item": f"{f} {pc}", "Paper": pv,
                                "Computed": cv, "Source": "output/tables", "Match": bool(ok)})


def check_table3(section: Path) -> None:
    rows = table_rows((section / "results" / "RQ2.tex").read_text(), "tab:causal_effects_merged")
    comp = pd.read_csv(HELD / "table_3_lingam_strength_direction.csv").set_index("Factor")
    for r in rows:
        if len(r) == 6 and r[1] in comp.index:
            for c, col in enumerate(CLASS_COLS):
                compare("Table 3", f"{r[1]} class {c + 1}", r[2 + c], comp.loc[r[1], col], "output/tables")


def check_performance(section: Path) -> None:
    rows = table_rows((section / "results" / "RQ3.tex").read_text(), "tab:performance_by_model")
    comp = {"Product": pd.read_csv(PAPER / "table_4_prediction_product.csv"),
            "Core": pd.read_csv(PAPER / "table_5_prediction_core.csv")}
    scope = ""
    for r in rows:
        if len(r) != 14 or r[2] == "P" or not num(r[2]):
            continue
        scope = r[0] or scope
        for c, col in enumerate(CLASS_COLS):
            for k, metric in enumerate(["Precision", "Recall", "F1"]):
                t = comp[scope]
                v = t[(t["Model"] == r[1]) & (t["Metric"] == metric)][col].iloc[0]
                compare("Table performance (P/R/F1)", f"{scope} {r[1]} {metric} class {c + 1}", r[2 + 3 * c + k], v,
                        "output/paper_protocol")


def check_macro(section: Path) -> None:
    rows = table_rows((section / "results" / "RQ3.tex").read_text(), "tab:summary_accuracy_only")
    comp = pd.read_csv(PAPER / "table_6_macro_f1.csv").set_index("Model")
    for r in rows:
        if len(r) == 3 and r[0] in comp.index:
            compare("Table macro-F1", f"Core {r[0]}", r[1], comp.loc[r[0], "Core macro-F1"], "output/paper_protocol")
            compare("Table macro-F1", f"Product {r[0]}", r[2], comp.loc[r[0], "Product macro-F1"], "output/paper_protocol")


def check_effects(section: Path) -> None:
    rows = table_rows((section / "results" / "RQ3.tex").read_text(), "tab:classwise_f1_effect")
    comp = pd.read_csv(PAPER / "table_7_effect_size_f1.csv")
    cls = ""
    for r in rows:
        if len(r) != 6 or r[1] in ("", "Model"):
            continue
        cls = r[0] or cls
        c = int(cls[0])
        hit = comp[(comp["Class"] == CLASS_COLS[c - 1]) & (comp["Model"] == r[1])].iloc[0]
        for scope, fi, di in [("Core", 2, 3), ("Product", 4, 5)]:
            compare("Table effect size (F1)", f"class {c} {r[1]} {scope} F1", r[fi], hit[f"{scope} F1"], "output/paper_protocol")
            if r[di] != "--":
                pd_, pp = r[di].split("/")
                cd, cp = hit[f"{scope} d/p"].split("/")
                compare("Table effect size (d)", f"class {c} {r[1]} {scope} d", pd_, float(cd), "output/paper_protocol")
                compare("Table effect size (p)", f"class {c} {r[1]} {scope} p", pp, float(cp), "output/paper_protocol")


def check_rq4(section: Path) -> None:
    tex = (section / "results" / "RQ4.tex").read_text()
    comp = pd.read_csv(PAPER / "table_8_short_duration_bft_stats.csv").set_index("Metric")
    for r in table_rows(tex, "tab:class1_stats_lingam"):
        if len(r) == 3 and r[0] in comp.index:
            compare("Table short-duration stats", f"Product {r[0]}", r[1], comp.loc[r[0], "Product"], "output/paper_protocol")
            compare("Table short-duration stats", f"Core {r[0]}", r[2], comp.loc[r[0], "Core"], "output/paper_protocol")
    comp = pd.read_csv(PAPER / "table_9_binary_classification_accuracy.csv")
    comp["key"] = comp["Model"].str.replace(" ", "")
    comp = comp.set_index("key")
    for r in table_rows(tex, "tab:binary_class_accuracy"):
        key = r[0].replace(" ", "")
        if len(r) == 3 and key in comp.index:
            compare("Table binary accuracy", f"Core {r[0]}", r[1], comp.loc[key, "Core Accuracy"], "output/paper_protocol")
            compare("Table binary accuracy", f"Product {r[0]}", r[2], comp.loc[key, "Product Accuracy"], "output/paper_protocol")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=Path, default=HERE.parent / "section")
    args = ap.parse_args()
    if not PAPER.exists():
        raise SystemExit("Run `python analyze.py` first.")
    check_table1(args.section)
    check_heatmaps()
    check_table3(args.section)
    check_performance(args.section)
    check_macro(args.section)
    check_effects(args.section)
    check_rq4(args.section)
    out = pd.DataFrame(RESULTS)
    out.to_csv(HERE / "output" / "manuscript_check.csv", index=False)
    summary = out.groupby("Table", sort=False)["Match"].agg(Matched="sum", Total="count").reset_index()
    print(summary.to_string(index=False))
    print(f"\nAll: {int(out['Match'].sum())} / {len(out)} printed values reproduced "
          f"(details: output/manuscript_check.csv)")


if __name__ == "__main__":
    main()
