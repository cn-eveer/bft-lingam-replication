#!/usr/bin/env python3
"""Compute every paper table and figure from data/raw (see replicate.py)."""

import argparse
from pathlib import Path

import replicate

if __name__ == "__main__":
    replicate.ensure_fixed_hash_seed()  # deterministic BN structure search (see replicate.learn_structure)
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain-lingam", action="store_true")
    ap.add_argument("--n-sampling", type=int, default=replicate.N_BOOTSTRAP)
    ap.add_argument("--include-working-time", action="store_true")
    ap.add_argument("--include-workloads", action="store_true")
    ap.add_argument("--relearn-bn", action="store_true")
    args = ap.parse_args()
    replicate.RELEARN_BN = args.relearn_bn
    summary = replicate.run_all(replicate.DEFAULT_OUTPUT, args.retrain_lingam, args.n_sampling,
                                args.include_working_time, include_workloads=args.include_workloads)
    print(summary.to_string(index=False))

    # Compare with the tables printed in the manuscript when its LaTeX source is next to this package
    section = Path(__file__).resolve().parent.parent / "section"
    if section.exists():
        import sys

        import check_manuscript

        sys.argv = [sys.argv[0], "--section", str(section)]
        print("\n== Manuscript check (values read from section/*.tex) ==")
        check_manuscript.main()
