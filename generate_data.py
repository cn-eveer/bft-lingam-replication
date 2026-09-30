#!/usr/bin/env python3
"""Write the preprocessed analysis datasets (4-class and 2-class) used by every step."""

import replicate

if __name__ == "__main__":
    out = replicate.DEFAULT_OUTPUT / "analysis_data"
    out.mkdir(parents=True, exist_ok=True)
    for data_type in replicate.DATA_TYPES:
        for n_classes in (4, 2):
            path = out / f"{data_type}_{n_classes}class.csv"
            replicate.preprocess(data_type, n_classes).to_csv(path, index=False)
            print(f"[data] {path}")
