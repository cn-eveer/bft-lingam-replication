#!/usr/bin/env python3
"""Record input checksums and the filtering funnel (Bugzilla collection itself is not re-run)."""

import replicate

if __name__ == "__main__":
    tables, _ = replicate.ensure_dirs(replicate.DEFAULT_OUTPUT)
    print(replicate.step_01_data_collection(tables).to_string(index=False))
