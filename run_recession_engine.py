#!/usr/bin/env python3
"""
Compatibility runner that delegates to the canonical scheduler pipeline.

This script is intentionally thin to avoid maintaining duplicate training logic
outside `scheduler/update_job.py` and `recession_engine/*`.
"""

import argparse
import sys

from scheduler.update_job import run_update_job


def main():
    parser = argparse.ArgumentParser(description="Run recession engine update pipeline")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in months")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Training data end date (YYYY-MM-DD)")
    parser.add_argument("--max-features", type=int, default=None, help="Maximum selected features")
    parser.add_argument("--threshold-override", type=float, default=None,
                        help="Optional manual threshold override in [0,1]")
    args = parser.parse_args()

    ok = run_update_job(
        horizon_months=args.horizon,
        train_end_date=args.train_end,
        max_features=args.max_features,
        threshold_override=args.threshold_override,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
