"""STUDENT IMPLEMENTATION: validate 65 runs, aggregate five seeds, and plot.

Require exactly layers 0..12 and seeds 0..4, without duplicates, mixed splits,
nonfinite scores, inconsistent counts, or mixed data/protocol fingerprints.
Write 13 rows with fields:
layer,split,n_seeds,mean_accuracy,std_accuracy,se_accuracy,data_sha256,protocol_sha256
Use std(ddof=1) / sqrt(5), not population SD or binomial standard error.
Plot mean dev accuracy and +/- 1 SE. Label layer 0 as embedding-module output.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    raise NotImplementedError("Implement validated five-seed aggregation and the layer accuracy figure")


if __name__ == "__main__":
    main()
