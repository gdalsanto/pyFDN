"""
Dedicated direct DSS2PR example.

Uses ``dss_to_pr_direct`` (numeric DSS-only path).
"""

from __future__ import annotations

import numpy as np

import pyFDN


def main() -> None:
    np.random.seed(11)

    n = 4
    delays = np.array([41, 53, 67, 79], dtype=int)
    a_num = 0.65 * pyFDN.random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    residues, poles, direct, is_pair, _ = pyFDN.dss_to_pr_direct(
        delays,
        a_num,
        b,
        c,
        d,
        mode="eig",
    )

    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, a_num, b, c, d)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    err = np.max(np.abs(ir_time - ir_modal))
    print("max |IR_time - IR_modal|:", err)
    assert err < 1e-7


if __name__ == "__main__":
    main()

