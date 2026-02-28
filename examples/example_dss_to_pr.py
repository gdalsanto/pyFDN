"""
Translation of fdnToolbox/Examples/example_dss2pr.m.

Small modal-decomposition example:
- synthesize IR directly from DSS
- decompose to poles/residues with dss_to_pr
- re-synthesize IR from poles/residues
- estimate residues from IR + poles with impz_to_res
"""

from __future__ import annotations

import numpy as np

import pyFDN


def main() -> None:
    np.random.seed(5)
    fs = 48_000
    impulse_response_length = fs // 4

    # Define FDN
    n = 4
    num_input = 1
    num_output = 1
    input_gain = np.eye(n, num_input)
    output_gain = np.eye(num_output, n)
    direct = np.zeros((num_output, num_input))
    delays = np.random.randint(50, 101, size=n)
    feedback_matrix = pyFDN.random_orthogonal(n)

    # Absorption filters (identity)
    absorption = pyFDN.ZScalar(np.ones((n, 1)), is_diagonal=True)

    # Compute
    ir_time_domain = pyFDN.dss_to_impz(
        impulse_response_length, delays, feedback_matrix, input_gain, output_gain, direct
    )[:, 0, 0]
    residues, poles, direct_term, is_conjugate, _ = pyFDN.dss_to_pr(
        delays,
        feedback_matrix,
        input_gain,
        output_gain,
        direct,
        absorption_filters=absorption,
        verbose=False,
    )
    ir_res_pol = pyFDN.pr_to_impz(
        residues, poles, direct_term, is_conjugate, impulse_response_length
    )[:, 0, 0]
    res_ir, _, _ = pyFDN.impz_to_res(ir_time_domain, poles, is_conjugate)

    difference = ir_time_domain - ir_res_pol
    print("max |IR_time - IR_res/pol|:", np.max(np.abs(difference)))
    print("max |residues - residues_from_ir|:", np.max(np.abs(residues[:, 0, 0] - res_ir)))

    # Verification translated from MATLAB asserts
    assert np.max(np.abs(difference)) < 1e-8
    assert np.max(np.abs(residues[:, 0, 0] - res_ir)) < 1e-8


if __name__ == "__main__":
    main()

