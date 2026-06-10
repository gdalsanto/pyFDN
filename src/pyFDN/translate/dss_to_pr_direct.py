"""Direct DSS modal decomposition (dss2pr_direct-style, numeric only)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.auxiliary.poles import reduce_conjugate_pairs
from pyFDN.translate.dss_to_ss import dss_to_ss


def _dss_to_res_direct(
    poles: np.ndarray,
    delays: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Residues from poles using SVD-based formula (same as FLAMO path).

    P(z) = diag(z^m) - A(z), dP/dz = diag(m z^{m-1}) - A'(z). For a static A
    the derivative term vanishes; for an FIR polynomial A of shape (N, N, L)
    in z^{-1} convention, A'(z) = sum_k (-k) A_k z^{-k-1}. At each pole:
    SVD(P) gives right null vector r, left null vector l;
    denom = l^H (dP/dz) r; residue numerator = (C @ r) @ (l^H @ B).
    """
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    delays = np.asarray(delays, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    D = np.asarray(D, dtype=np.complex128)

    n_poles = poles.size
    n = delays.size
    n_in = B.shape[1]
    n_out = C.shape[0]

    r_den = np.zeros(n_poles, dtype=np.complex128)
    r_nom = np.zeros((n_poles, n_out, n_in), dtype=np.complex128)
    eig_right = np.zeros((n, n_poles), dtype=np.complex128)
    eig_left = np.zeros((n, n_poles), dtype=np.complex128)

    a_orders = np.arange(A.shape[2]) if A.ndim == 3 else None

    for it, pole in enumerate(poles):
        # P(z) = diag(z^m) - A(z), dP/dz = diag(m z^{m-1}) - A'(z)
        z_m = np.power(pole, delays)
        z_m1 = np.power(pole, delays - 1.0)
        if a_orders is None:
            a_val = A
            da_val = 0.0
        else:
            z_pow = np.power(pole, -a_orders)  # z^{-k}
            a_val = np.sum(A * z_pow[None, None, :], axis=2)
            da_val = np.sum(
                A * (-a_orders * np.power(pole, -a_orders - 1))[None, None, :],
                axis=2,
            )
        p = np.diag(z_m) - a_val
        dp = np.diag(delays * z_m1) - da_val

        # Null vectors: P r = 0, l^H P = 0 (same as FLAMO)
        u, s, vh = np.linalg.svd(p, full_matrices=False)
        r = vh.conj().T[:, -1]
        l = u[:, -1]

        denom = np.vdot(l, dp @ r)
        r_den[it] = denom
        eig_right[:, it] = r
        eig_left[:, it] = l

        cr = C @ r.reshape(-1, 1)
        lh_b = np.conj(l).reshape(1, -1) @ B
        r_nom[it, :, :] = cr @ lh_b

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn(
            "There are multipoles. The residues are set to zero.", stacklevel=2
        )
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)

    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, D, undriven, eigenvectors


def dss_to_pr_direct(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    mode: str = "eig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Direct poles/residues from simple DSS (numeric matrices only).

    ``A`` may be a static matrix of shape (N, N) or an FIR polynomial matrix
    of shape (N, N, L) in z^{-1} convention (e.g. a paraunitary feedback
    matrix or absorption FIRs folded into the loop). Polynomial matrices
    require ``mode='roots'``.

    Pole extraction modes:
      - ``eig``: eigenvalues of equivalent state-space matrix
      - ``roots``: roots of generalized characteristic polynomial
      - ``polyeig``: currently mapped to roots-based extraction
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    a_mat = np.asarray(A, dtype=np.complex128)
    b_mat = np.asarray(B, dtype=np.complex128)
    c_mat = np.asarray(C, dtype=np.complex128)
    d_mat = np.asarray(D, dtype=np.complex128)

    n = delays_arr.size
    if a_mat.shape[:2] != (n, n) or a_mat.ndim not in (2, 3):
        raise ValueError(
            f"A must have shape ({n},{n}) or ({n},{n},L), got {a_mat.shape}"
        )

    mode_l = str(mode).lower()
    if a_mat.ndim == 3 and mode_l == "eig":
        raise ValueError(
            "Polynomial (FIR) feedback matrices require mode='roots' "
            "(no equivalent static state-space matrix)."
        )
    a_poly = np.real_if_close(a_mat, tol=1000)
    if np.iscomplexobj(a_poly) and np.max(np.abs(np.imag(a_poly))) > 1e-12:
        raise ValueError(
            "roots/polyeig mode currently requires a real-valued feedback matrix."
        )
    a_poly = np.asarray(np.real(a_poly), dtype=float)

    if mode_l == "eig":
        aa, _, _, _ = dss_to_ss(delays_arr, a_mat)
        poles = np.linalg.eigvals(aa)
    elif mode_l in ("roots", "polyeig"):
        if mode_l == "polyeig":
            warnings.warn(
                "mode='polyeig' is currently mapped to roots-based extraction in dss_to_pr_direct.",
                stacklevel=2,
            )
        # GCP p has p[k] = coef of z^{-k}; polynomial in w = z^{-1} is sum_k p[k] w^k
        p = general_char_poly(delays_arr, a_poly)
        w_roots = np.roots(p[::-1])  # np.roots expects high-to-low coefficient order
        poles = np.array(
            [1.0 / w for w in w_roots if np.abs(w) > 1e-14], dtype=np.complex128
        )
    else:
        raise ValueError("mode must be 'eig', 'roots', or 'polyeig'")

    poles, is_conjugate_pair, non_paired = reduce_conjugate_pairs(poles)
    residues, direct, undriven, eigenvectors = _dss_to_res_direct(
        poles, delays_arr, a_mat, b_mat, c_mat, d_mat
    )

    meta_data: dict[str, Any] = {
        "nonPairedPoles": non_paired,
        "undrivenResidues": undriven,
        "eigenvectors": eigenvectors,
    }
    return residues, poles, direct, is_conjugate_pair, meta_data
