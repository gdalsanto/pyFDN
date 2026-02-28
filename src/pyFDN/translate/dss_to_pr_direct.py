"""Direct DSS modal decomposition (dss2pr_direct-style, numeric only)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.translate.dss_to_ss import dss_to_ss


def _adjugate(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.complex128)
    m, n = arr.shape
    if m != n:
        raise ValueError("Adjugate expects a square matrix")
    if n < 2:
        return np.ones((1, 1), dtype=np.complex128)
    u, s, vh = np.linalg.svd(arr, full_matrices=True)
    v = vh.conj().T
    s_ex = np.ones(n, dtype=np.complex128)
    for i in range(n):
        s_ex[i] = np.prod(np.delete(s, i)) if n > 1 else 1.0
    det_uv = np.linalg.det(u @ v.conj().T)
    return det_uv * ((v * s_ex.reshape(1, -1)) @ u.conj().T)


def _reduce_conjugate_pairs(
    poles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    lhs = np.column_stack([np.real(poles), np.imag(poles)])
    rhs = np.column_stack([np.real(poles), -np.imag(poles)])
    dist = np.linalg.norm(lhs[:, None, :] - rhs[None, :, :], axis=2)
    pair_index = np.argmin(dist, axis=1)

    pair_type = np.zeros(n_poles, dtype=int)
    for it in range(n_poles):
        if pair_type[it] != 0:
            continue
        if it == pair_index[it]:
            pair_type[it] = 1
        elif it == pair_index[pair_index[it]]:
            pair_type[it] = 2
            pair_type[pair_index[it]] = 3
        else:
            pair_type[it] = -1

    if np.any(pair_type == -1):
        warnings.warn(f"{np.sum(pair_type == -1)} poles could not be paired", stacklevel=2)

    is_conjugate = np.ones(n_poles, dtype=bool)
    is_conjugate[pair_type == 1] = False
    non_paired = poles[pair_type == -1]
    select = (pair_type == 1) | (pair_type == 2) | (pair_type == -1)
    poles_out = poles[select]
    poles_out = np.real(poles_out) + 1j * np.abs(np.imag(poles_out))
    return poles_out, is_conjugate[select], non_paired


@dataclass
class _SimpleLoop:
    delays: np.ndarray
    feedback: np.ndarray

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.float64).ravel()
        self.feedback = np.asarray(self.feedback, dtype=np.complex128)
        self.n = int(self.delays.size)
        if self.feedback.shape != (self.n, self.n):
            raise ValueError(
                "Feedback dimensions must match number of delays: "
                f"expected ({self.n},{self.n}), got {self.feedback.shape}"
            )

    def _delay_inv_at(self, z: complex) -> np.ndarray:
        return np.diag(np.power(z, self.delays)).astype(np.complex128)

    def _delay_inv_der(self, z: complex) -> np.ndarray:
        return np.diag(self.delays * np.power(z, self.delays - 1.0)).astype(np.complex128)

    def at(self, z: complex) -> np.ndarray:
        return self._delay_inv_at(z) - self.feedback

    def der(self, z: complex) -> np.ndarray:
        return self._delay_inv_der(z)


def _dss_to_res_direct(
    poles: np.ndarray,
    loop: _SimpleLoop,
    b_mat: np.ndarray,
    c_mat: np.ndarray,
    d_term: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    n = loop.n

    b0 = np.asarray(b_mat, dtype=np.complex128)
    c0 = np.asarray(c_mat, dtype=np.complex128)
    n_in = b0.shape[1]
    n_out = c0.shape[0]

    r_den = np.zeros(n_poles, dtype=np.complex128)
    for it, pole in enumerate(poles):
        r_den[it] = np.trace(_adjugate(loop.at(pole)) @ loop.der(pole))

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn("There are multipoles. The residues are set to zero.", stacklevel=2)
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    r_nom = np.zeros((n_poles, n_out, n_in), dtype=np.complex128)
    eig_right = np.zeros((n, n_poles), dtype=np.complex128)
    eig_left = np.zeros((n, n_poles), dtype=np.complex128)

    for it, pole in enumerate(poles):
        l = np.asarray(loop.at(pole), dtype=np.complex128)
        adj = _adjugate(l)
        r_nom[it, :, :] = c0 @ adj @ b0

        u, s, vh = np.linalg.svd(adj, full_matrices=False)
        s1 = s[0] if s.size else 0.0
        denom = np.sqrt(r_den[it])
        if np.abs(denom) > 0 and np.isfinite(denom):
            eig_right[:, it] = u[:, 0] * np.sqrt(s1) / denom
            eig_left[:, it] = vh.conj().T[:, 0] * np.conj(np.sqrt(s1)) / np.conj(denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)

    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, np.asarray(d_term, dtype=np.complex128), undriven, eigenvectors


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
    if a_mat.shape != (n, n):
        raise ValueError(f"A must have shape ({n},{n}), got {a_mat.shape}")

    loop = _SimpleLoop(delays=delays_arr, feedback=a_mat)

    mode_l = str(mode).lower()
    a_poly = np.real_if_close(a_mat, tol=1000)
    if np.iscomplexobj(a_poly) and np.max(np.abs(np.imag(a_poly))) > 1e-12:
        raise ValueError(
            "roots/polyeig mode currently requires a real-valued static feedback matrix."
        )
    a_poly = np.asarray(np.real(a_poly), dtype=float)

    if mode_l == "eig":
        aa, _, _, _ = dss_to_ss(delays_arr, a_mat)
        poles = np.linalg.eigvals(aa)
    elif mode_l == "roots":
        p = general_char_poly(delays_arr, a_poly)
        poles = np.roots(p)
    elif mode_l == "polyeig":
        warnings.warn(
            "mode='polyeig' is currently mapped to roots-based extraction in dss_to_pr_direct.",
            stacklevel=2,
        )
        p = general_char_poly(delays_arr, a_poly)
        poles = np.roots(p)
    else:
        raise ValueError("mode must be 'eig', 'roots', or 'polyeig'")

    poles, is_conjugate_pair, non_paired = _reduce_conjugate_pairs(poles)
    residues, direct, undriven, eigenvectors = _dss_to_res_direct(
        poles, loop, b_mat, c_mat, d_mat
    )

    meta_data: dict[str, Any] = {
        "nonPairedPoles": non_paired,
        "undrivenResidues": undriven,
        "eigenvectors": eigenvectors,
    }
    return residues, poles, direct, is_conjugate_pair, meta_data

