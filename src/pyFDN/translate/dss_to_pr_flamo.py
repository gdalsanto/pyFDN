"""FLAMO-only modal decomposition entry point (no ZFilter dependency here)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.flamo_runtime_probe import probe_flamo_runtime
from pyFDN.auxiliary.math import det_polynomial, poly_degree
from pyFDN.translate.dss_to_flamo import dss_to_flamo


class _ConstantMatrixProbe:
    def __init__(self, matrix: ArrayLike):
        mat = np.asarray(matrix, dtype=np.complex128)
        if mat.ndim == 0:
            mat = mat.reshape(1, 1)
        if mat.ndim == 1:
            mat = np.diag(mat)
        if mat.ndim != 2:
            raise ValueError(f"Constant probe expects 2-D matrix, got {mat.shape}")
        self._mat = mat
        self.output_channels, self.input_channels = mat.shape

    def at(self, z: complex) -> np.ndarray:
        return self._mat

    def der(self, z: complex) -> np.ndarray:
        return np.zeros_like(self._mat)


class _PolynomialMatrixProbe:
    """Polynomial matrix in z^-1 with coeff shape (n_out, n_in, order)."""

    def __init__(self, coeffs: np.ndarray):
        arr = np.asarray(coeffs, dtype=np.complex128)
        if arr.ndim != 3:
            raise ValueError(f"Polynomial probe expects 3-D coeffs, got {arr.shape}")
        self.coeffs = arr
        self.output_channels, self.input_channels, self.order = arr.shape

    def at(self, z: complex) -> np.ndarray:
        k = np.arange(self.order, dtype=np.float64)
        z_pow = np.power(z, -k).reshape(1, 1, -1)
        return np.sum(self.coeffs * z_pow, axis=2)

    def der(self, z: complex) -> np.ndarray:
        k = np.arange(self.order, dtype=np.float64)
        dz_pow = (-k * np.power(z, -k - 1)).reshape(1, 1, -1)
        return np.sum(self.coeffs * dz_pow, axis=2)


class _PassthroughProbe:
    """Probe adapter for objects that already expose at/der."""

    def __init__(self, obj: Any):
        self.obj = obj
        out_ch = getattr(obj, "output_channels", None)
        in_ch = getattr(obj, "input_channels", None)
        if out_ch is None or in_ch is None:
            val = np.asarray(obj.at(1.0 + 0j), dtype=np.complex128)
            if val.ndim == 1:
                val = np.diag(val)
            if val.ndim != 2:
                raise ValueError("Passthrough probe requires 2-D .at(z) output")
            out_ch, in_ch = val.shape
        self.output_channels = int(out_ch)
        self.input_channels = int(in_ch)

    def at(self, z: complex) -> np.ndarray:
        val = np.asarray(self.obj.at(z), dtype=np.complex128)
        if val.ndim == 1:
            val = np.diag(val)
        return val

    def der(self, z: complex) -> np.ndarray:
        val = np.asarray(self.obj.der(z), dtype=np.complex128)
        if val.ndim == 1:
            val = np.diag(val)
        return val


class _FlamoGraphProbe:
    """Probe adapter for FLAMO graph objects via autograd probing."""

    def __init__(self, model: Any):
        self.model = model
        h0 = np.asarray(
            probe_flamo_runtime(model, 1.0 + 0j, derivative=False), dtype=np.complex128
        )
        if h0.ndim != 2:
            raise ValueError(f"Graph probe at scalar z must be 2-D, got {h0.shape}")
        self.output_channels, self.input_channels = h0.shape

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(
            probe_flamo_runtime(self.model, z, derivative=False), dtype=np.complex128
        )

    def der(self, z: complex) -> np.ndarray:
        _, dh = probe_flamo_runtime(self.model, z, derivative=True)
        return np.asarray(dh, dtype=np.complex128)


class _InverseProbe:
    """Inverse transfer probe: H^-1 and derivative -(H^-1 H' H^-1)."""

    def __init__(self, base_probe: Any):
        self.base_probe = base_probe
        self.input_channels = base_probe.output_channels
        self.output_channels = base_probe.input_channels

    def at(self, z: complex) -> np.ndarray:
        return np.linalg.inv(np.asarray(self.base_probe.at(z), dtype=np.complex128))

    def der(self, z: complex) -> np.ndarray:
        h = np.asarray(self.base_probe.at(z), dtype=np.complex128)
        dh = np.asarray(self.base_probe.der(z), dtype=np.complex128)
        h_inv = np.linalg.inv(h)
        return -h_inv @ dh @ h_inv


class _DelayInverseProbe:
    """Diagonal delay inverse probe: diag(z^m)."""

    def __init__(self, delays: ArrayLike):
        self.delays = np.asarray(delays, dtype=np.float64).ravel()
        n = int(self.delays.size)
        self.input_channels = n
        self.output_channels = n

    def at(self, z: complex) -> np.ndarray:
        return np.diag(np.power(z, self.delays)).astype(np.complex128)

    def der(self, z: complex) -> np.ndarray:
        return np.diag(self.delays * np.power(z, self.delays - 1.0)).astype(np.complex128)


class _SeriesProbe:
    """Series composition: H(z) = H_left(z) @ H_right(z)."""

    def __init__(self, left: Any, right: Any):
        self.left = left
        self.right = right
        left_in = int(getattr(left, "input_channels"))
        right_out = int(getattr(right, "output_channels"))
        if left_in != right_out:
            raise ValueError(
                "Series probe dimension mismatch: "
                f"left input={left_in}, right output={right_out}"
            )
        self.input_channels = int(getattr(right, "input_channels"))
        self.output_channels = int(getattr(left, "output_channels"))

    def at(self, z: complex) -> np.ndarray:
        h_l = np.asarray(self.left.at(z), dtype=np.complex128)
        h_r = np.asarray(self.right.at(z), dtype=np.complex128)
        return h_l @ h_r

    def der(self, z: complex) -> np.ndarray:
        h_l = np.asarray(self.left.at(z), dtype=np.complex128)
        dh_l = np.asarray(self.left.der(z), dtype=np.complex128)
        h_r = np.asarray(self.right.at(z), dtype=np.complex128)
        dh_r = np.asarray(self.right.der(z), dtype=np.complex128)
        return dh_l @ h_r + h_l @ dh_r


class _DifferenceProbe:
    """Difference composition: H(z) = A(z) - B(z)."""

    def __init__(self, minuend: Any, subtrahend: Any):
        self.minuend = minuend
        self.subtrahend = subtrahend
        a_out = int(getattr(minuend, "output_channels"))
        a_in = int(getattr(minuend, "input_channels"))
        b_out = int(getattr(subtrahend, "output_channels"))
        b_in = int(getattr(subtrahend, "input_channels"))
        if (a_out, a_in) != (b_out, b_in):
            raise ValueError(
                "Difference probe dimension mismatch: "
                f"A=({a_out},{a_in}) B=({b_out},{b_in})"
            )
        self.output_channels = a_out
        self.input_channels = a_in

    def at(self, z: complex) -> np.ndarray:
        a = np.asarray(self.minuend.at(z), dtype=np.complex128)
        b = np.asarray(self.subtrahend.at(z), dtype=np.complex128)
        return a - b

    def der(self, z: complex) -> np.ndarray:
        da = np.asarray(self.minuend.der(z), dtype=np.complex128)
        db = np.asarray(self.subtrahend.der(z), dtype=np.complex128)
        return da - db


class _ReciprocalArgumentProbe:
    """Argument transform probe: H(1/z)."""

    def __init__(self, base_probe: Any):
        self.base_probe = base_probe
        self.input_channels = int(getattr(base_probe, "input_channels"))
        self.output_channels = int(getattr(base_probe, "output_channels"))

    def at(self, z: complex) -> np.ndarray:
        inv_z = 1.0 / z
        return np.asarray(self.base_probe.at(inv_z), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        inv_z = 1.0 / z
        d_base = np.asarray(self.base_probe.der(inv_z), dtype=np.complex128)
        return -(d_base / (z**2))


def _matrix_delay_units(poly_mat: np.ndarray) -> int:
    arr = np.asarray(poly_mat, dtype=np.complex128)
    if arr.ndim != 3:
        return 0
    n_out, n_in, order = arr.shape
    if n_out == n_in:
        try:
            det_poly = det_polynomial(arr, "z^-1")
            return max(int(poly_degree(det_poly, "z^-1")), 0)
        except Exception:
            return max(order - 1, 0)
    return max(order - 1, 0)


def _to_probe(
    value: Any,
    *,
    name: str,
    delay_units_hint: int | None = None,
) -> tuple[Any, int]:
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value)
        if arr.ndim <= 2:
            return _ConstantMatrixProbe(arr), 0
        if arr.ndim == 3:
            return _PolynomialMatrixProbe(arr), _matrix_delay_units(arr)
        raise ValueError(f"{name} numeric input must be 2-D or 3-D, got {arr.shape}")

    if hasattr(value, "at") and hasattr(value, "der"):
        return _PassthroughProbe(value), int(delay_units_hint or 0)

    if delay_units_hint is None:
        warnings.warn(
            f"{name} was passed as a FLAMO graph without delay_units_hint; "
            "assuming 0 matrix delay units for pole initialization.",
            stacklevel=2,
        )
    return _FlamoGraphProbe(value), int(delay_units_hint or 0)


def _as_module_list(node: Any) -> list[Any]:
    """Return modules in processing order for a FLAMO node/series."""
    try:
        modules = list(node)
    except Exception:
        return [node]
    if len(modules) == 0:
        return [node]
    return modules


def _compose_module_chain_probe(
    modules: list[Any], *, identity_dim: int | None = None
) -> Any:
    """
    Compose a FLAMO module chain into a single probe.

    For modules [m1, m2, ..., mk], returns H = Hk @ ... @ H2 @ H1.
    """
    if len(modules) == 0:
        if identity_dim is None:
            raise ValueError("Empty module chain requires identity_dim.")
        return _ConstantMatrixProbe(np.eye(identity_dim, dtype=np.complex128))

    probe: Any = _FlamoGraphProbe(modules[0])
    for module in modules[1:]:
        probe = _SeriesProbe(_FlamoGraphProbe(module), probe)
    return probe


def _is_delay_like_module(module: Any) -> bool:
    name = module.__class__.__name__.lower()
    if "delay" not in name:
        return False
    # Delay modules expose sample conversion helper in FLAMO dsp.
    return hasattr(module, "s2sample")


def _extract_flamo_recursion_probes(
    model: Any,
    delays: np.ndarray,
) -> tuple[Any, Any, Any, Any, Any]:
    """
    Extract B/C/D probes and loop probes from a dss_to_flamo-like graph.

    Expected core topology:
        Parallel(sum_output=True):
            branchA = Series(input_gain, Recursion(feedforward, feedback), output_gain)
            branchB = direct gain path
    """
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model

    if not hasattr(core, "branchA") or not hasattr(core, "branchB"):
        raise ValueError(
            "flamo_to_pr expects a FLAMO core with branchA/branchB "
            "(e.g., produced by dss_to_flamo)."
        )

    fdn_branch = core.branchA
    direct_branch = core.branchB
    fdn_modules = _as_module_list(fdn_branch)

    rec_indices = [
        i
        for i, mod in enumerate(fdn_modules)
        if hasattr(mod, "feedforward") and hasattr(mod, "feedback")
    ]
    if len(rec_indices) != 1:
        raise ValueError(
            "Could not locate a unique Recursion block in branchA. "
            "Expected dss_to_flamo branch layout."
        )

    rec_idx = rec_indices[0]
    recursion = fdn_modules[rec_idx]

    n = int(np.asarray(delays).size)
    in_modules = fdn_modules[:rec_idx]
    out_modules = fdn_modules[rec_idx + 1 :]
    b_probe = _compose_module_chain_probe(in_modules, identity_dim=n)
    c_probe = _compose_module_chain_probe(out_modules, identity_dim=n)
    direct_probe = _compose_module_chain_probe(_as_module_list(direct_branch))

    ff_modules = _as_module_list(recursion.feedforward)
    if len(ff_modules) == 0 or not _is_delay_like_module(ff_modules[0]):
        raise ValueError(
            "Recursion feedforward must start with a FLAMO delay module "
            "for flamo_to_pr decomposition."
        )
    post_delay_modules = ff_modules[1:]
    if len(post_delay_modules) == 0:
        fwd_inv_probe = _ConstantMatrixProbe(np.eye(n, dtype=np.complex128))
    else:
        post_probe = _compose_module_chain_probe(post_delay_modules, identity_dim=n)
        fwd_inv_probe = _InverseProbe(post_probe)

    fb_probe = _compose_module_chain_probe(_as_module_list(recursion.feedback))
    return b_probe, c_probe, direct_probe, fwd_inv_probe, fb_probe


def _rcond(mat: np.ndarray) -> float:
    try:
        cond = np.linalg.cond(mat)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(cond) or cond == 0:
        return 0.0
    return float(1.0 / cond)


def _adjugate(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.complex128)
    m, n = arr.shape
    if m != n:
        raise ValueError("Adjugate expects a square matrix.")
    if n < 2:
        return np.ones((1, 1), dtype=np.complex128)
    u, s, vh = np.linalg.svd(arr, full_matrices=True)
    v = vh.conj().T
    s_ex = np.ones(n, dtype=np.complex128)
    for i in range(n):
        s_ex[i] = np.prod(np.delete(s, i)) if n > 1 else 1.0
    det_uv = np.linalg.det(u @ v.conj().T)
    return det_uv * ((v * s_ex.reshape(1, -1)) @ u.conj().T)


@dataclass
class _FDNLoopFlamo:
    delays: np.ndarray
    forward_inv_probe: Any
    feedback_probe: Any
    feedback_inv_probe: Any | None
    number_of_delay_units: int
    number_of_matrix_delays: int

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.float64).ravel()
        self.n = int(self.delays.size)
        self.delay_inv_probe = _DelayInverseProbe(self.delays)
        self.forward_probe = _SeriesProbe(self.delay_inv_probe, self.forward_inv_probe)
        self.loop_probe = _DifferenceProbe(self.forward_probe, self.feedback_probe)

        feedback_inv_base = (
            self.feedback_inv_probe
            if self.feedback_inv_probe is not None
            else _InverseProbe(self.feedback_probe)
        )
        rev_forward = _SeriesProbe(
            self.delay_inv_probe,
            _ReciprocalArgumentProbe(_InverseProbe(self.forward_inv_probe)),
        )
        rev_feedback = _ReciprocalArgumentProbe(feedback_inv_base)
        self.loop_rev_probe = _DifferenceProbe(rev_forward, rev_feedback)

    def at(self, z: complex) -> np.ndarray:
        return self.loop_probe.at(z)

    def der(self, z: complex) -> np.ndarray:
        return self.loop_probe.der(z)

    def at_rev(self, z: complex) -> np.ndarray:
        return self.loop_rev_probe.at(z)

    def inverse_newton_step(self, z: complex) -> complex:
        p = self.at(z)
        dp = self.der(z)
        try:
            newton = np.trace(np.linalg.solve(p, dp))
        except np.linalg.LinAlgError:
            return np.inf + 0j
        return newton + self.number_of_matrix_delays / z


def _pole_quality(poles: np.ndarray, loop: _FDNLoopFlamo) -> np.ndarray:
    quality = np.zeros_like(poles, dtype=np.float64)
    for i, pole in enumerate(np.asarray(poles, dtype=np.complex128).ravel()):
        m = loop.at_rev(1.0 / pole) if np.abs(pole) > 1 else loop.at(pole)
        if np.ndim(m) == 0:
            q = float(np.abs(m))
        else:
            q = _rcond(np.asarray(m, dtype=np.complex128))
        if np.max(np.abs(np.asarray(m))) > 1e10:
            q = 1e10
        quality[i] = q
    return quality


def _sort_by(a: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ind = np.argsort(key)
    return a[ind], ind


def _compute_deflation(
    it: int,
    poles: np.ndarray,
    inv_newton_step: complex,
    *,
    deflation_type: str,
    number_of_neighbors: int,
    deflation_max_error: float,
    steps: int,
) -> tuple[complex, bool]:
    pole = poles[it]
    if deflation_type == "fullDeflation":
        neighbor_distance = pole - poles
        neighbor_distance[it] = 1.0 / np.finfo(float).eps
        return np.sum(1.0 / neighbor_distance), True
    if deflation_type == "noDeflation":
        return 0.0 + 0.0j, False
    if deflation_type != "neighborDeflation":
        raise ValueError(f"Unknown deflation type: {deflation_type}")

    n_poles = poles.size
    if steps == 1:
        neighbor_deflation = 0.0 + 0.0j
        factor_nonneighbor = (n_poles - 1) / 2.0
    else:
        n_neigh = int(max(0, min(number_of_neighbors, n_poles - 1)))
        if n_neigh % 2 != 0:
            n_neigh -= 1
        if n_neigh <= 0:
            neighbor_deflation = 0.0 + 0.0j
            factor_nonneighbor = (n_poles - 1) / 2.0
        else:
            offsets = np.concatenate(
                [np.arange(-n_neigh // 2, 0), np.arange(1, n_neigh // 2 + 1)]
            )
            idx = (it + offsets) % n_poles
            neighbor_deflation = np.sum(1.0 / (pole - poles[idx]))
            factor_nonneighbor = (n_poles - n_neigh - 1) / 2.0

    equi_deflation = np.conj(pole) * factor_nonneighbor
    deflation = neighbor_deflation + equi_deflation
    if steps != 1 and np.abs(inv_newton_step - deflation) < deflation_max_error:
        return _compute_deflation(
            it,
            poles,
            inv_newton_step,
            deflation_type="fullDeflation",
            number_of_neighbors=number_of_neighbors,
            deflation_max_error=deflation_max_error,
            steps=steps,
        )
    return deflation, False


def _refine_pole_positions(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    *,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    poles, _ = _sort_by(poles, np.angle(poles))
    n_poles = poles.size

    step_counter = 0
    exact_counter = 0
    record_poles: list[np.ndarray] = [poles.copy()]

    number_of_neighbors = int(round(n_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality(poles, loop)
    quality_last = quality.copy()
    current_deflation = deflation_type

    if verbose:
        print(
            f"Ehrlich-Aberth Iteration with {n_poles} poles and a maximum of "
            f"{maximum_iterations} iterations"
        )

    for steps in range(1, maximum_iterations + 1):
        if current_deflation == "neighborDeflation":
            poles, sort_ind = _sort_by(poles, np.angle(poles))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        non_converged = np.where(quality > quality_threshold)[0]
        if non_converged.size < n_poles / 10.0:
            current_deflation = "fullDeflation"

        for it in non_converged:
            if quality[it] <= quality_threshold:
                continue
            step_counter += 1
            inv_newton = loop.inverse_newton_step(poles[it])
            deflation, is_exact = _compute_deflation(
                it,
                poles,
                inv_newton,
                deflation_type=current_deflation,
                number_of_neighbors=number_of_neighbors,
                deflation_max_error=deflation_max_error,
                steps=steps,
            )
            denom = inv_newton - deflation
            if not np.isfinite(denom) or np.abs(denom) == 0:
                continue
            poles[it] = poles[it] - 1.0 / denom
            quality[it] = _pole_quality(np.array([poles[it]]), loop)[0]
            exact_counter += int(is_exact)

        if verbose:
            record_poles.append(poles.copy())

        max_improvement = float(np.abs(np.max(quality_last - quality)))
        if max_improvement < quality_threshold:
            if verbose:
                print("No further improvement possible")
            break
        if verbose:
            print(
                f"Iteration: {steps}, Maximum Improvement: {max_improvement}, "
                f"Worst Pole Quality: {np.max(quality)}, "
                f"Number of Non-converged Poles: {non_converged.size}"
            )
        quality_last = quality.copy()

    if verbose:
        print(f"Number of Exact Deflations: {exact_counter}")

    meta = {
        "stepCounter": int(step_counter),
        "exactCounter": int(exact_counter),
        "recordNeighborDeflation": [],
        "recordNewton": [],
        "recordPoles": np.asarray(record_poles, dtype=np.complex128),
    }
    return poles, quality, meta


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


def _evaluate_direct_term(direct: Any) -> np.ndarray:
    if isinstance(direct, (np.ndarray, list, tuple)):
        return np.asarray(direct, dtype=np.complex128)
    probe, _ = _to_probe(direct, name="D")
    return np.asarray(probe.at(1.0 + 0j), dtype=np.complex128)


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    b_probe: Any,
    c_probe: Any,
    direct: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size

    b_ref = np.asarray(b_probe.at(1.0 + 0j), dtype=np.complex128)
    c_ref = np.asarray(c_probe.at(1.0 + 0j), dtype=np.complex128)
    n_in = b_ref.shape[1]
    n_out = c_ref.shape[0]
    n = loop.n

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
        b = np.asarray(b_probe.at(pole), dtype=np.complex128)
        c = np.asarray(c_probe.at(pole), dtype=np.complex128)
        l = np.asarray(loop.at(pole), dtype=np.complex128)
        adj_p = _adjugate(l)

        r_nom[it, :, :] = c @ adj_p @ b

        u, s, vh = np.linalg.svd(adj_p, full_matrices=False)
        s1 = s[0] if s.size else 0.0
        denom = np.sqrt(r_den[it])
        if np.abs(denom) > 0 and np.isfinite(denom):
            eig_right[:, it] = u[:, 0] * np.sqrt(s1) / denom
            eig_left[:, it] = vh.conj().T[:, 0] * np.conj(np.sqrt(s1)) / np.conj(denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)
    direct_term = _evaluate_direct_term(direct)
    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, direct_term, undriven, eigenvectors


def flamo_to_pr(
    model: Any,
    delays: ArrayLike,
    *,
    inverse_matrix: Any | None = None,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = 0,
    absorption_delay_units: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Poles/residues directly from a FLAMO model.

    The model is expected to be built from :func:`dss_to_flamo` (or follow the
    same topology), from which the recursion loop and B/C/D branches are
    extracted for decomposition.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    b_probe, c_probe, direct_probe, fwd_inv_probe, fb_probe = _extract_flamo_recursion_probes(
        model, delays_arr
    )
    fb_delay_units_i = int(feedback_delay_units or 0)
    fwd_delay_units_i = int(absorption_delay_units or 0)

    fb_inv_probe = None
    if inverse_matrix is not None:
        fb_inv_probe, _ = _to_probe(
            inverse_matrix,
            name="inverse_matrix",
            delay_units_hint=0,
        )

    if quality_threshold is None:
        quality_threshold = 1000.0 * np.finfo(float).eps

    loop = _FDNLoopFlamo(
        delays=delays_arr,
        forward_inv_probe=fwd_inv_probe,
        feedback_probe=fb_probe,
        feedback_inv_probe=fb_inv_probe,
        number_of_matrix_delays=fb_delay_units_i,
        number_of_delay_units=int(np.sum(delays_arr) + fwd_delay_units_i + fb_delay_units_i),
    )

    n_poles = int(loop.number_of_delay_units)
    if n_poles <= 0:
        raise ValueError("Computed number_of_poles must be positive.")

    # Pole initialization
    pole_angles = np.linspace(0.0, 2.0 * np.pi, n_poles, endpoint=False)
    poles = np.exp(1j * pole_angles)

    poles, quality, meta_refine = _refine_pole_positions(
        poles,
        loop,
        quality_threshold=float(quality_threshold),
        maximum_iterations=int(maximum_iterations),
        deflation_type=str(deflation_type),
        verbose=bool(verbose),
    )

    meta_data: dict[str, Any] = dict(meta_refine)
    meta_data["refinedPoles"] = poles.copy()

    if reject_unstable_poles:
        stable = np.abs(poles) <= 1.0
        poles = poles[stable]
        quality = quality[stable]

    is_converged = quality < float(quality_threshold) * 1000.0
    poles = poles[is_converged]
    meta_data["convergedPoles"] = poles.copy()

    if poles.size != n_poles:
        warnings.warn(
            f"Some poles did not converge: {poles.size} instead of {n_poles}",
            stacklevel=2,
        )

    poles, is_conjugate, non_paired = _reduce_conjugate_pairs(poles)
    meta_data["nonPairedPoles"] = non_paired
    if verbose:
        final_count = int(np.sum(is_conjugate.astype(int) + 1))
        print(f"Final number of poles are: {final_count} of possible {n_poles}")

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, b_probe, c_probe, direct_probe
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors

    return residues, poles, direct, is_conjugate, meta_data


def dss_to_pr_flamo(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    inverse_matrix: Any | None = None,
    deflation_type: str = "fullDeflation",
    absorption_filters: Any | None = None,
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = 0,
    absorption_delay_units: int | None = 0,
    Fs: float = 1.0,
    nfft: int = 2**16,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    DSS -> FLAMO -> PR wrapper.

    Converts delay state-space to a FLAMO core using :func:`dss_to_flamo`, then
    decomposes that FLAMO model via :func:`flamo_to_pr`.
    """
    if absorption_filters is not None:
        raise ValueError(
            "dss_to_pr_flamo no longer accepts absorption_filters directly. "
            "Build a FLAMO model with dss_to_flamo(post_delay_module=...) and "
            "use flamo_to_pr(model, delays, ...)."
        )

    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    if not all(isinstance(v, (np.ndarray, list, tuple)) for v in (A, B, C, D)):
        raise TypeError(
            "dss_to_pr_flamo expects numeric DSS matrices (A,B,C,D). "
            "For existing FLAMO graph models, use flamo_to_pr(model, delays, ...)."
        )

    model = dss_to_flamo(
        A=np.asarray(A, dtype=np.float64),
        B=np.asarray(B, dtype=np.float64),
        C=np.asarray(C, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64),
        m=delays_arr,
        Fs=float(Fs),
        nfft=int(nfft),
        device=device,
        shell=False,
    )

    return flamo_to_pr(
        model,
        delays_arr,
        inverse_matrix=inverse_matrix,
        deflation_type=deflation_type,
        reject_unstable_poles=reject_unstable_poles,
        quality_threshold=quality_threshold,
        maximum_iterations=maximum_iterations,
        verbose=verbose,
        feedback_delay_units=feedback_delay_units,
        absorption_delay_units=absorption_delay_units,
    )

