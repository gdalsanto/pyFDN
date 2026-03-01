"""FLAMO-only modal decomposition entry point (no ZFilter dependency here)."""

from __future__ import annotations

from collections import OrderedDict
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import torch

from pyFDN.auxiliary.poles import reduce_conjugate_pairs
from pyFDN.translate.dss_to_flamo import dss_to_flamo


class _IdentityProbe:
    """Constant identity matrix probe used for empty module chains."""

    def __init__(self, size: int):
        self.size = int(size)
        self.output_channels = self.size
        self.input_channels = self.size
        self._eye = np.eye(self.size, dtype=np.complex128)

    def at_z(self, z: complex) -> np.ndarray:
        return self._eye

    def at_w(self, w: complex) -> np.ndarray:
        return self._eye

    def at(self, z: complex) -> np.ndarray:
        return self.at_z(z)

    def der(self, z: complex) -> np.ndarray:
        return np.zeros_like(self._eye)


def _infer_model_device(model: Any) -> torch.device:
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            first = next(params())
            return first.device
        except Exception:
            pass
    return torch.device("cpu")


def _infer_model_complex_dtype(model: Any) -> torch.dtype:
    dt = getattr(model, "dtype", None)
    if dt in (torch.float16, torch.float32):
        return torch.complex64
    return torch.complex128


def _as_torch_complex_scalar(z: complex, *, model: Any) -> torch.Tensor:
    return torch.tensor(
        complex(np.asarray(z, dtype=np.complex128)),
        device=_infer_model_device(model),
        dtype=_infer_model_complex_dtype(model),
    )


class _FlamoGraphProbe:
    """Probe adapter for FLAMO graph objects via probe(z) / probe_w(w)."""

    def __init__(self, model: Any):
        self.model = model
        h0 = np.asarray(self.at_z(1.0 + 0j), dtype=np.complex128)
        if h0.ndim != 2:
            raise ValueError(f"Graph probe at scalar z must be 2-D, got {h0.shape}")
        self.output_channels, self.input_channels = h0.shape

    def at_z(self, z: complex) -> np.ndarray:
        """H(z) via FLAMO model.probe(z)."""
        z_t = _as_torch_complex_scalar(z, model=self.model)
        out = self.model.probe(z_t)
        if isinstance(out, tuple):
            if len(out) == 0:
                raise RuntimeError("model.probe returned empty tuple")
            out = out[0]
        return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)

    def at_w(self, w: complex) -> np.ndarray:
        """H(1/w) via FLAMO model.probe_w(w) when available, else at_z(1/w)."""
        probe_w_fn = getattr(self.model, "probe_w", None)
        if callable(probe_w_fn):
            w_t = _as_torch_complex_scalar(w, model=self.model)
            out = probe_w_fn(w_t)
            if isinstance(out, tuple) and out:
                out = out[0]
            return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)
        return self.at_z(1.0 / w)

    def at(self, z: complex) -> np.ndarray:
        """Backward-compatible alias for at_z(z)."""
        return self.at_z(z)

    def der(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.model)
        if callable(getattr(self.model, "probe_with_derivative", None)):
            out = self.model.probe_with_derivative(z_t)
        else:
            out = self.model.probe(z_t, derivative=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError(
                "FLAMO model must expose derivative probe support "
                "(probe_with_derivative or probe(..., derivative=True))."
            )
        return np.asarray(out[1].detach().cpu().numpy(), dtype=np.complex128)


class _FlamoRecursionCharacteristicProbe:
    """Probe adapter for Recursion.probe_recursion(z) / probe_recursion_w(w) = P."""

    def __init__(self, recursion: Any):
        self.recursion = recursion
        p0 = np.asarray(self.at_z(1.0 + 0j), dtype=np.complex128)
        if p0.ndim != 2:
            raise ValueError(f"Recursion characteristic probe must be 2-D, got {p0.shape}")
        self.output_channels, self.input_channels = p0.shape

    def at_z(self, z: complex) -> np.ndarray:
        """P(z) via FLAMO Recursion.probe_recursion(z)."""
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        out = self.recursion.probe_recursion(z_t)
        if isinstance(out, tuple):
            if len(out) == 0:
                raise RuntimeError("probe_recursion returned empty tuple")
            out = out[0]
        return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)

    def at_w(self, w: complex) -> np.ndarray:
        """P(w) = P(1/z) via FLAMO Recursion.probe_recursion_w(w)."""
        if np.abs(w) < 1e-14:
            n = self.output_channels
            return np.full((n, n), np.inf + 0j, dtype=np.complex128)
        w_t = _as_torch_complex_scalar(w, model=self.recursion)
        out = self.recursion.probe_recursion_w(w_t)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)

    def at(self, z: complex) -> np.ndarray:
        """Backward-compatible alias for at_z(z)."""
        return self.at_z(z)

    def der(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        if callable(getattr(self.recursion, "probe_recursion_with_derivative", None)):
            out = self.recursion.probe_recursion_with_derivative(z_t)
        else:
            out = self.recursion.probe_recursion(z_t, derivative=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError(
                "Recursion must expose derivative characteristic probe support "
                "(probe_recursion_with_derivative or probe_recursion(..., derivative=True))."
            )
        return np.asarray(out[1].detach().cpu().numpy(), dtype=np.complex128)

    def log_det_derivative(self, z: complex) -> complex:
        """(d/dz) log det P(z) via FLAMO Recursion.log_det_derivative(z)."""
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        out = self.recursion.log_det_derivative(z_t)
        return complex(out.detach().cpu().numpy())

    def log_det_derivative_w(self, w: complex) -> complex:
        """(d/dw) log det P(w) at w=z^{-1} via FLAMO Recursion.log_det_derivative_w(w)."""
        w_t = _as_torch_complex_scalar(w, model=self.recursion)
        out = self.recursion.log_det_derivative_w(w_t)
        return complex(out.detach().cpu().numpy())


@dataclass
class _CharacteristicDecomposition:
    """H(z)=C(z)P(z)^{-1}B(z)+D(z) with probes for B,C,D."""

    p_probe: Any
    f_probe: Any
    in_probe: Any
    out_probe: Any
    direct_probe: Any


def _decomposition_to_public_dict(
    decomposition: _CharacteristicDecomposition,
) -> dict[str, Any]:
    """Public dict: P, F (feedforward), B (input path), C, D probes. H = C P^{-1} F B + D."""
    return {
        "P": decomposition.p_probe,
        "F": decomposition.f_probe,
        "B": decomposition.in_probe,
        "C": decomposition.out_probe,
        "D": decomposition.direct_probe,
    }


def _as_module_list(node: Any) -> list[Any]:
    """Return modules in processing order for a FLAMO node/series."""
    try:
        modules = list(node)
    except Exception:
        return [node]
    if len(modules) == 0:
        return [node]
    return modules


@dataclass
class FlamoDecompositionForPR:
    """
    Decomposition of a FLAMO model into small subgraphs for poles/residues.
    All subgraph fields are FLAMO modules (with .probe(z)); None means identity.
    """

    recursion_module: Any
    delays: np.ndarray
    in_subgraph: Any | None
    f_subgraph: Any
    out_subgraph: Any | None
    direct_subgraph: Any


def _series_slice_to_subgraph(series: Any, start: int, end: int) -> Any | None:
    """
    Return the slice of a FLAMO Series as a subgraph (same module refs).
    Returns None for empty slice, single module ref, or new Series(OrderedDict(slice)).
    """
    n = end - start
    if n <= 0:
        return None
    if n == 1:
        return series[start]
    try:
        from flamo.processor import system as flamo_system  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FLAMO system.Series is required for multi-module subgraphs."
        ) from exc
    items = list(series._modules.items())[start:end]
    return flamo_system.Series(OrderedDict(items))


def flamo_decompose_for_pr(model: Any) -> FlamoDecompositionForPR:
    """
    Decompose a FLAMO model into the subgraphs needed for poles/residues.

    Expects core with branchA (Series of input_gain, Recursion(feedforward, feedback), output_gain)
    and branchB (direct path). Returns small FLAMO subgraphs (no probing); pass the result
    to :func:`flamo_to_pr` as ``decomposition=...``.
    """
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    if not hasattr(core, "branchA") or not hasattr(core, "branchB"):
        raise ValueError(
            "Model core must have branchA and branchB "
            "(e.g., from dss_to_flamo)."
        )
    fdn_branch = core.branchA
    direct_branch = core.branchB
    fdn_modules = _as_module_list(fdn_branch)
    recs = [
        m
        for m in fdn_modules
        if hasattr(m, "feedforward") and hasattr(m, "feedback")
    ]
    if len(recs) != 1:
        raise ValueError("branchA must contain exactly one Recursion.")
    recursion_module = recs[0]
    rec_idx = fdn_modules.index(recursion_module)
    n_fdn = len(fdn_modules)
    delays = _delays_from_recursion(recursion_module)
    if delays.size == 0:
        raise ValueError("Recursion has no delays (empty delay module).")

    in_subgraph = _series_slice_to_subgraph(fdn_branch, 0, rec_idx)
    out_subgraph = _series_slice_to_subgraph(fdn_branch, rec_idx + 1, n_fdn)
    return FlamoDecompositionForPR(
        recursion_module=recursion_module,
        delays=delays,
        in_subgraph=in_subgraph,
        f_subgraph=recursion_module.feedforward,
        out_subgraph=out_subgraph,
        direct_subgraph=direct_branch,
    )


def _delays_from_recursion(recursion_module: Any) -> np.ndarray:
    """
    Return 1D array of delay lengths in samples from the recursion's feedforward.
    Looks at the delay module in the recursion and sums the number of delays (per line).
    """
    ff = recursion_module.feedforward
    delay_mod = getattr(ff, "delay", ff)
    param = delay_mod.param
    if callable(getattr(delay_mod, "map", None)):
        sec = delay_mod.map(param)
    else:
        sec = param
    samples = delay_mod.s2sample(sec)
    out = np.asarray(samples.detach().cpu().numpy(), dtype=np.float64).ravel()
    return np.asarray(np.round(out), dtype=int)


def _subgraph_to_probe(subgraph: Any | None, *, identity_dim: int) -> Any:
    """Wrap a FLAMO subgraph (or None) as a probe with .at_z(z) / .at_w(w) -> ndarray."""
    if subgraph is None:
        return _IdentityProbe(identity_dim)
    return _FlamoGraphProbe(subgraph)


def _extract_flamo_recursion_probes(
    decomposition: FlamoDecompositionForPR,
) -> _CharacteristicDecomposition:
    """Build characteristic decomposition from pre-decomposed FLAMO subgraphs."""
    n = int(decomposition.delays.size)
    rec = decomposition.recursion_module

    f_probe = _FlamoGraphProbe(decomposition.f_subgraph)
    in_probe = _subgraph_to_probe(decomposition.in_subgraph, identity_dim=n)
    out_probe = _subgraph_to_probe(decomposition.out_subgraph, identity_dim=n)
    direct_probe = _FlamoGraphProbe(decomposition.direct_subgraph)

    characteristic_probe = _FlamoRecursionCharacteristicProbe(rec)
    return _CharacteristicDecomposition(
        p_probe=characteristic_probe,
        f_probe=f_probe,
        in_probe=in_probe,
        out_probe=out_probe,
        direct_probe=direct_probe,
    )


def flamo_extract_pr_decomposition(model: Any) -> dict[str, Any]:
    """
    Extract the H(z)=C P(z)^{-1}B+D probes from a FLAMO model.

    Uses :func:`flamo_decompose_for_pr` then builds probe adapters. Returns a dict
    with keys ``"P"``, ``"F"`` (feedforward), ``"B"`` (input path), ``"C"``, ``"D"``.
    For poles/residues use
    :func:`flamo_to_pr`(model) or :func:`flamo_to_pr`(decomposition=...).
    """
    decomp = flamo_decompose_for_pr(model)
    decomposition = _extract_flamo_recursion_probes(decomp)
    return _decomposition_to_public_dict(decomposition)


def _rcond(mat: np.ndarray) -> float:
    try:
        cond = np.linalg.cond(mat)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(cond) or cond == 0:
        return 0.0
    return float(1.0 / cond)


@dataclass
class _FDNLoopFlamo:
    characteristic_probe: Any

    def __post_init__(self):
        out_ch = int(getattr(self.characteristic_probe, "output_channels"))
        in_ch = int(getattr(self.characteristic_probe, "input_channels"))
        if out_ch != in_ch:
            raise ValueError(
                "Characteristic probe must be square, "
                f"got ({out_ch},{in_ch})."
            )
        self.n = out_ch

    def at_z(self, z: complex) -> np.ndarray:
        """P(z) via FLAMO characteristic probe (Recursion.probe_recursion(z))."""
        return np.asarray(self.characteristic_probe.at_z(z), dtype=np.complex128)

    def at_w(self, w: complex) -> np.ndarray:
        """P(w) = P(1/z) via FLAMO probe_recursion_w(w)."""
        return np.asarray(self.characteristic_probe.at_w(w), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        return np.asarray(self.characteristic_probe.der(z), dtype=np.complex128)

    def log_det_derivative_z(self, z: complex) -> complex:
        """Return (d/dz) log det P(z) via FLAMO log_det_derivative(z)."""
        out = self.characteristic_probe.log_det_derivative(z)
        return complex(out)

    def log_det_derivative_w(self, w: complex) -> complex:
        """Return (d/dw) log det P(1/w) via FLAMO log_det_derivative_w(w)."""
        out = self.characteristic_probe.log_det_derivative_w(w)
        return complex(out)

    def inverse_newton_step(self, z: complex, *, use_w_plane_for_small_z: bool = True) -> complex:
        """Backwards-compatible alias for z-domain Newton term."""
        return self.log_det_derivative_z(z)

    def inverse_newton_step_w(self, w: complex) -> complex:
        """Newton term directly in w-domain."""
        return self.log_det_derivative_w(w)


def _pole_quality_z(poles_z: np.ndarray, loop: _FDNLoopFlamo) -> np.ndarray:
    """Pole quality in z-domain: rcond(P(z)) for each pole z."""
    quality = np.zeros_like(poles_z, dtype=np.float64)
    for i, z in enumerate(np.asarray(poles_z, dtype=np.complex128).ravel()):
        m = loop.at_z(z)
        if np.ndim(m) == 0:
            q = float(np.abs(m))
        else:
            m = np.asarray(m, dtype=np.complex128)
            q = _rcond(m)
        
        if np.all(np.isfinite(m)) and np.max(np.abs(m)) > 1e10:
            q = 1e10
        quality[i] = q
    return quality


def _pole_quality_w(roots_w: np.ndarray, loop: _FDNLoopFlamo) -> np.ndarray:
    """
    Pole quality for roots w: rcond(P(1/w)).
    Uses z-domain (P(z) at z=1/w) when |w| > 1 for numerical stability;
    otherwise uses w-domain probing (handles small |w| via loop.at_w).
    """
    roots_w = np.asarray(roots_w, dtype=np.complex128).ravel()
    quality = np.zeros_like(roots_w, dtype=np.float64)
    for i, w in enumerate(roots_w):
        if np.abs(w) > 1:
            z = 1.0 / w
            q = _pole_quality_z(np.array([z]), loop)[0]
        else:
            m = loop.at_w(w)
            if np.ndim(m) == 0:
                q = float(np.abs(m))
            else:
                m = np.asarray(m, dtype=np.complex128)
                q = _rcond(m)
                if np.all(np.isfinite(m)) and np.max(np.abs(m)) > 1e10:
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

    # Compute neighbor deflation.
    n_poles = poles.size
    if steps == 1: # First iteration, no neighbors.
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


def _refine_pole_positions_w(
    roots_w: np.ndarray,
    loop: _FDNLoopFlamo,
    *,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
    refinement_tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    roots_w = np.asarray(roots_w, dtype=np.complex128).ravel()
    roots_w, _ = _sort_by(roots_w, np.angle(roots_w))
    n_poles = roots_w.size

    newton_step_counter = 0
    exact_counter = 0
    record_roots_w: list[np.ndarray] = [roots_w.copy()]

    number_of_neighbors = int(round(n_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality_w(roots_w, loop)
    quality_last = quality.copy()
    current_deflation = deflation_type

    if verbose:
        print(
            f"Ehrlich-Aberth Iteration in w-domain with {n_poles} poles and a maximum of "
            f"{maximum_iterations} iterations"
        )


    for iteration_counter in range(1, maximum_iterations + 1):
        if current_deflation == "neighborDeflation":
            roots_w, sort_ind = _sort_by(roots_w, np.angle(roots_w))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        roots_w_old = roots_w.copy()
        non_converged = np.where(quality > quality_threshold)[0]

        # If fewer than 10% of poles are non-converged, switch to full deflation.
        if non_converged.size < n_poles / 10.0:
            current_deflation = "fullDeflation"

        # Iterate over non-converged poles.
        for it in non_converged:
            if quality[it] <= quality_threshold:
                continue
            if quality[it] > 10000.0:
                # If pole quality is too bad, set it to a random value on the unit circle.
                roots_w[it] = np.exp(1j * np.random.uniform(0, 2 * np.pi))
                quality[it] = _pole_quality_w(np.array([roots_w[it]]), loop)[0]
                if verbose:
                    print(f"Pole {it} was at {roots_w_old[it]:.3f} and set to random value on unit circle: {roots_w[it]:.3f}")
            
            # Compute Newton step and deflation.
            newton_step_counter += 1
            w_i = roots_w[it]
            inv_newton_w = loop.inverse_newton_step_w(w_i)
            deflation, is_exact = _compute_deflation(
                it,
                roots_w,
                inv_newton_w,
                deflation_type=current_deflation,
                number_of_neighbors=number_of_neighbors,
                deflation_max_error=deflation_max_error,
                steps=iteration_counter,
            )
            denom = inv_newton_w - deflation
            if not np.isfinite(denom) or np.abs(denom) < 1e-20:
                continue
            roots_w[it] = w_i - 1.0 / denom
            quality[it] = _pole_quality_w(np.array([roots_w[it]]), loop)[0]
            exact_counter += int(is_exact)

        if verbose:
            record_roots_w.append(roots_w.copy())

        if refinement_tol is not None:
            max_step = float(np.max(np.abs(roots_w - roots_w_old)))
            if max_step < refinement_tol:
                if verbose:
                    print(f"Converged (max |Δw| = {max_step:.3e} < {refinement_tol})")
                break
        else:
            max_improvement = float(np.abs(np.max(quality_last - quality)))
            if max_improvement < quality_threshold:
                if verbose:
                    print("No further improvement possible")
                break
        if verbose:
            max_improvement = float(np.abs(np.max(quality_last - quality)))
            print(
                f"Iter: {iteration_counter}, "
                f"Max Improvement: {max_improvement:.3e}, "
                f"Worst Pole Quality: {np.max(quality):.3e}, "
                f"Number of Non-converged Poles: {non_converged.size}"
            )
        quality_last = quality.copy()

    if verbose:
        print(f"Number of Exact Deflations: {exact_counter}")
        print(f"Number of Newton Steps: {newton_step_counter}")
        print(f"Number of Poles: {n_poles}")
        print(f"Number of Non-converged Poles: {non_converged.size}")
    meta = {
        "newtonStepCounter": int(newton_step_counter),
        "iterations": int(iteration_counter),
        "exactCounter": int(exact_counter),
        "recordNeighborDeflation": [],
        "recordNewton": [],
        "recordRootsW": np.asarray(record_roots_w, dtype=np.complex128),
    }
    return roots_w, quality, meta


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    decomposition: _CharacteristicDecomposition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    n_in = int(getattr(decomposition.in_probe, "input_channels"))
    n_out = int(getattr(decomposition.out_probe, "output_channels"))
    n = loop.n

    r_den = np.zeros(n_poles, dtype=np.complex128)
    r_nom = np.zeros((n_poles, n_out, n_in), dtype=np.complex128)
    eig_right = np.zeros((n, n_poles), dtype=np.complex128)
    eig_left = np.zeros((n, n_poles), dtype=np.complex128)

    for it, pole in enumerate(poles):
        p = np.asarray(decomposition.p_probe.at(pole), dtype=np.complex128)
        dp = np.asarray(decomposition.p_probe.der(pole), dtype=np.complex128)
        f_at = np.asarray(decomposition.f_probe.at(pole), dtype=np.complex128)
        in_at = np.asarray(decomposition.in_probe.at(pole), dtype=np.complex128)
        b = f_at @ in_at
        c = np.asarray(decomposition.out_probe.at(pole), dtype=np.complex128)

        # Null vectors for simple poles:
        # P(lambda) r = 0,  l^H P(lambda) = 0
        u, s, vh = np.linalg.svd(p, full_matrices=False)
        r = vh.conj().T[:, -1]
        l = u[:, -1]

        denom = np.vdot(l, dp @ r)  # l^H (dP/dz) r
        r_den[it] = denom
        eig_right[:, it] = r
        eig_left[:, it] = l

        cr = c @ r.reshape(-1, 1)
        lh_b = np.conj(l).reshape(1, -1) @ b
        r_nom[it, :, :] = cr @ lh_b

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn("There are multipoles. The residues are set to zero.", stacklevel=2)
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)
    direct_term = np.asarray(
        decomposition.direct_probe.at(1.0 + 0j), dtype=np.complex128
    )
    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, direct_term, undriven, eigenvectors


def flamo_to_pr(
    model: Any | None = None,
    *,
    decomposition: FlamoDecompositionForPR | None = None,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float = 1e-7,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Poles/residues from a FLAMO transfer H(z)=C(z)P(z)^{-1}B(z)+D(z).

    Pass either a full FLAMO **model** (decomposition is done via
    :func:`flamo_decompose_for_pr`) or a **decomposition** returned by that
    function. Poles are refined in the w-domain (w = 1/z) then converted to z.

    For DSS matrices (A,B,C,D) use :func:`dss_to_pr_flamo` instead.
    """
    if decomposition is None:
        if model is None:
            raise ValueError("Provide model or decomposition.")
        decomposition = flamo_decompose_for_pr(model)
    delays_arr = decomposition.delays
    decomposition_obj = _extract_flamo_recursion_probes(decomposition)

    loop = _FDNLoopFlamo(characteristic_probe=decomposition_obj.p_probe)

    n_poles = int(np.sum(delays_arr))

    # n_poles = n_poles

    # Initialize on unit circle in w-domain and refine there.
    root_angles = np.linspace(0.0, 2.0 * np.pi, n_poles, endpoint=False)
    roots_w = np.exp(1j * root_angles)

    roots_w, quality, meta_refine = _refine_pole_positions_w(
        roots_w,
        loop,
        quality_threshold=float(quality_threshold),
        maximum_iterations=int(maximum_iterations),
        deflation_type=str(deflation_type),
        verbose=bool(verbose),
        refinement_tol=refinement_tol,
    )

    meta_data: dict[str, Any] = dict(meta_refine)
    refined_roots_w = roots_w.copy()
    meta_data["refinedRootsW"] = refined_roots_w
    
    is_stable = np.abs(roots_w) >= 1.0
    is_converged = quality < float(quality_threshold) * 1000.0
    
    if reject_unstable_poles:
        is_valid = is_stable & is_converged
    else:
        is_valid = is_converged
    roots_w = roots_w[is_valid]
    quality = quality[is_valid]

    if verbose:
        print(f"Number of Stable Poles: {np.sum(is_stable)}")
        print(f"Number of Converged Poles: {np.sum(is_converged)}")
        print(f"Number of Valid Poles: {np.sum(is_valid)}")

    poles = 1.0 / roots_w

    poles, is_conjugate, non_paired = reduce_conjugate_pairs(poles, verbose=verbose)
    meta_data["nonPairedPoles"] = non_paired

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, decomposition_obj
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors
    meta_data["decomposition"] = _decomposition_to_public_dict(decomposition_obj)

    return residues, poles, direct, is_conjugate, meta_data


def dss_to_pr_flamo(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float = 1e-7,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    verbose: bool = True,
    dtype: Any = None,
    Fs: float = 1.0,
    nfft: int = 2**16,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    DSS -> FLAMO -> PR in one call.

    Builds a FLAMO model with :func:`dss_to_flamo`, then runs :func:`flamo_to_pr`
    on it. For an existing FLAMO model use :func:`flamo_to_pr` directly.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
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
        dtype=dtype,
    )

    return flamo_to_pr(
        model,
        deflation_type=deflation_type,
        reject_unstable_poles=reject_unstable_poles,
        quality_threshold=quality_threshold,
        maximum_iterations=maximum_iterations,
        refinement_tol=refinement_tol,
        verbose=verbose,
    )

