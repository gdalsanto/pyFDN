"""
Arbitrary z-plane probing for FLAMO graphs.

This module provides utilities to evaluate a FLAMO graph at arbitrary complex
points ``z`` (not only unit-circle FFT bins), and to compute the derivative
``dH/dz`` at the same points.

The implementation reuses FLAMO graph semantics:
- Series: cascaded matrix product
- Parallel: summed or concatenated branch outputs
- Recursion: closed-loop solve ``(I - F B)^-1 F``

The focus is model decomposition / Newton-style workflows that need
``H(z_k)`` and ``dH/dz(z_k)`` at user-defined probe points ``z_k``.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from pyFDN.auxiliary.filters import ZFilter


LeafEvaluator = Callable[
    [Any, complex, str],
    tuple[np.ndarray, np.ndarray] | None,
]

_DFILT_DEPRECATION = (
    "dfilt_type and dfilt_parameter are deprecated (MATLAB compatibility). "
    "They will be removed in a future version."
)


def _typename(module: Any) -> str:
    return type(module).__name__


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch-like or array-like values to numpy arrays."""
    out = value
    if hasattr(out, "detach"):
        out = out.detach()
    if hasattr(out, "cpu"):
        out = out.cpu()
    if hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out)


def _as_complex_scalar(value: Any, *, default: complex) -> complex:
    if value is None:
        return complex(default)
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return complex(arr.item())
    if arr.size == 1:
        return complex(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar value, got shape {arr.shape}")


def _mapped_param(module: Any) -> np.ndarray:
    """Return mapped module parameters as numpy array."""
    param = getattr(module, "param", None)
    if param is None:
        state_getter = getattr(module, "state_dict", None)
        if callable(state_getter):
            state = state_getter()
            param = state.get("param")
    if param is None:
        raise ValueError(
            f"Module of type {_typename(module)!r} has no accessible 'param'."
        )

    map_fn = getattr(module, "map", None)
    mapped = map_fn(param) if callable(map_fn) else param
    return _to_numpy(mapped)


def _ensure_2d_matrix(arr: np.ndarray, *, path: str, what: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.complex128)
    if out.ndim != 2:
        raise ValueError(
            f"{what} at {path} must be a 2-D matrix, got shape {out.shape}."
        )
    return out


def _poly_zm1_and_der(coeff: np.ndarray, z: complex) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate polynomial in z^-1 and its derivative.

    Parameters
    ----------
    coeff:
        Coefficients along axis 0, i.e.
        ``P(z) = sum_k coeff[k] * z^{-k}``.
    z:
        Complex evaluation point.
    """
    coeff_c = np.asarray(coeff, dtype=np.complex128)
    k = np.arange(coeff_c.shape[0], dtype=np.float64)
    shape = (coeff_c.shape[0],) + (1,) * (coeff_c.ndim - 1)
    z_pow = np.power(z, -k).reshape(shape)
    p = np.sum(coeff_c * z_pow, axis=0)

    dz_pow = (-k * np.power(z, -k - 1)).reshape(shape)
    dp = np.sum(coeff_c * dz_pow, axis=0)
    return p, dp


def _series_children(module: Any) -> list[tuple[str, Any]]:
    modules = getattr(module, "_modules", None)
    if modules is None:
        return []
    if isinstance(modules, OrderedDict) or hasattr(modules, "items"):
        return list(modules.items())
    if hasattr(modules, "__iter__"):
        return [(str(i), m) for i, m in enumerate(modules)]
    return []


def _shell_core(module: Any) -> Any | None:
    core_getter = getattr(module, "get_core", None)
    if callable(core_getter):
        core = core_getter()
        if core is not None:
            return core
    return getattr(module, "_Shell__core", None) or getattr(module, "core", None)


def _shell_input_output(module: Any) -> tuple[Any | None, Any | None]:
    in_getter = getattr(module, "get_inputLayer", None)
    out_getter = getattr(module, "get_outputLayer", None)
    in_layer = in_getter() if callable(in_getter) else None
    out_layer = out_getter() if callable(out_getter) else None
    in_layer = (
        in_layer
        if in_layer is not None
        else getattr(module, "_Shell__input_layer", None) or getattr(module, "input_layer", None)
    )
    out_layer = (
        out_layer
        if out_layer is not None
        else getattr(module, "_Shell__output_layer", None) or getattr(module, "output_layer", None)
    )
    return in_layer, out_layer


def _parallel_branches(module: Any) -> tuple[Any | None, Any | None]:
    br_a = getattr(module, "branchA", None) or getattr(module, "brA", None)
    br_b = getattr(module, "branchB", None) or getattr(module, "brB", None)
    return br_a, br_b


def _recursion_paths(module: Any) -> tuple[Any | None, Any | None]:
    ff = getattr(module, "feedforward", None) or getattr(module, "fF", None)
    fb = getattr(module, "feedback", None) or getattr(module, "fB", None)
    return ff, fb


def _identity_for_module(module: Any, *, path: str) -> np.ndarray:
    n_in = getattr(module, "input_channels", None)
    n_out = getattr(module, "output_channels", None)
    if n_in is None or n_out is None:
        raise ValueError(
            f"Cannot infer identity size for empty Series at {path}. "
            "Missing input/output channel metadata."
        )
    if int(n_in) != int(n_out):
        raise ValueError(
            f"Empty Series at {path} is not square ({n_out}x{n_in}); "
            "cannot represent it as identity."
        )
    return np.eye(int(n_out), dtype=np.complex128)


def _eval_gain_like(module: Any) -> tuple[np.ndarray, np.ndarray]:
    mat = _ensure_2d_matrix(_mapped_param(module), path="leaf", what="Gain/Matrix parameter")
    return mat, np.zeros_like(mat)


def _eval_parallel_gain(module: Any) -> tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(_mapped_param(module), dtype=np.complex128).reshape(-1)
    mat = np.diag(vec)
    return mat, np.zeros_like(mat)


def _delay_samples(module: Any) -> np.ndarray:
    param = getattr(module, "param", None)
    if param is None:
        raise ValueError(f"{_typename(module)} has no 'param' attribute.")
    map_fn = getattr(module, "map", None)
    mapped = map_fn(param) if callable(map_fn) else param

    s2sample_fn = getattr(module, "s2sample", None)
    if callable(s2sample_fn):
        return np.asarray(_to_numpy(s2sample_fn(mapped)), dtype=np.float64)

    fs = _as_complex_scalar(getattr(module, "fs", None), default=1.0).real
    unit = _as_complex_scalar(getattr(module, "unit", None), default=1.0).real
    return np.asarray(_to_numpy(mapped), dtype=np.float64) * fs / unit


def _eval_delay(module: Any, z: complex, *, diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    m = _delay_samples(module)
    if bool(getattr(module, "isint", False)):
        m = np.round(m)
    gamma = _as_complex_scalar(getattr(module, "gamma", None), default=1.0)

    val = np.power(gamma, m) * np.power(z, -m)
    der = -m * np.power(gamma, m) * np.power(z, -m - 1)

    if diagonal:
        v = np.asarray(val, dtype=np.complex128).reshape(-1)
        d = np.asarray(der, dtype=np.complex128).reshape(-1)
        return np.diag(v), np.diag(d)

    return _ensure_2d_matrix(val, path="leaf", what="Delay response"), _ensure_2d_matrix(
        der, path="leaf", what="Delay derivative"
    )


def _eval_filter(module: Any, z: complex, *, diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.asarray(_mapped_param(module), dtype=np.complex128)
    gamma = _as_complex_scalar(getattr(module, "gamma", None), default=1.0)
    taps = np.arange(coeff.shape[0], dtype=np.float64)
    scale_shape = (coeff.shape[0],) + (1,) * (coeff.ndim - 1)
    coeff_aa = coeff * np.power(gamma, taps).reshape(scale_shape)

    h, dh = _poly_zm1_and_der(coeff_aa, z)
    if diagonal:
        hv = np.asarray(h, dtype=np.complex128).reshape(-1)
        dhv = np.asarray(dh, dtype=np.complex128).reshape(-1)
        return np.diag(hv), np.diag(dhv)

    return _ensure_2d_matrix(h, path="leaf", what="Filter response"), _ensure_2d_matrix(
        dh, path="leaf", what="Filter derivative"
    )


def _eval_sos(module: Any, z: complex, *, diagonal: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate SOS/parallelSOS module at ``z`` and compute derivative.

    SOS coefficients are interpreted as ``[b0, b1, b2, a0, a1, a2]`` with
    polynomial variable ``z^-1``.
    """
    param = np.asarray(_mapped_param(module), dtype=np.complex128)
    gamma = _as_complex_scalar(getattr(module, "gamma", None), default=1.0)

    if diagonal:
        # parallelSOSFilter: (K, 6, N)
        if param.ndim != 3 or param.shape[1] != 6:
            raise ValueError(
                f"parallelSOSFilter expects parameter shape (K,6,N), got {param.shape}."
            )
        b = param[:, :3, :]  # (K,3,N)
        a = param[:, 3:6, :]  # (K,3,N)
        scale = np.power(gamma, np.arange(3, dtype=np.float64)).reshape(1, 3, 1)
        b_aa = b * scale
        a_aa = a * scale

        sec_h = np.zeros((param.shape[0], param.shape[2]), dtype=np.complex128)
        sec_dh = np.zeros_like(sec_h)
        for k in range(param.shape[0]):
            num, dnum = _poly_zm1_and_der(b_aa[k], z)
            den, dden = _poly_zm1_and_der(a_aa[k], z)
            with np.errstate(divide="ignore", invalid="ignore"):
                hk = num / den
                dhk = (dnum * den - num * dden) / (den**2)
            sec_h[k] = np.where(np.isfinite(hk), hk, 0.0)
            sec_dh[k] = np.where(np.isfinite(dhk), dhk, 0.0)

        h = np.prod(sec_h, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sec_dh / sec_h
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        dh = h * np.sum(ratio, axis=0)
        return np.diag(h), np.diag(dh)

    # SOSFilter: (K, 6, Nout, Nin)
    if param.ndim != 4 or param.shape[1] != 6:
        raise ValueError(
            f"SOSFilter expects parameter shape (K,6,Nout,Nin), got {param.shape}."
        )
    b = param[:, :3, :, :]
    a = param[:, 3:6, :, :]
    scale = np.power(gamma, np.arange(3, dtype=np.float64)).reshape(1, 3, 1, 1)
    b_aa = b * scale
    a_aa = a * scale

    sec_h = np.zeros((param.shape[0], param.shape[2], param.shape[3]), dtype=np.complex128)
    sec_dh = np.zeros_like(sec_h)
    for k in range(param.shape[0]):
        num, dnum = _poly_zm1_and_der(b_aa[k], z)
        den, dden = _poly_zm1_and_der(a_aa[k], z)
        with np.errstate(divide="ignore", invalid="ignore"):
            hk = num / den
            dhk = (dnum * den - num * dden) / (den**2)
        sec_h[k] = np.where(np.isfinite(hk), hk, 0.0)
        sec_dh[k] = np.where(np.isfinite(dhk), dhk, 0.0)

    h = np.prod(sec_h, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sec_dh / sec_h
        ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    dh = h * np.sum(ratio, axis=0)
    return h, dh


def _leaf_eval(
    module: Any,
    z: complex,
    *,
    path: str,
    leaf_evaluator: LeafEvaluator | None,
) -> tuple[np.ndarray, np.ndarray]:
    if leaf_evaluator is not None:
        custom = leaf_evaluator(module, z, path)
        if custom is not None:
            h_custom, dh_custom = custom
            return _ensure_2d_matrix(
                h_custom, path=path, what="Custom leaf response"
            ), _ensure_2d_matrix(
                dh_custom, path=path, what="Custom leaf derivative"
            )

    tname = _typename(module)

    if isinstance(module, (np.ndarray, list, tuple)):
        mat = _ensure_2d_matrix(np.asarray(module), path=path, what="Numeric leaf")
        return mat, np.zeros_like(mat)

    if tname in {"Gain", "Matrix"}:
        h, dh = _eval_gain_like(module)
        return h, dh
    if tname == "parallelGain":
        h, dh = _eval_parallel_gain(module)
        return h, dh
    if tname == "Delay":
        return _eval_delay(module, z, diagonal=False)
    if tname == "parallelDelay":
        return _eval_delay(module, z, diagonal=True)
    if tname == "Filter":
        return _eval_filter(module, z, diagonal=False)
    if tname == "parallelFilter":
        return _eval_filter(module, z, diagonal=True)
    if tname == "SOSFilter":
        return _eval_sos(module, z, diagonal=False)
    if tname == "parallelSOSFilter":
        return _eval_sos(module, z, diagonal=True)

    if tname in {"FFT", "iFFT", "Identity"}:
        n_in = getattr(module, "input_channels", None)
        n_out = getattr(module, "output_channels", None)
        if n_in is None and n_out is None:
            # shell I/O layers without explicit metadata are treated as scalar identity
            eye = np.eye(1, dtype=np.complex128)
            return eye, np.zeros_like(eye)
        if n_in is None:
            n_in = n_out
        if n_out is None:
            n_out = n_in
        if int(n_in) != int(n_out):
            raise ValueError(
                f"{tname} at {path} has non-square shape ({n_out}x{n_in}), "
                "cannot map to z-domain identity."
            )
        eye = np.eye(int(n_in), dtype=np.complex128)
        return eye, np.zeros_like(eye)

    at_fn = getattr(module, "at", None)
    if callable(at_fn):
        h_raw = np.asarray(at_fn(z), dtype=np.complex128)
        if h_raw.ndim == 1:
            h_raw = np.diag(h_raw)
        h = _ensure_2d_matrix(h_raw, path=path, what=f"{tname}.at(z)")
        der_fn = getattr(module, "der", None)
        if callable(der_fn):
            dh_raw = np.asarray(der_fn(z), dtype=np.complex128)
            if dh_raw.ndim == 1:
                dh_raw = np.diag(dh_raw)
            dh = _ensure_2d_matrix(dh_raw, path=path, what=f"{tname}.der(z)")
        else:
            dh = np.zeros_like(h)
        return h, dh

    raise NotImplementedError(
        f"Unsupported FLAMO leaf module type {tname!r} at {path}. "
        "Provide a custom leaf_evaluator to handle it."
    )


def _eval_node_scalar(
    module: Any,
    z: complex,
    *,
    path: str,
    include_shell_io: bool,
    leaf_evaluator: LeafEvaluator | None,
) -> tuple[np.ndarray, np.ndarray]:
    tname = _typename(module)

    if tname == "Shell":
        core = _shell_core(module)
        if core is None:
            raise ValueError(f"Could not locate core module in Shell at {path}.")
        if not include_shell_io:
            return _eval_node_scalar(
                core,
                z,
                path=f"{path}/core",
                include_shell_io=include_shell_io,
                leaf_evaluator=leaf_evaluator,
            )
        in_layer, out_layer = _shell_input_output(module)
        blocks: list[tuple[str, Any]] = []
        if in_layer is not None:
            blocks.append(("input_layer", in_layer))
        blocks.append(("core", core))
        if out_layer is not None:
            blocks.append(("output_layer", out_layer))

        h: np.ndarray | None = None
        dh: np.ndarray | None = None
        for name, blk in blocks:
            h_blk, dh_blk = _eval_node_scalar(
                blk,
                z,
                path=f"{path}/{name}",
                include_shell_io=include_shell_io,
                leaf_evaluator=leaf_evaluator,
            )
            if h is None:
                h = h_blk
                dh = dh_blk
            else:
                # Series composition: H_new = H_blk @ H
                dh = dh_blk @ h + h_blk @ dh
                h = h_blk @ h
        if h is None or dh is None:
            raise ValueError(f"Shell at {path} has no evaluable blocks.")
        return h, dh

    if tname == "Series":
        children = _series_children(module)
        if not children:
            eye = _identity_for_module(module, path=path)
            return eye, np.zeros_like(eye)

        h: np.ndarray | None = None
        dh: np.ndarray | None = None
        for name, child in children:
            h_child, dh_child = _eval_node_scalar(
                child,
                z,
                path=f"{path}/{name}",
                include_shell_io=include_shell_io,
                leaf_evaluator=leaf_evaluator,
            )
            if h is None:
                h = h_child
                dh = dh_child
            else:
                dh = dh_child @ h + h_child @ dh
                h = h_child @ h
        if h is None or dh is None:
            raise RuntimeError(f"Internal error evaluating Series at {path}.")
        return h, dh

    if tname == "Parallel":
        br_a, br_b = _parallel_branches(module)
        if br_a is None or br_b is None:
            raise ValueError(f"Parallel at {path} does not expose both branches.")
        h_a, dh_a = _eval_node_scalar(
            br_a,
            z,
            path=f"{path}/branchA",
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )
        h_b, dh_b = _eval_node_scalar(
            br_b,
            z,
            path=f"{path}/branchB",
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )
        sum_output = bool(getattr(module, "sum_output", True))
        if sum_output:
            if h_a.shape != h_b.shape:
                raise ValueError(
                    f"Parallel(sum) branch shapes differ at {path}: "
                    f"{h_a.shape} vs {h_b.shape}."
                )
            return h_a + h_b, dh_a + dh_b

        if h_a.shape[1] != h_b.shape[1]:
            raise ValueError(
                f"Parallel(concat) input-channel mismatch at {path}: "
                f"{h_a.shape[1]} vs {h_b.shape[1]}."
            )
        h = np.concatenate([h_a, h_b], axis=0)
        dh = np.concatenate([dh_a, dh_b], axis=0)
        return h, dh

    if tname == "Recursion":
        ff, fb = _recursion_paths(module)
        if ff is None or fb is None:
            raise ValueError(f"Recursion at {path} does not expose both paths.")
        h_ff, dh_ff = _eval_node_scalar(
            ff,
            z,
            path=f"{path}/feedforward",
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )
        h_fb, dh_fb = _eval_node_scalar(
            fb,
            z,
            path=f"{path}/feedback",
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )

        # Closed-loop transfer: H = (I - F B)^-1 F
        loop = h_ff @ h_fb
        if loop.shape[0] != loop.shape[1]:
            raise ValueError(
                f"Recursion loop matrix at {path} must be square, got {loop.shape}."
            )
        eye = np.eye(loop.shape[0], dtype=np.complex128)
        a_mat = eye - loop
        h = np.linalg.solve(a_mat, h_ff)

        # Differentiate (I - L)H = F:
        # (I - L) dH = dF + dL H, with dL = dF B + F dB
        d_loop = dh_ff @ h_fb + h_ff @ dh_fb
        rhs = dh_ff + d_loop @ h
        dh = np.linalg.solve(a_mat, rhs)
        return h, dh

    return _leaf_eval(module, z, path=path, leaf_evaluator=leaf_evaluator)


def probe_flamo_z(
    model: Any,
    z: complex | np.ndarray,
    *,
    derivative: bool = True,
    include_shell_io: bool = False,
    leaf_evaluator: LeafEvaluator | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Probe a FLAMO graph at arbitrary z-plane points.

    Parameters
    ----------
    model
        FLAMO module/graph (Shell, Series, Parallel, Recursion, DSP leaf), or
        a compatible object exposing equivalent attributes.
    z
        Complex scalar or array of complex probe points.
    derivative
        If ``True`` (default), also return ``dH/dz``.
    include_shell_io
        If ``False`` (default), Shell input/output layers are ignored and only
        the core is evaluated. This is typically what you want for z-domain
        transfer probing.
    leaf_evaluator
        Optional callback to evaluate custom leaf modules:
        ``leaf_evaluator(module, z_scalar, path) -> (H, dH) | None``.
        Return ``None`` to fall back to built-in handling.

    Returns
    -------
    H : ndarray
        Transfer matrix values.
        - Scalar ``z`` -> shape ``(n_out, n_in)``
        - Array ``z``   -> shape ``z.shape + (n_out, n_in)``
    dH : ndarray, optional
        Derivative ``dH/dz`` with the same shape as ``H`` when
        ``derivative=True``.
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    is_scalar = z_arr.ndim == 0
    z_flat = z_arr.reshape(-1)

    h_list: list[np.ndarray] = []
    dh_list: list[np.ndarray] = []
    for idx, z_val in enumerate(z_flat):
        h, dh = _eval_node_scalar(
            model,
            complex(z_val),
            path=f"root[{idx}]",
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )
        h_list.append(h.astype(np.complex128, copy=False))
        dh_list.append(dh.astype(np.complex128, copy=False))

    h_stack = np.stack(h_list, axis=0)
    dh_stack = np.stack(dh_list, axis=0)

    if is_scalar:
        h_out = h_stack[0]
        dh_out = dh_stack[0]
    else:
        h_out = h_stack.reshape(z_arr.shape + h_stack.shape[1:])
        dh_out = dh_stack.reshape(z_arr.shape + dh_stack.shape[1:])

    if derivative:
        return h_out, dh_out
    return h_out


class FlamoGraphZFilter(ZFilter):
    """
    ZFilter adapter around a FLAMO graph.

    This allows FLAMO graphs to be used in code that expects ``.at(z)`` and
    ``.der(z)`` methods (e.g. Newton-style decomposition workflows).
    """

    def __init__(
        self,
        model: Any,
        *,
        include_shell_io: bool = False,
        leaf_evaluator: LeafEvaluator | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_shell_io = include_shell_io
        self.leaf_evaluator = leaf_evaluator
        self.is_diagonal = False

        # Infer matrix size from one probe. This avoids assumptions on graph internals.
        h = probe_flamo_z(
            model,
            1.0 + 0.0j,
            derivative=False,
            include_shell_io=include_shell_io,
            leaf_evaluator=leaf_evaluator,
        )
        h_mat = np.asarray(h, dtype=np.complex128)
        if h_mat.ndim != 2:
            raise ValueError(
                f"Expected 2-D transfer matrix at scalar z, got shape {h_mat.shape}."
            )
        self.n, self.m = h_mat.shape
        self.number_of_delay_units = 0

    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        return np.asarray(
            probe_flamo_z(
                self.model,
                z,
                derivative=False,
                include_shell_io=self.include_shell_io,
                leaf_evaluator=self.leaf_evaluator,
            ),
            dtype=np.complex128,
        )

    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        _, dh = probe_flamo_z(
            self.model,
            z,
            derivative=True,
            include_shell_io=self.include_shell_io,
            leaf_evaluator=self.leaf_evaluator,
        )
        return np.asarray(dh, dtype=np.complex128)

    def inverse(self):
        raise NotImplementedError("Inverse is not defined for generic FLAMO graphs.")

    def dfilt_type(self):
        warnings.warn(_DFILT_DEPRECATION, DeprecationWarning, stacklevel=2)
        return "none"

    def dfilt_parameter(self, n: int, m: int):
        warnings.warn(_DFILT_DEPRECATION, DeprecationWarning, stacklevel=2)
        raise NotImplementedError(
            "dfilt_parameter is not available for FlamoGraphZFilter."
        )


def flamo_graph_to_zfilter(
    model: Any,
    *,
    include_shell_io: bool = False,
    leaf_evaluator: LeafEvaluator | None = None,
) -> FlamoGraphZFilter:
    """
    Wrap a FLAMO graph in a ``ZFilter``-compatible adapter.
    """
    return FlamoGraphZFilter(
        model,
        include_shell_io=include_shell_io,
        leaf_evaluator=leaf_evaluator,
    )

