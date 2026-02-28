"""
Autograd-based arbitrary z-plane probing for FLAMO-like graphs.

Stage A implementation: keep FLAMO untouched and provide a pyFDN-side probe
backend that:

1. Reuses FLAMO graph structure (Shell/Series/Parallel/Recursion + leaves),
2. Evaluates transfer matrices at arbitrary complex ``z``,
3. Computes ``dH/dz`` via torch autograd (Wirtinger reconstruction).
"""

from __future__ import annotations

import types
import warnings
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from pyFDN.auxiliary.filters import ZFilter

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a dependency
    _HAS_TORCH = False


LeafValueEvaluator = Callable[[Any, "torch.Tensor", str], "torch.Tensor | None"]

_DFILT_DEPRECATION = (
    "dfilt_type and dfilt_parameter are deprecated (MATLAB compatibility). "
    "They will be removed in a future version."
)


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "flamo_autograd_probe requires torch. Install torch to use this backend."
        )


def _typename(module: Any) -> str:
    return type(module).__name__


def _to_torch(value: Any, *, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    out = value
    if hasattr(out, "detach"):
        out = out.detach()
    if hasattr(out, "cpu"):
        out = out.cpu()
    if hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out)


def _mapped_param_torch(module: Any, *, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
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
    return _to_torch(mapped, device=device, dtype=dtype)


def _as_complex(value: "torch.Tensor", *, complex_dtype: "torch.dtype") -> "torch.Tensor":
    if value.is_complex():
        return value.to(dtype=complex_dtype)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    real = value.to(dtype=real_dtype)
    return torch.complex(real, torch.zeros_like(real))


def _ensure_2d(tensor: "torch.Tensor", *, what: str, path: str) -> "torch.Tensor":
    if tensor.ndim != 2:
        raise ValueError(f"{what} at {path} must be 2-D, got shape {tuple(tensor.shape)}.")
    return tensor


def _infer_probe_device(model: Any) -> "torch.device":
    param = getattr(model, "param", None)
    if isinstance(param, torch.Tensor):
        return param.device
    for child in getattr(model, "_modules", {}).values():
        dev = _infer_probe_device(child)
        if dev is not None:
            return dev
    return torch.device("cpu")


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
    getter = getattr(module, "get_core", None)
    if callable(getter):
        core = getter()
        if core is not None:
            return core
    return getattr(module, "_Shell__core", None) or getattr(module, "core", None)


def _shell_io(module: Any) -> tuple[Any | None, Any | None]:
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
    a = getattr(module, "branchA", None) or getattr(module, "brA", None)
    b = getattr(module, "branchB", None) or getattr(module, "brB", None)
    return a, b


def _recursion_paths(module: Any) -> tuple[Any | None, Any | None]:
    ff = getattr(module, "feedforward", None) or getattr(module, "fF", None)
    fb = getattr(module, "feedback", None) or getattr(module, "fB", None)
    return ff, fb


def _identity_like(module: Any, *, device: "torch.device", dtype: "torch.dtype", path: str) -> "torch.Tensor":
    n_in = getattr(module, "input_channels", None)
    n_out = getattr(module, "output_channels", None)
    if n_in is None or n_out is None:
        return torch.eye(1, device=device, dtype=dtype)
    if int(n_in) != int(n_out):
        raise ValueError(
            f"Cannot build identity for non-square block at {path}: {n_out}x{n_in}."
        )
    return torch.eye(int(n_in), device=device, dtype=dtype)


def _eval_leaf_value_torch(
    module: Any,
    z: "torch.Tensor",
    *,
    path: str,
    leaf_value_evaluator: LeafValueEvaluator | None,
) -> "torch.Tensor":
    device = z.device
    dtype = z.dtype
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    tname = _typename(module)

    if leaf_value_evaluator is not None:
        custom = leaf_value_evaluator(module, z, path)
        if custom is not None:
            return _ensure_2d(custom, what="custom leaf value", path=path)

    if isinstance(module, (np.ndarray, list, tuple)):
        m = _to_torch(module, device=device, dtype=real_dtype)
        return _as_complex(_ensure_2d(m, what="numeric leaf", path=path), complex_dtype=dtype)

    if tname in {"Gain", "Matrix"}:
        p = _mapped_param_torch(module, device=device, dtype=real_dtype)
        return _as_complex(_ensure_2d(p, what=f"{tname} parameter", path=path), complex_dtype=dtype)

    if tname == "parallelGain":
        p = _mapped_param_torch(module, device=device, dtype=real_dtype).reshape(-1)
        return torch.diag(_as_complex(p, complex_dtype=dtype))

    if tname in {"Delay", "parallelDelay"}:
        p = _mapped_param_torch(module, device=device, dtype=real_dtype)
        s2sample = getattr(module, "s2sample", None)
        if callable(s2sample):
            m = _to_torch(s2sample(p), device=device, dtype=real_dtype)
        else:
            fs = float(_to_numpy(getattr(module, "fs", 1.0)).reshape(-1)[0])
            unit = float(_to_numpy(getattr(module, "unit", 1.0)).reshape(-1)[0])
            m = p * fs / unit
        if bool(getattr(module, "isint", False)):
            m = torch.round(m)
        gamma_raw = getattr(module, "gamma", None)
        gamma = (
            _to_torch(gamma_raw, device=device, dtype=real_dtype)
            if gamma_raw is not None
            else torch.tensor(1.0, device=device, dtype=real_dtype)
        )
        val = torch.pow(gamma, m) * torch.pow(z, -_as_complex(m, complex_dtype=dtype))
        if tname == "parallelDelay":
            return torch.diag(val.reshape(-1))
        return _ensure_2d(val, what="Delay value", path=path)

    if tname in {"Filter", "parallelFilter"}:
        # Mapped coeffs: (taps,out,in) or (taps,n)
        coeff = _mapped_param_torch(module, device=device, dtype=real_dtype)
        gamma_raw = getattr(module, "gamma", None)
        gamma = (
            _to_torch(gamma_raw, device=device, dtype=real_dtype)
            if gamma_raw is not None
            else torch.tensor(1.0, device=device, dtype=real_dtype)
        )
        taps = coeff.shape[0]
        k = torch.arange(taps, device=device, dtype=real_dtype)
        scale_shape = (taps,) + (1,) * (coeff.ndim - 1)
        coeff_aa = coeff * torch.pow(gamma, k).reshape(scale_shape)
        z_pow = torch.pow(z, -_as_complex(k, complex_dtype=dtype)).reshape(scale_shape)
        h = torch.sum(_as_complex(coeff_aa, complex_dtype=dtype) * z_pow, dim=0)
        if tname == "parallelFilter":
            return torch.diag(h.reshape(-1))
        return _ensure_2d(h, what="Filter value", path=path)

    if tname in {"SOSFilter", "parallelSOSFilter"}:
        # Mapped coeffs: SOSFilter (K,6,out,in), parallelSOSFilter (K,6,n)
        param = _mapped_param_torch(module, device=device, dtype=real_dtype)
        gamma_raw = getattr(module, "gamma", None)
        gamma = (
            _to_torch(gamma_raw, device=device, dtype=real_dtype)
            if gamma_raw is not None
            else torch.tensor(1.0, device=device, dtype=real_dtype)
        )
        k = torch.arange(3, device=device, dtype=real_dtype)
        aa = torch.pow(gamma, k)
        z_pow = torch.pow(z, -_as_complex(k, complex_dtype=dtype))

        if tname == "parallelSOSFilter":
            if param.ndim != 3 or param.shape[1] != 6:
                raise ValueError(
                    f"parallelSOSFilter at {path} expects (K,6,N), got {tuple(param.shape)}"
                )
            b = param[:, :3, :] * aa.reshape(1, 3, 1)
            a = param[:, 3:6, :] * aa.reshape(1, 3, 1)
            num = torch.sum(_as_complex(b, complex_dtype=dtype) * z_pow.reshape(1, 3, 1), dim=1)
            den = torch.sum(_as_complex(a, complex_dtype=dtype) * z_pow.reshape(1, 3, 1), dim=1)
            h_sec = num / den
            h = torch.prod(h_sec, dim=0)
            return torch.diag(h.reshape(-1))

        if param.ndim != 4 or param.shape[1] != 6:
            raise ValueError(
                f"SOSFilter at {path} expects (K,6,Nout,Nin), got {tuple(param.shape)}"
            )
        b = param[:, :3, :, :] * aa.reshape(1, 3, 1, 1)
        a = param[:, 3:6, :, :] * aa.reshape(1, 3, 1, 1)
        num = torch.sum(_as_complex(b, complex_dtype=dtype) * z_pow.reshape(1, 3, 1, 1), dim=1)
        den = torch.sum(_as_complex(a, complex_dtype=dtype) * z_pow.reshape(1, 3, 1, 1), dim=1)
        h_sec = num / den
        return torch.prod(h_sec, dim=0)

    if tname in {"FFT", "iFFT", "Identity"}:
        return _identity_like(module, device=device, dtype=dtype, path=path)

    at_fn = getattr(module, "at", None)
    if callable(at_fn):
        # Fallback for non-FLAMO leaves: this path is not differentiable wrt z.
        val = np.asarray(at_fn(complex(_to_numpy(z).item())))
        tensor = _to_torch(val, device=device, dtype=torch.complex128).to(dtype=dtype)
        if tensor.ndim == 1:
            tensor = torch.diag(tensor)
        return _ensure_2d(tensor, what=f"{tname}.at(z)", path=path)

    raise NotImplementedError(
        f"Unsupported FLAMO leaf module type {tname!r} at {path} "
        "(autograd probe backend)."
    )


def _eval_node_value_torch(
    module: Any,
    z: "torch.Tensor",
    *,
    include_shell_io: bool,
    path: str,
    leaf_value_evaluator: LeafValueEvaluator | None,
) -> "torch.Tensor":
    tname = _typename(module)
    device = z.device
    dtype = z.dtype

    if tname == "Shell":
        core = _shell_core(module)
        if core is None:
            raise ValueError(f"Could not locate core in Shell at {path}")
        if not include_shell_io:
            return _eval_node_value_torch(
                core,
                z,
                include_shell_io=include_shell_io,
                path=f"{path}/core",
                leaf_value_evaluator=leaf_value_evaluator,
            )
        in_layer, out_layer = _shell_io(module)
        blocks: list[tuple[str, Any]] = []
        if in_layer is not None:
            blocks.append(("input_layer", in_layer))
        blocks.append(("core", core))
        if out_layer is not None:
            blocks.append(("output_layer", out_layer))

        h: torch.Tensor | None = None
        for name, blk in blocks:
            h_blk = _eval_node_value_torch(
                blk,
                z,
                include_shell_io=include_shell_io,
                path=f"{path}/{name}",
                leaf_value_evaluator=leaf_value_evaluator,
            )
            if h is None:
                h = h_blk
            else:
                h = h_blk @ h
        if h is None:
            return torch.eye(1, device=device, dtype=dtype)
        return h

    if tname == "Series":
        children = _series_children(module)
        if not children:
            return _identity_like(module, device=device, dtype=dtype, path=path)
        h: torch.Tensor | None = None
        for name, child in children:
            h_child = _eval_node_value_torch(
                child,
                z,
                include_shell_io=include_shell_io,
                path=f"{path}/{name}",
                leaf_value_evaluator=leaf_value_evaluator,
            )
            if h is None:
                h = h_child
            else:
                h = h_child @ h
        assert h is not None
        return h

    if tname == "Parallel":
        a, b = _parallel_branches(module)
        if a is None or b is None:
            raise ValueError(f"Parallel at {path} does not expose both branches")
        ha = _eval_node_value_torch(
            a,
            z,
            include_shell_io=include_shell_io,
            path=f"{path}/branchA",
            leaf_value_evaluator=leaf_value_evaluator,
        )
        hb = _eval_node_value_torch(
            b,
            z,
            include_shell_io=include_shell_io,
            path=f"{path}/branchB",
            leaf_value_evaluator=leaf_value_evaluator,
        )
        if bool(getattr(module, "sum_output", True)):
            if ha.shape != hb.shape:
                raise ValueError(
                    f"Parallel(sum) branch shapes differ at {path}: "
                    f"{tuple(ha.shape)} vs {tuple(hb.shape)}"
                )
            return ha + hb
        if ha.shape[1] != hb.shape[1]:
            raise ValueError(
                f"Parallel(concat) input-channel mismatch at {path}: "
                f"{ha.shape[1]} vs {hb.shape[1]}"
            )
        return torch.cat([ha, hb], dim=0)

    if tname == "Recursion":
        ff_mod, fb_mod = _recursion_paths(module)
        if ff_mod is None or fb_mod is None:
            raise ValueError(f"Recursion at {path} does not expose both paths")
        ff = _eval_node_value_torch(
            ff_mod,
            z,
            include_shell_io=include_shell_io,
            path=f"{path}/feedforward",
            leaf_value_evaluator=leaf_value_evaluator,
        )
        fb = _eval_node_value_torch(
            fb_mod,
            z,
            include_shell_io=include_shell_io,
            path=f"{path}/feedback",
            leaf_value_evaluator=leaf_value_evaluator,
        )
        loop = ff @ fb
        if loop.shape[0] != loop.shape[1]:
            raise ValueError(
                f"Recursion loop matrix at {path} must be square, got {tuple(loop.shape)}"
            )
        eye = torch.eye(loop.shape[0], device=device, dtype=dtype)
        return torch.linalg.solve(eye - loop, ff)

    return _eval_leaf_value_torch(
        module,
        z,
        path=path,
        leaf_value_evaluator=leaf_value_evaluator,
    )


def _wirtinger_dfdz_from_jacobian(jac: "torch.Tensor") -> "torch.Tensor":
    """
    Convert jacobian of view_as_real output wrt (x,y) to d/dz.

    Input jac shape: (..., 2, 2) where last-but-one is output [u,v] and
    last is input [x,y].
    """
    du_dx = jac[..., 0, 0]
    du_dy = jac[..., 0, 1]
    dv_dx = jac[..., 1, 0]
    dv_dy = jac[..., 1, 1]
    real = 0.5 * (du_dx + dv_dy)
    imag = 0.5 * (dv_dx - du_dy)
    return torch.complex(real, imag)


def probe_flamo_z_autograd(
    model: Any,
    z: complex | np.ndarray,
    *,
    derivative: bool = True,
    include_shell_io: bool = False,
    leaf_value_evaluator: LeafValueEvaluator | None = None,
    create_graph: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Probe FLAMO graph at arbitrary z points and compute dH/dz via autograd.

    Notes
    -----
    - Derivatives are computed with Wirtinger reconstruction from the jacobian
      of ``(Re(H), Im(H))`` wrt ``(Re(z), Im(z))``.
    - Scalar and array ``z`` are supported.
    """
    _require_torch()
    z_arr = np.asarray(z, dtype=np.complex128)
    is_scalar = z_arr.ndim == 0
    z_flat = z_arr.reshape(-1)

    device = _infer_probe_device(model)
    complex_dtype = torch.complex128
    real_dtype = torch.float64

    def _probe_value(z_t: "torch.Tensor") -> "torch.Tensor":
        return _eval_node_value_torch(
            model,
            z_t,
            include_shell_io=include_shell_io,
            path="root",
            leaf_value_evaluator=leaf_value_evaluator,
        )

    h_list: list[np.ndarray] = []
    dh_list: list[np.ndarray] = []

    for z_val in z_flat:
        zr = torch.tensor(float(np.real(z_val)), device=device, dtype=real_dtype)
        zi = torch.tensor(float(np.imag(z_val)), device=device, dtype=real_dtype)
        z_t = torch.complex(zr, zi)

        if derivative:
            xy = torch.stack([zr, zi]).detach().clone().requires_grad_(True)

            def _fn_xy(xy_var: torch.Tensor) -> torch.Tensor:
                z_var = torch.complex(xy_var[0], xy_var[1])
                h_var = _probe_value(z_var)
                return torch.view_as_real(h_var.to(dtype=complex_dtype))

            jac = torch.autograd.functional.jacobian(
                _fn_xy,
                xy,
                create_graph=create_graph,
            )
            h_t = _probe_value(torch.complex(xy[0], xy[1])).to(dtype=complex_dtype)
            dh_t = _wirtinger_dfdz_from_jacobian(jac)
        else:
            h_t = _probe_value(z_t).to(dtype=complex_dtype)
            dh_t = None

        h_list.append(_to_numpy(h_t))
        if derivative and dh_t is not None:
            dh_list.append(_to_numpy(dh_t))

    h_stack = np.stack(h_list, axis=0)
    if is_scalar:
        h_out = h_stack[0]
    else:
        h_out = h_stack.reshape(z_arr.shape + h_stack.shape[1:])

    if not derivative:
        return h_out

    dh_stack = np.stack(dh_list, axis=0)
    if is_scalar:
        dh_out = dh_stack[0]
    else:
        dh_out = dh_stack.reshape(z_arr.shape + dh_stack.shape[1:])
    return h_out, dh_out


class FlamoAutogradGraphZFilter(ZFilter):
    """ZFilter adapter backed by autograd probing of a FLAMO graph."""

    def __init__(
        self,
        model: Any,
        *,
        include_shell_io: bool = False,
        leaf_value_evaluator: LeafValueEvaluator | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_shell_io = include_shell_io
        self.leaf_value_evaluator = leaf_value_evaluator
        self.is_diagonal = False
        self.number_of_delay_units = 0

        h = probe_flamo_z_autograd(
            model,
            1.0 + 0j,
            derivative=False,
            include_shell_io=include_shell_io,
            leaf_value_evaluator=leaf_value_evaluator,
        )
        h_arr = np.asarray(h, dtype=np.complex128)
        if h_arr.ndim != 2:
            raise ValueError(
                f"Expected 2-D transfer matrix at scalar z, got shape {h_arr.shape}."
            )
        self.n, self.m = h_arr.shape

    @staticmethod
    def _as_scalar(z: complex | np.ndarray) -> complex:
        arr = np.asarray(z)
        if arr.ndim == 0:
            return complex(arr.item())
        if arr.size == 1:
            return complex(arr.reshape(-1)[0])
        raise ValueError("FlamoAutogradGraphZFilter expects scalar z for .at/.der")

    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        z_scalar = self._as_scalar(z)
        return np.asarray(
            probe_flamo_z_autograd(
                self.model,
                z_scalar,
                derivative=False,
                include_shell_io=self.include_shell_io,
                leaf_value_evaluator=self.leaf_value_evaluator,
            ),
            dtype=np.complex128,
        )

    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        z_scalar = self._as_scalar(z)
        _, dh = probe_flamo_z_autograd(
            self.model,
            z_scalar,
            derivative=True,
            include_shell_io=self.include_shell_io,
            leaf_value_evaluator=self.leaf_value_evaluator,
        )
        return np.asarray(dh, dtype=np.complex128)

    def inverse(self):
        raise NotImplementedError(
            "Inverse is not defined for generic FlamoAutogradGraphZFilter."
        )

    def dfilt_type(self):
        warnings.warn(_DFILT_DEPRECATION, DeprecationWarning, stacklevel=2)
        return "none"

    def dfilt_parameter(self, n: int, m: int):
        warnings.warn(_DFILT_DEPRECATION, DeprecationWarning, stacklevel=2)
        raise NotImplementedError(
            "dfilt_parameter is not available for FlamoAutogradGraphZFilter."
        )


def flamo_graph_to_autograd_zfilter(
    model: Any,
    *,
    include_shell_io: bool = False,
    leaf_value_evaluator: LeafValueEvaluator | None = None,
) -> FlamoAutogradGraphZFilter:
    """Wrap a FLAMO graph with autograd-backed .at(z)/.der(z) interface."""
    return FlamoAutogradGraphZFilter(
        model,
        include_shell_io=include_shell_io,
        leaf_value_evaluator=leaf_value_evaluator,
    )


def attach_autograd_probe(
    model: Any,
    *,
    include_shell_io: bool = False,
    leaf_value_evaluator: LeafValueEvaluator | None = None,
) -> Any:
    """
    Attach ``probe`` / ``probe_with_derivative`` methods to a FLAMO model.

    This gives a lightweight "alternative forward path" without changing FLAMO.
    """

    def _probe(self, z, derivative: bool = False):
        return probe_flamo_z_autograd(
            self,
            z,
            derivative=derivative,
            include_shell_io=include_shell_io,
            leaf_value_evaluator=leaf_value_evaluator,
        )

    def _probe_with_derivative(self, z):
        return probe_flamo_z_autograd(
            self,
            z,
            derivative=True,
            include_shell_io=include_shell_io,
            leaf_value_evaluator=leaf_value_evaluator,
        )

    model.probe = types.MethodType(_probe, model)
    model.probe_with_derivative = types.MethodType(_probe_with_derivative, model)
    return model

