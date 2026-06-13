"""Gallery of feedback matrices, specialized systems, and complete FDN builds.

Translation of fdnMatrixGallery.m from fdnToolbox.
"""

# TODO: Menzel matrix

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, NoReturn, overload

import numpy as np

from .householder_matrix import householder_matrix
from .random_orthogonal import random_orthogonal


class FDNSystem(NamedTuple):
    """Full FDN system matrices returned by :func:`fdn_system_gallery`."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class FDNBuild:
    """Complete FDN parameters returned by :func:`fdn_build_gallery`.

    ``filters`` is either ``None`` or a per-delay SOS bank with shape
    ``(num_sections, 6, N)`` suitable for ``dss_to_flamo(..., sos_filter=...)``.
    ``post_eq`` is an optional per-output SOS bank with shape
    ``(num_sections, 6, num_outputs)`` suitable for the ``output_filter``
    argument of :func:`pyFDN.dss_to_flamo`.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    delays: np.ndarray
    fs: float
    filters: np.ndarray | None = None
    post_eq: np.ndarray | None = None


def _circulant(v: np.ndarray, direction: int = 1) -> np.ndarray:
    """Build a circulant matrix from first-row vector v.

    ``direction=1``: each row shifts right; ``direction=-1``: shifts left.
    """
    v = np.asarray(v, dtype=float).ravel()
    N = len(v)
    rows = [np.roll(v, i * direction) for i in range(N)]
    return np.array(rows)


# Pure feedback-matrix types (return np.ndarray).
_MATRIX_TYPES = [
    "orthogonal",
    "Hadamard",
    "circulant",
    "Householder",
    "parallel",
    "permutation",
    "diagonallySimilarToOrthogonal",
    "tinyRotation",
    "Anderson",
]

# Full-system types (return FDNSystem).
_SYSTEM_TYPES = [
    "series",
    "nestedAllpass",
    "polettiAllpass",
    "homogeneousAllpass",
    "SchroederReverberator",
    "allpassInFDN",
]

# Complete, ready-to-render FDN builds (return FDNBuild).
_BUILD_TYPES = [
    "vanilla",
    "vanillaBroadband",
    "vanillaFirstOrder",
    "roomSmall",
    "roomLarge",
]

# Filter (FIR paraunitary) matrix types (return (N, N, L) np.ndarray).
_FILTER_MATRIX_TYPES = [
    "RandomDense",
    "Velvet",
    "FromElementals",
]


def _build_rng(
    rng: np.random.Generator | int | None, default_seed: int
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(default_seed if rng is None else rng)


def _random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((N, N)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def _build_io_matrices(
    N: int,
    num_inputs: int,
    num_outputs: int,
    io_type: str,
    input_scale: float,
    output_scale: float,
    direct_gain: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(num_inputs, num_outputs) < 1:
        raise ValueError("num_inputs and num_outputs must be positive")

    if io_type == "ones":
        B = np.ones((N, num_inputs))
        C = np.ones((num_outputs, N))
    elif io_type == "normalized":
        B = np.ones((N, num_inputs)) / np.sqrt(N)
        C = np.ones((num_outputs, N)) / np.sqrt(N)
    elif io_type == "identity":
        B = np.eye(N, num_inputs)
        C = np.eye(num_outputs, N)
    elif io_type == "random":
        B = rng.standard_normal((N, num_inputs))
        C = rng.standard_normal((num_outputs, N))
    else:
        raise ValueError(
            "io_type must be one of 'ones', 'normalized', 'identity', or 'random'"
        )

    B = input_scale * B
    C = output_scale * C
    if direct_gain is None:
        D = rng.standard_normal((num_outputs, num_inputs))
    else:
        D = np.full((num_outputs, num_inputs), direct_gain, dtype=float)
    return B, C, D


def _constant_decay_filters(delays: np.ndarray, rt60: float, fs: float) -> np.ndarray:
    from ..auxiliary.acoustics import first_order_absorption

    return first_order_absorption(rt60, rt60, delays, fs)


def _build_post_eq(
    num_outputs: int,
    delays: np.ndarray,
    fs: float,
    rt60_dc: float | None,
    rt60_nyquist: float | None,
    delay: float | None,
) -> np.ndarray | None:
    if rt60_dc is None:
        if rt60_nyquist is not None or delay is not None:
            raise ValueError("post_eq_rt60 must be provided when configuring post EQ")
        return None
    if rt60_dc <= 0 or (rt60_nyquist is not None and rt60_nyquist <= 0):
        raise ValueError("post EQ reverberation times must be positive")

    from ..auxiliary.acoustics import first_order_absorption

    post_eq_delay = float(np.mean(delays)) if delay is None else float(delay)
    if post_eq_delay <= 0:
        raise ValueError("post_eq_delay must be positive")
    rt60_nyquist = rt60_dc if rt60_nyquist is None else rt60_nyquist
    return first_order_absorption(
        rt60_dc,
        rt60_nyquist,
        np.full(num_outputs, post_eq_delay),
        fs,
    )


@overload
def fdn_build_gallery(
    N: int | None = ...,
    build_type: None = ...,
    *,
    fs: float = ...,
    delays: np.ndarray | None = ...,
    delay_range: tuple[int, int] = ...,
    sort_delays: bool = ...,
    num_inputs: int = ...,
    num_outputs: int = ...,
    io_type: str = ...,
    input_scale: float = ...,
    output_scale: float = ...,
    direct_gain: float | None = ...,
    rt60: float = ...,
    rt60_nyquist: float = ...,
    gain_per_sample: float | None = ...,
    feedback_gain: float = ...,
    post_eq_rt60: float | None = ...,
    post_eq_rt60_nyquist: float | None = ...,
    post_eq_delay: float | None = ...,
    rng: np.random.Generator | int | None = ...,
) -> list[str]: ...


@overload
def fdn_build_gallery(
    N: int | None = ...,
    build_type: str = ...,
    *,
    fs: float = ...,
    delays: np.ndarray | None = ...,
    delay_range: tuple[int, int] = ...,
    sort_delays: bool = ...,
    num_inputs: int = ...,
    num_outputs: int = ...,
    io_type: str = ...,
    input_scale: float = ...,
    output_scale: float = ...,
    direct_gain: float | None = ...,
    rt60: float = ...,
    rt60_nyquist: float = ...,
    gain_per_sample: float | None = ...,
    feedback_gain: float = ...,
    post_eq_rt60: float | None = ...,
    post_eq_rt60_nyquist: float | None = ...,
    post_eq_delay: float | None = ...,
    rng: np.random.Generator | int | None = ...,
) -> FDNBuild: ...


def fdn_build_gallery(
    N: int | None = None,
    build_type: str | None = None,
    *,
    fs: float = 48_000.0,
    delays: np.ndarray | None = None,
    delay_range: tuple[int, int] = (400, 1200),
    sort_delays: bool = False,
    num_inputs: int = 1,
    num_outputs: int = 1,
    io_type: str = "normalized",
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    direct_gain: float | None = 0.0,
    rt60: float = 2.0,
    rt60_nyquist: float = 0.5,
    gain_per_sample: float | None = None,
    feedback_gain: float = 1.0,
    post_eq_rt60: float | None = None,
    post_eq_rt60_nyquist: float | None = None,
    post_eq_delay: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> FDNBuild | list[str]:
    """Return a complete FDN build or list the available preset types.

    The vanilla presets share configurable delays and I/O matrices:

    - ``"vanilla"`` uses an orthogonal feedback matrix without decay filters.
    - ``"vanillaBroadband"`` puts delay-proportional broadband decay in ``A``.
    - ``"vanillaFirstOrder"`` keeps ``A`` orthogonal and returns first-order
      absorption filters for the requested DC and Nyquist reverberation times.
    - ``"roomSmall"`` and ``"roomLarge"`` use short and long random delay
      ranges respectively. Their size is configurable and they include
      broadband absorption filters.

    Random presets use a local :class:`numpy.random.Generator`; passing an
    integer or generator makes the build reproducible without mutating NumPy's
    global random state. The default presets are deterministic.

    Args:
        N: Number of delay lines. Inferred from ``delays`` when possible and
            defaults to six for room presets.
        build_type: Preset name, or ``None`` to list available names.
        fs: Sample rate in Hz.
        delays: Optional explicit delay lengths in samples.
        delay_range: Half-open random delay range when ``delays`` is omitted.
        sort_delays: Sort randomly generated or supplied delays.
        num_inputs: Number of input channels.
        num_outputs: Number of output channels.
        io_type: I/O matrix style: ``ones``, ``normalized``, ``identity``, or
            ``random``.
        input_scale: Scalar applied to ``B``.
        output_scale: Scalar applied to ``C``.
        direct_gain: Constant direct gain, or ``None`` for random ``D``.
        rt60: Broadband or DC reverberation time in seconds.
        rt60_nyquist: Nyquist reverberation time for first-order absorption.
        gain_per_sample: Optional explicit broadband gain, overriding ``rt60``.
        feedback_gain: Scalar applied to the orthogonal feedback matrix.
        post_eq_rt60: Optional post-EQ reverberation time at DC. Providing it
            enables a first-order output filter.
        post_eq_rt60_nyquist: Optional post-EQ reverberation time at Nyquist.
            Defaults to ``post_eq_rt60`` for a flat output gain.
        post_eq_delay: Effective delay used to turn post-EQ reverberation times
            into filter gains. Defaults to the mean FDN delay.
        rng: Local NumPy generator or integer seed.

    Returns:
        A complete :class:`FDNBuild`, or the list of preset names.
    """
    if build_type is None:
        return list(_BUILD_TYPES)
    if build_type not in _BUILD_TYPES:
        raise ValueError(
            f"Unknown build_type {build_type!r}. Supported: {_BUILD_TYPES}"
        )
    if fs <= 0:
        raise ValueError("fs must be positive")
    if rt60 <= 0 or rt60_nyquist <= 0:
        raise ValueError("reverberation times must be positive")

    room_presets = {
        "roomSmall": ((400, 800), 0.525, 5),
        "roomLarge": ((1100, 2600), 4.2, 6),
    }
    if build_type in room_presets:
        room_delay_range, preset_rt60, default_seed = room_presets[build_type]
        if delays is not None:
            raise ValueError(f"{build_type} generates its delay lengths")
        N = 6 if N is None else N
        if N < 1:
            raise ValueError("N must be positive")
        local_rng = _build_rng(rng, default_seed)
        delays_array = local_rng.integers(*room_delay_range, size=N)
        if sort_delays:
            delays_array = np.sort(delays_array)
        A = _random_orthogonal(N, local_rng)
        B, C, D = _build_io_matrices(
            N,
            num_inputs,
            num_outputs,
            io_type,
            input_scale,
            output_scale,
            direct_gain,
            local_rng,
        )
        room_filters = _constant_decay_filters(delays_array, preset_rt60, fs)
        post_eq = _build_post_eq(
            num_outputs,
            delays_array,
            fs,
            post_eq_rt60,
            post_eq_rt60_nyquist,
            post_eq_delay,
        )
        return FDNBuild(A, B, C, D, delays_array, float(fs), room_filters, post_eq)

    if delays is not None:
        delays_array = np.asarray(delays, dtype=int).ravel()
        if N is None:
            N = delays_array.size
        elif delays_array.size != N:
            raise ValueError("delays must contain exactly N values")
    else:
        if N is None:
            raise ValueError("N must be provided for vanilla builds")
        low, high = delay_range
        if low < 1 or high <= low:
            raise ValueError("delay_range must satisfy 1 <= low < high")
        local_rng = _build_rng(rng, 0)
        delays_array = local_rng.integers(low, high, size=N)

    if sort_delays:
        delays_array = np.sort(delays_array)

    if np.any(delays_array < 1):
        raise ValueError("all delays must be positive")
    local_rng = _build_rng(rng, 0) if delays is not None else local_rng
    A = feedback_gain * _random_orthogonal(N, local_rng)
    B, C, D = _build_io_matrices(
        N,
        num_inputs,
        num_outputs,
        io_type,
        input_scale,
        output_scale,
        direct_gain,
        local_rng,
    )

    filters: np.ndarray | None = None
    if build_type == "vanillaBroadband":
        if gain_per_sample is None:
            from ..auxiliary.acoustics import rt_to_gain_per_sample

            gain_per_sample = rt_to_gain_per_sample(rt60, fs)
        if not 0 < gain_per_sample <= 1:
            raise ValueError("gain_per_sample must be in (0, 1]")
        A = np.diag(gain_per_sample**delays_array) @ A
    elif build_type == "vanillaFirstOrder":
        from ..auxiliary.acoustics import first_order_absorption

        filters = first_order_absorption(rt60, rt60_nyquist, delays_array, float(fs))

    post_eq = _build_post_eq(
        num_outputs,
        delays_array,
        fs,
        post_eq_rt60,
        post_eq_rt60_nyquist,
        post_eq_delay,
    )
    return FDNBuild(A, B, C, D, delays_array, float(fs), filters, post_eq)


@overload
def filter_matrix_gallery(
    N: int,
    matrix_type: str,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> np.ndarray: ...


@overload
def filter_matrix_gallery(
    N: int | None = ...,
    matrix_type: None = ...,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> list[str]: ...


@overload
def filter_matrix_gallery(
    N: None = ...,
    matrix_type: str = ...,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> NoReturn: ...


def filter_matrix_gallery(
    N: int | None = None,
    matrix_type: str | None = None,
    *,
    num_stages: int = 3,
    sparsity: float = 3.0,
    stage_matrix_type: str = "Hadamard",
) -> np.ndarray | list[str]:
    """Return an FIR (filter) feedback matrix of the requested type, or list all type names.

    All types are paraunitary (lossless): ``A^T(z^{-1}) A(z) = I``. Used as
    scattering feedback matrices in an FDN (Schlecht & Habets 2020).

    Args:
        N: Matrix size.  Ignored when ``matrix_type`` is ``None``.
        matrix_type: One of ``"RandomDense"`` (dense cascaded paraunitary
            matrix), ``"Velvet"`` (sparse velvet-noise feedback matrix), or
            ``"FromElementals"`` (cascade of degree-one lossless factors,
            polynomial degree ``N * num_stages``).  Pass ``None`` (or call
            with no arguments) to get the list of all type names.
        num_stages: Number of cascade stages (or degree factor for
            ``"FromElementals"``).
        sparsity: Sparsity of the ``"Velvet"`` type (ignored otherwise).
        stage_matrix_type: Stage matrix for ``"RandomDense"`` and ``"Velvet"``:
            ``"Hadamard"`` or ``"random"`` (random orthogonal; avoids the
            structural double poles at z = ±1 of Hadamard stages).

    Returns:
        Feedback matrix of shape ``(N, N, L)`` in z^{-1} convention, or a list
        of type-name strings.

    Example::

        filter_matrix_gallery()              # → list of type strings
        filter_matrix_gallery(4, "Velvet", num_stages=3, sparsity=3)
    """
    if matrix_type is None:
        return list(_FILTER_MATRIX_TYPES)

    if N is None:
        raise ValueError("N must be provided when matrix_type is specified")

    if matrix_type == "RandomDense":
        from .construct_cascaded_paraunitary_matrix import (
            construct_cascaded_paraunitary_matrix,
        )

        return construct_cascaded_paraunitary_matrix(
            N, num_stages, matrix_type=stage_matrix_type
        )[0]

    if matrix_type == "Velvet":
        from .construct_cascaded_paraunitary_matrix import (
            construct_cascaded_paraunitary_matrix,
        )

        return construct_cascaded_paraunitary_matrix(
            N, num_stages, sparsity=sparsity, matrix_type=stage_matrix_type
        )[0]

    if matrix_type == "FromElementals":
        from .construct_paraunitary_from_elementals import (
            construct_paraunitary_from_elementals,
        )

        return construct_paraunitary_from_elementals(N, N * num_stages)[0]

    raise ValueError(
        f"Unknown matrix_type {matrix_type!r}. Supported: {_FILTER_MATRIX_TYPES}"
    )


def fdn_matrix_gallery(
    N: int | None = None,
    matrix_type: str | None = None,
) -> np.ndarray | list[str]:
    """Return a feedback matrix of the requested type, or list all type names.

    Args:
        N: Matrix size.  Ignored when ``matrix_type`` is ``None``.
        matrix_type: One of the supported type strings.  Pass ``None`` (or call
                     with no arguments) to get the list of all type names.

    Returns:
        Feedback matrix of shape ``(N, N)``, or a list of type-name strings.

    Example::

        fdn_matrix_gallery()             # → list of type strings
        fdn_matrix_gallery(4, "orthogonal")
        fdn_matrix_gallery(8, "Hadamard")
    """
    if matrix_type is None:
        return list(_MATRIX_TYPES)

    if N is None:
        raise ValueError("N must be provided when matrix_type is specified")

    if matrix_type == "orthogonal":
        return random_orthogonal(N)

    if matrix_type == "Hadamard":
        from scipy.linalg import hadamard

        return hadamard(N) / np.sqrt(N)

    if matrix_type == "circulant":
        r_fft = np.fft.fft(np.random.randn(N))
        r_fft /= np.abs(r_fft)
        r = np.real(np.fft.ifft(r_fft))
        direction = np.random.choice([-1, 1])
        return _circulant(r, direction)

    if matrix_type == "Householder":
        return householder_matrix(np.random.rand(N))

    if matrix_type == "parallel":
        return np.eye(N)

    if matrix_type == "permutation":
        P = np.eye(N)
        return P[np.random.permutation(N)]

    if matrix_type == "diagonallySimilarToOrthogonal":
        D = np.diag(np.random.randn(N))
        return np.linalg.solve(D, random_orthogonal(N)) @ D

    if matrix_type == "tinyRotation":
        import torch

        from ..auxiliary.tiny_rotation_matrix import tiny_rotation_matrix

        return tiny_rotation_matrix(N, 0.01, dtype=torch.float64).numpy()

    if matrix_type == "Anderson":
        from .anderson_matrix import anderson_matrix

        return anderson_matrix(N)

    if matrix_type in _SYSTEM_TYPES:
        raise ValueError(
            f"'{matrix_type}' returns a full FDN system; use fdn_system_gallery() instead."
        )

    raise ValueError(f"Unknown matrix_type {matrix_type!r}. Supported: {_MATRIX_TYPES}")


def fdn_system_gallery(
    N: int | None = None,
    system_type: str | None = None,
) -> FDNSystem | list[str]:
    """Return a full FDN system (A, B, C, D) of the requested type, or list all type names.

    Args:
        N: System order.  Ignored when ``system_type`` is ``None``.
        system_type: One of the supported type strings.  Pass ``None`` (or call
                     with no arguments) to get the list of all type names.

    Returns:
        :class:`FDNSystem` named tuple ``(A, B, C, D)``, or a list of type-name strings.

    Example::

        fdn_system_gallery()                         # → list of type strings
        fdn_system_gallery(8, "allpassInFDN")
    """
    if system_type is None:
        return list(_SYSTEM_TYPES)

    if N is None:
        raise ValueError("N must be provided when system_type is specified")

    if system_type == "series":
        from ..auxiliary.allpass import series_allpass

        g = np.random.rand(N) * 0.6 + 0.2
        A, B, C, D = series_allpass(g)
        return FDNSystem(A, B, C, D)

    if system_type == "nestedAllpass":
        from ..auxiliary.allpass import nested_allpass

        g = np.random.rand(N) * 0.6 + 0.2
        A, B, C, D = nested_allpass(g)
        return FDNSystem(A, B, C, D)

    if system_type == "polettiAllpass":
        from ..auxiliary.allpass import poletti_allpass

        A, B, C, D = poletti_allpass(0.7, random_orthogonal(N))
        return FDNSystem(A, B, C, D)

    if system_type == "homogeneousAllpass":
        from .allpass_FDN.homogeneous_allpass_fdn import homogeneous_allpass_fdn
        from .allpass_FDN.rand_admissible_homogeneous_allpass import (
            rand_admissible_homogeneous_allpass,
        )

        G = np.diag(np.random.uniform(0.9, 0.99, N))
        X = rand_admissible_homogeneous_allpass(G, (0.8, 0.99))
        A, B, C, D, _ = homogeneous_allpass_fdn(G, X)
        return FDNSystem(A, B, C, D)

    if system_type == "SchroederReverberator":
        from .schroeder_reverberator import schroeder_reverberator

        N_c = N // 2
        N_a = N - N_c
        comb_gain = np.random.uniform(0.5, 0.9, N_c) * np.random.choice([-1, 1], N_c)
        allpass_gain = np.random.uniform(0.2, 0.7, N_a)
        b = np.random.randn(N_c, 1) / np.sqrt(N_c)
        c = np.random.randn(1, N_c) / np.sqrt(N_c)
        d = 0.0
        A, B, C, D = schroeder_reverberator(allpass_gain, comb_gain, b, c, d)
        return FDNSystem(A, B, C, D)

    if system_type == "allpassInFDN":
        from .allpass_in_fdn import allpass_in_fdn

        g = np.random.uniform(-0.8, 0.8, N // 2)
        A_inner = random_orthogonal(N // 2)
        b_inner = np.random.randn(N // 2, 1) / np.sqrt(N // 2)
        c_inner = np.random.randn(1, N // 2) / np.sqrt(N // 2)
        A, B, C, D = allpass_in_fdn(g, A_inner, b_inner, c_inner, 0.0)
        return FDNSystem(A, B, C, D)

    raise ValueError(f"Unknown system_type {system_type!r}. Supported: {_SYSTEM_TYPES}")
