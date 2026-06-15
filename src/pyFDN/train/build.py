"""Step 1 of training: build a trainable flamo FDN model from a config.

``build_fdn`` turns a small config (delays/N, decay, what is trainable) into a
flamo ``Shell`` -- a real model you can render, inspect (:func:`pyFDN.flamo_time_response`,
:func:`pyFDN.plot_flamo_graph`), train, and extract. It does **not** require an
:class:`~pyFDN.FDNBuild`; ``trainable_from_build`` is the optional adapter for
when you already have one (e.g. from :func:`pyFDN.fdn_build_gallery` or a
decomposition).

The default topology matches the render path (:func:`pyFDN.dss_to_flamo`): an
orthogonal ``Matrix`` feedback, fixed integer delays with frozen first-order
absorption realizing ``rt`` (2 s by default; ``rt=None`` is lossless), trainable
input/output gains, and a direct path that **always exists** (zero by default).
Wiring goes through the shared :func:`pyFDN.auxiliary.flamo.assemble_fdn_core`,
so train and render topologies cannot drift.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

# Feedback-matrix parametrization (flamo ``dsp.Matrix`` map). "orthogonal" keeps
# the matrix on SO(N) during training (the colorless choice); "random" trains it
# unconstrained.
MatrixParam = Literal["orthogonal", "random"]


@dataclass(frozen=True)
class Trainable:
    """Which FDN parameter groups receive gradients during training.

    A property of the *model* (step 1), not the objective: each flag maps 1:1
    onto the ``requires_grad`` of the corresponding flamo module. Delays are
    always fixed integers (trainable/fractional delays are out of scope), so
    there is deliberately no ``delays`` flag.
    """

    feedback: bool = True
    input_gain: bool = True
    output_gain: bool = True
    direct: bool = False


def build_fdn(
    *,
    delays: np.ndarray | None = None,
    N: int | None = None,
    rt: float | tuple[float, float] | None = 2.0,
    matrix: MatrixParam = "orthogonal",
    feedback: np.ndarray | None = None,
    input_gain: np.ndarray | None = None,
    output_gain: np.ndarray | None = None,
    direct: float | np.ndarray = 0.0,
    trainable: Trainable | None = None,
    fs: float = 48000.0,
    nfft: int = 2**14,
    output: str = "time",
    device: Any = None,
    dtype: Any = None,
    rng: np.random.Generator | int | None = None,
) -> Any:
    """Build a trainable flamo ``Shell`` from a config.

    Parameters
    ----------
    delays : np.ndarray, optional
        Explicit integer delay lengths in samples. If omitted, ``N`` coprime
        delays are sampled (:func:`pyFDN.sample_delay_lengths`).
    N : int, optional
        Number of delay lines when ``delays`` is omitted.
    rt : float, (rt_dc, rt_nyquist), or None
        Reverberation time in seconds, realized as frozen in-loop first-order
        absorption. ``None`` builds a lossless FDN (the colorless prototype).
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    feedback : np.ndarray, optional
        Initial ``(N, N)`` feedback matrix; defaults to a random SO(N) matrix.
    input_gain, output_gain : np.ndarray, optional
        ``B`` (``(N, n_in)``) and ``C`` (``(n_out, N)``); default column/row of
        ones (SIMO/MISO).
    direct : float or np.ndarray
        Direct path ``D``; a scalar fills ``(n_out, n_in)``. The module always
        exists (trainable via ``Trainable(direct=True)``).
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`Trainable`).
    fs, nfft, output, device, dtype : see :func:`trainable_from_build`.
    rng : np.random.Generator, int, or None
        Seed for the sampled delays / default feedback matrix.

    Returns
    -------
    flamo.processor.system.Shell
        A trainable model (time output by default, so it renders immediately).
    """
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild
    from pyFDN.generate.sample_delay_lengths import sample_delay_lengths

    local_rng = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    if delays is not None:
        delays_arr = np.asarray(delays, dtype=int).ravel()
        n = int(delays_arr.size)
    elif N is not None:
        n = int(N)
        delays_arr = sample_delay_lengths(n, coprime=True, rng=local_rng)
    else:
        raise ValueError("provide either delays= or N=")

    if feedback is not None:
        a = np.asarray(feedback, dtype=float)
        if a.shape != (n, n):
            raise ValueError(f"feedback must have shape ({n}, {n}), got {a.shape}")
    else:
        a = _random_so_n(n, local_rng)

    # Default IO is "normalized" (ones / sqrt(N)): puts the initial |H| near unity
    # so a colorless objective starts well-conditioned. Override with input_gain/
    # output_gain for other layouts.
    b = (
        np.ones((n, 1)) / np.sqrt(n)
        if input_gain is None
        else np.asarray(input_gain, float).reshape(n, -1)
    )
    c = (
        np.ones((1, n)) / np.sqrt(n)
        if output_gain is None
        else np.atleast_2d(np.asarray(output_gain, float))
    )
    n_out, n_in = c.shape[0], b.shape[1]
    d = _resolve_direct(direct, n_out, n_in)

    filters = None
    if rt is not None:
        from pyFDN.auxiliary.acoustics import first_order_absorption

        rt_dc, rt_ny = _rt_pair(rt)
        filters = first_order_absorption(rt_dc, rt_ny, delays_arr, float(fs))

    build = FDNBuild(
        A=a,
        B=b,
        C=c,
        D=d,
        delays=delays_arr,
        fs=float(fs),
        filters=filters,
        post_eq=None,
    )
    return trainable_from_build(
        build,
        trainable=trainable,
        matrix=matrix,
        nfft=nfft,
        output=output,
        device=device,
        dtype=dtype,
    )


def trainable_from_build(
    build: FDNBuild,
    *,
    trainable: Trainable | None = None,
    matrix: MatrixParam = "orthogonal",
    nfft: int = 2**14,
    output: str = "time",
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build a trainable flamo ``Shell`` initialized from an ``FDNBuild``.

    The single place that turns FDN parameters into a trainable model;
    :func:`build_fdn` constructs an ``FDNBuild`` from its config and delegates
    here. The direct path is always present (from ``build.D``, or zeros), so the
    core is a ``Parallel`` -- colorless training therefore uses pyFDN's own
    sparsity loss (:func:`pyFDN.train.criteria.colorless_sparsity_loss`).

    Parameters
    ----------
    build : FDNBuild
        Initial FDN (``A``/``B``/``C``/``D``/``delays``/``fs`` + optional
        ``filters``/``post_eq``).
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`Trainable`).
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    nfft : int
        FFT size.
    output : str
        ``"time"`` (iFFT) or ``"magnitude"`` output layer. ``train_fdn`` resets
        this to match the objective, so the build-time choice only affects direct
        inspection of the returned model.
    device, dtype : optional
        Torch device / dtype (default cpu-or-cuda / float32).

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.auxiliary.flamo import (
        assemble_fdn_core,
        gain_module,
        matrix_module,
        sos_filter_module,
        wrap_fdn_shell,
    )

    trainable = trainable or Trainable()
    fs = float(build.fs)
    b = np.asarray(build.B, dtype=np.float64)
    c = np.asarray(build.C, dtype=np.float64)
    d = (
        np.asarray(build.D, dtype=np.float64)
        if build.D is not None
        else np.zeros((c.shape[0], b.shape[1]))
    )

    input_gain = gain_module(
        b, nfft, device=device, dtype=dtype, requires_grad=trainable.input_gain
    )
    output_gain = gain_module(
        c, nfft, device=device, dtype=dtype, requires_grad=trainable.output_gain
    )
    feedback = matrix_module(
        build.A,
        nfft,
        matrix_type=matrix,
        device=device,
        dtype=dtype,
        requires_grad=trainable.feedback,
    )
    # Direct path is ALWAYS wired (zero by default) so the same model serves any
    # objective; the core is therefore a Parallel.
    direct_gain = gain_module(
        d, nfft, device=device, dtype=dtype, requires_grad=trainable.direct
    )
    delays = _frozen_delays(
        np.asarray(build.delays, dtype=np.float64).ravel(), fs, nfft, device, dtype
    )

    # In-loop absorption (decay) is a frozen build property.
    loop_filter = (
        sos_filter_module(
            np.asarray(build.filters, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
        )
        if build.filters is not None
        else None
    )
    output_filter = (
        sos_filter_module(
            np.asarray(build.post_eq, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        if build.post_eq is not None
        else None
    )

    core = assemble_fdn_core(
        input_gain=input_gain,
        feedback=feedback,
        delays=delays,
        output_gain=output_gain,
        direct=direct_gain,
        loop_filter=loop_filter,
        output_filter=output_filter,
    )
    return wrap_fdn_shell(core, nfft=nfft, dtype=dtype, output=output)


def with_decay(
    build: FDNBuild,
    rt: float | tuple[float, float],
    *,
    rt_crossover: float | None = None,
) -> FDNBuild:
    """Return a copy of ``build`` with homogeneous decay realized analytically.

    Sets per-delay first-order absorption (:func:`pyFDN.first_order_absorption`)
    matching ``rt`` (a single value, or ``(rt_dc, rt_nyquist)``). Homogeneous
    decay is a similarity transform that does not change colouration, so this is
    the natural way to add decay to a colorless-optimized lossless build.
    """
    from pyFDN.auxiliary.acoustics import first_order_absorption

    rt_dc, rt_ny = _rt_pair(rt)
    filters = first_order_absorption(
        rt_dc, rt_ny, np.asarray(build.delays), float(build.fs), rt_crossover
    )
    return dataclasses.replace(build, filters=filters)


def _rt_pair(rt: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(rt, (tuple, list)):
        return float(rt[0]), float(rt[1])
    return float(rt), float(rt)


def _resolve_direct(direct: float | np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    arr = np.asarray(direct, dtype=float)
    if arr.ndim == 0:
        return np.full((n_out, n_in), float(arr))
    return arr.reshape(n_out, n_in)


def _random_so_n(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal matrix projected into SO(N) (det = +1).

    Landing in SO(N) means the orthogonal parametrization's preimage
    (``logm``) round-trips without the det<0 projection warning.
    """
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


def _frozen_delays(
    delay_samples: np.ndarray, fs: float, nfft: int, device: Any, dtype: Any
) -> Any:
    """Frozen integer parallelDelay from delay lengths in samples."""
    from pyFDN.auxiliary.flamo import delay_module

    return delay_module(
        np.asarray(delay_samples, dtype=np.float64) / float(fs),
        nfft,
        Fs=fs,
        device=device,
        dtype=dtype,
        isint=True,
        requires_grad=False,
    )
