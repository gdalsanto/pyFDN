"""Objective setup for :func:`pyFDN.train_fdn`: map a training *mode* (plus
optional target data) to a flamo dataset, loss criteria, and the model output
domain to measure in.

The mode alone fixes the loss, the output domain, and what ``target`` data it
needs:

==========================  ====================================  ============
mode                        loss                                  target
==========================  ====================================  ============
``"colorless"``             magnitude MSE vs a flat response      none
``"match_magnitude"``       magnitude MSE vs ``|H|``              ``|H|``
``"match_spectrogram"``     multi-resolution STFT (``mss``)       impulse resp.
``"match_mel_spectrogram"`` mel multi-resolution STFT             impulse resp.
==========================  ====================================  ============

Every mode also adds a feedback-matrix sparsity penalty
(:func:`colorless_sparsity_loss`, weight ``sparsity_alpha``, default 0.2; 0
disables it) that biases the mixing matrix toward dense, colorless mixing -- it
only bites when the feedback matrix is trainable.

flamo/torch are imported lazily inside :func:`build_objective` and
:func:`colorless_sparsity_loss`, so this module imports without torch.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Objective = Literal[
    "colorless",
    "match_magnitude",
    "match_spectrogram",
    "match_mel_spectrogram",
]

# (criterion, alpha, requires_model) tuples for flamo's Trainer.register_criterion.
Criterion = tuple[Any, float, bool]

# mode -> (target_kind, output_domain).
#   target_kind: "flat" (no data), "magnitude" (|H| data), "time" (impulse resp.)
_MODES: dict[str, tuple[str, str]] = {
    "colorless": ("flat", "magnitude"),
    "match_magnitude": ("magnitude", "magnitude"),
    "match_spectrogram": ("time", "time"),
    "match_mel_spectrogram": ("time", "time"),
}


def output_domain(mode: str) -> str:
    """The model output layer a mode measures in (``"time"`` or ``"magnitude"``)."""
    if mode not in _MODES:
        raise ValueError(
            f"unknown training mode {mode!r}; choose from {sorted(_MODES)}"
        )
    return _MODES[mode][1]


def colorless_sparsity_loss() -> Any:
    """A flamo-compatible sparsity penalty on the FDN feedback matrix.

    Same role and formula as flamo's ``sparsity_loss`` (the colorless penalty of
    *Optimizing Tiny Colorless FDNs*, Dal Santo et al.), but it locates the
    feedback matrix through pyFDN's graph walk
    (:func:`pyFDN.auxiliary.flamo_graph.feedback_matrix_module`), which resolves
    it for both a plain ``Series`` core and a ``Parallel`` core (the FDN summed
    with an always-present direct path). flamo's own ``sparsity_loss`` hard-codes
    attribute paths that assume a ``.mixing_matrix`` on the Parallel branch, so
    it raises once the direct path is always present.

    ``A = map(param)`` is computed **in-graph** so gradients flow. The returned
    ``nn.Module`` has the ``(y_pred, y_target, model)`` signature flamo's
    ``Trainer`` uses for ``requires_model=True`` criteria.
    """
    import torch.nn as nn

    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    class _ColorlessSparsity(nn.Module):
        def forward(self, y_pred: Any, y_target: Any, model: Any) -> Any:
            module = feedback_matrix_module(model)
            a = module.map(module.param)
            n = a.shape[-1]
            root_n = float(np.sqrt(n))
            # 0 when |A| is maximally dense (good mixing), 1 when fully sparse.
            return -(a.abs().sum() - n * root_n) / (n * (root_n - 1.0))

    return _ColorlessSparsity()


def build_objective(
    mode: str,
    *,
    target: Any,
    criteria: list[Criterion] | None,
    sparsity_alpha: float,
    mss_nfft: tuple[int, ...],
    fs: float,
    nfft: int,
    n_in: int,
    n_out: int,
    device: Any,
    dtype: Any,
    expand: int,
) -> tuple[Any, list[Criterion], str]:
    """Return ``(dataset, criteria, output_domain)`` for a training ``mode``.

    Maps the mode to its flamo dataset (with ``target`` shaped appropriately)
    and the full criteria list -- the mode's primary loss plus the sparsity
    regularizer -- unless a caller-supplied ``criteria`` override replaces it.
    flamo is imported here so the module stays torch-free.
    """
    if mode not in _MODES:
        raise ValueError(
            f"unknown training mode {mode!r}; choose from {sorted(_MODES)}"
        )
    target_kind, domain = _MODES[mode]
    if target_kind != "flat" and target is None:
        raise ValueError(f"mode {mode!r} requires target=")
    dataset = _dataset(target_kind, target, nfft, n_in, n_out, device, dtype, expand)
    if criteria is not None:
        return dataset, criteria, domain

    crit: list[Criterion] = [
        (_primary_loss(mode, nfft, mss_nfft, fs, device), 1.0, False)
    ]
    if sparsity_alpha > 0:
        crit.append((colorless_sparsity_loss(), float(sparsity_alpha), True))
    return dataset, crit, domain


def _dataset(
    target_kind: str,
    target: Any,
    nfft: int,
    n_in: int,
    n_out: int,
    device: Any,
    dtype: Any,
    expand: int,
) -> Any:
    import torch

    if target_kind == "flat":
        from flamo.optimize.dataset import DatasetColorless

        return DatasetColorless(
            input_shape=(1, nfft, n_in),
            target_shape=(1, nfft // 2 + 1, n_out),
            expand=expand,
            device=device,
            dtype=dtype,
        )

    from flamo.optimize.dataset import Dataset

    impulse = torch.zeros((1, nfft, n_in), device=device, dtype=dtype)
    impulse[:, 0, :] = 1.0

    # Shape target to (1, rows, n_out): rows = its own length for a magnitude
    # target, or nfft (zero-padded/truncated) for a time target; a single-channel
    # target is broadcast across outputs.
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != n_out:
        arr = np.repeat(arr[:, :1], n_out, axis=1)
    rows = arr.shape[0] if target_kind == "magnitude" else nfft
    buf = np.zeros((rows, n_out))
    length = min(arr.shape[0], rows)
    buf[:length, :] = arr[:length, :]

    tgt = torch.as_tensor(buf[None], device=device, dtype=dtype)
    return Dataset(input=impulse, target=tgt, expand=expand, device=device, dtype=dtype)


def _primary_loss(
    mode: str, nfft: int, mss_nfft: tuple[int, ...], fs: float, device: Any
) -> Any:
    if mode in ("colorless", "match_magnitude"):
        from flamo.optimize.loss import mse_loss

        return mse_loss(nfft=nfft, device=device)
    elif mode == "match_spectrogram":
        from flamo.optimize.loss import mss_loss

        return mss_loss(nfft=list(mss_nfft), sample_rate=int(fs), device=device)
    elif mode == "match_mel_spectrogram":
        from flamo.optimize.loss import mel_mss_loss  # match_mel_spectrogram

        return mel_mss_loss(nfft=list(mss_nfft), sample_rate=int(fs), device=device)
