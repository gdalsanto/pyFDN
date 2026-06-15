"""Step 2 of training: define what to optimize (the objective).

An :class:`Objective` is small, frozen data describing a training run -- a
*mode* (colorless / match_ir / match_magnitude / decay), the target data, and
the loss criteria. :func:`make_objective` is the factory; the engine
(:func:`pyFDN.train_fdn`) consumes the ``(dataset, criteria, output_domain)``
that :meth:`Objective.build` produces.

Trainability is **not** part of the objective -- it is a property of the model
(see :class:`pyFDN.Trainable` / :func:`pyFDN.build_fdn`). flamo and torch are
imported lazily inside :meth:`Objective.build`, so this module imports without
torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ObjectiveMode = Literal["colorless", "match_ir", "match_magnitude"]

# (criterion, alpha, requires_model) as flamo's Trainer.register_criterion wants;
# used both for the defaults and for a user-supplied criteria override.
Criterion = tuple[Any, float, bool]


@dataclass(frozen=True)
class Objective:
    """What a training run optimizes (step 2).

    Exactly one ``target`` payload is used per :attr:`mode`:

    * ``"colorless"`` -- flatten the magnitude response (no target).
    * ``"match_ir"`` -- match a time-domain impulse response (``target`` = IR).
    * ``"match_magnitude"`` -- match a one-sided spectrum (``target`` = ``|H|``).

    (Decay/RT is not an objective: it is a build property -- see
    :func:`pyFDN.build_fdn` ``rt=`` and :func:`pyFDN.with_decay`.)

    Parameters
    ----------
    mode : str
        One of :data:`ObjectiveMode`.
    target : np.ndarray, float, or None
        Target payload for the mode (see above).
    criteria : list of (criterion, alpha, requires_model), optional
        Override the default loss list with your own flamo criteria.
    sparsity_alpha, mag_alpha : float
        Default-loss weights (colorless: sparsity vs magnitude MSE).
    mss_nfft : tuple of int
        Multi-resolution STFT window sizes for ``match_ir``.
    fs : float
        Sampling rate in Hz.
    """

    mode: ObjectiveMode
    target: Any = None
    criteria: list[Criterion] | None = None
    sparsity_alpha: float = 1.0
    mag_alpha: float = 1.0
    mss_nfft: tuple[int, ...] = (256, 512, 1024)
    fs: float = 48000.0

    def __post_init__(self) -> None:
        if self.mode not in ("colorless", "match_ir", "match_magnitude"):
            raise ValueError(f"unknown objective mode {self.mode!r}")
        if self.mode in ("match_ir", "match_magnitude") and self.target is None:
            raise ValueError(f"Objective(mode={self.mode!r}) requires target=")

    @property
    def output_domain(self) -> str:
        """The model output layer this objective measures in."""
        return "magnitude" if self.mode in ("colorless", "match_magnitude") else "time"

    def build(
        self,
        *,
        nfft: int,
        n_in: int,
        n_out: int,
        device: Any,
        dtype: Any,
        expand: int,
    ) -> tuple[Any, list[Criterion], str]:
        """Return ``(dataset, criteria, output_domain)`` for this objective.

        flamo datasets/losses are imported here so the module stays torch-free.
        A user-supplied :attr:`criteria` replaces the default loss list.
        """
        dataset = self._dataset(
            nfft=nfft, n_in=n_in, n_out=n_out, device=device, dtype=dtype, expand=expand
        )
        criteria = (
            self.criteria
            if self.criteria is not None
            else self._default_criteria(nfft=nfft, device=device)
        )
        return dataset, criteria, self.output_domain

    # -- internals ---------------------------------------------------------

    def _dataset(
        self,
        *,
        nfft: int,
        n_in: int,
        n_out: int,
        device: Any,
        dtype: Any,
        expand: int,
    ) -> Any:
        import torch

        if self.mode == "colorless":
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
        target = self._target_tensor(nfft=nfft, n_out=n_out, device=device, dtype=dtype)
        return Dataset(
            input=impulse, target=target, expand=expand, device=device, dtype=dtype
        )

    def _target_tensor(self, *, nfft: int, n_out: int, device: Any, dtype: Any) -> Any:
        import torch

        if self.mode == "match_magnitude":
            arr = np.asarray(self.target, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            return torch.as_tensor(arr[None], device=device, dtype=dtype)

        ir = np.asarray(self.target, dtype=np.float64)
        if ir.ndim == 1:
            ir = ir[:, None]
        buf = np.zeros((nfft, n_out))
        length = min(ir.shape[0], nfft)
        if ir.shape[1] == n_out:
            buf[:length, :] = ir[:length, :]
        else:  # broadcast a single-channel target across outputs
            buf[:length, :] = ir[:length, :1]
        return torch.as_tensor(buf[None], device=device, dtype=dtype)

    def _default_criteria(self, *, nfft: int, device: Any) -> list[Criterion]:
        if self.mode == "colorless":
            from flamo.optimize.loss import mse_loss

            from .criteria import colorless_sparsity_loss

            return [
                (mse_loss(nfft=nfft, device=device), float(self.mag_alpha), False),
                (colorless_sparsity_loss(), float(self.sparsity_alpha), True),
            ]
        if self.mode == "match_magnitude":
            from flamo.optimize.loss import mse_loss

            return [(mse_loss(nfft=nfft, device=device), 1.0, False)]
        # match_ir
        from flamo.optimize.loss import mss_loss

        criterion = mss_loss(
            nfft=list(self.mss_nfft), sample_rate=int(self.fs), device=device
        )
        return [(criterion, 1.0, False)]


def make_objective(
    mode: ObjectiveMode,
    *,
    target: Any = None,
    criteria: list[Criterion] | None = None,
    sparsity_alpha: float = 1.0,
    mag_alpha: float = 1.0,
    mss_nfft: tuple[int, ...] = (256, 512, 1024),
    fs: float = 48000.0,
) -> Objective:
    """Construct an :class:`Objective` for ``mode`` (step 2 of training).

    Examples
    --------
    ``make_objective("colorless", sparsity_alpha=1.0)``,
    ``make_objective("match_ir", target=ir, mss_nfft=(256, 512))``,
    ``make_objective("match_magnitude", target=mag)``. Pass ``criteria=`` to
    override the default loss list with your own flamo criteria.
    """
    return Objective(
        mode=mode,
        target=target,
        criteria=criteria,
        sparsity_alpha=sparsity_alpha,
        mag_alpha=mag_alpha,
        mss_nfft=tuple(mss_nfft),
        fs=fs,
    )
