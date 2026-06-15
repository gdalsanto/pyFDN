"""Step 3 of training: optimize a model toward an objective.

:func:`train_fdn` runs flamo's ``Trainer`` on a model from
:func:`pyFDN.build_fdn` (or :func:`pyFDN.trainable_from_build`) toward an
:class:`pyFDN.Objective`, **in place**, and returns a :class:`TrainLog`.
Extraction (:func:`pyFDN.extract_build`) and scoring (the metrics) are separate,
explicit steps so the user keeps full control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .objectives import Objective


@dataclass
class TrainLog:
    """Optimization metadata from a training run.

    Attributes
    ----------
    train_loss, valid_loss : list of float
        Per-epoch total loss on the train / validation split.
    loss_log : dict of str to list of float
        Per-criterion loss history, when flamo records it.
    epochs_run : int
        Epochs actually run (< ``max_epochs`` if early stopping triggered).
    stopped_early : bool
        Whether training stopped before ``max_epochs``.
    """

    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    loss_log: dict[str, list[float]] = field(default_factory=dict)
    epochs_run: int = 0
    stopped_early: bool = False


def train_fdn(
    model: Any,
    objective: Objective,
    *,
    max_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 200,
    expand: int = 1000,
    patience: int = 100,
    device: Any = None,
    dtype: Any = None,
    rng: np.random.Generator | int | None = None,
    log: bool = False,
    train_dir: str | None = None,
) -> TrainLog:
    """Train ``model`` toward ``objective`` (in place) and return a TrainLog.

    Sets the model's output layer to match ``objective.output_domain``, builds
    the objective's dataset + criteria, and runs flamo's ``Trainer``. The model
    is mutated in place; read the result back with :func:`pyFDN.extract_build`.

    Parameters
    ----------
    model : flamo Shell
        A trainable model from :func:`pyFDN.build_fdn` / ``trainable_from_build``.
        ``dtype`` should match the model's (default float32 on both).
    objective : Objective
        What to optimize (:func:`pyFDN.make_objective`).
    max_epochs, lr, batch_size, expand, patience : optimization settings.
    device, dtype : optional
        Torch device / dtype (default cpu / float32).
    rng : np.random.Generator, int, or None
        Seeds ``torch.manual_seed`` for reproducible training.
    log : bool
        If True, flamo logs/checkpoints to ``train_dir`` (which must exist);
        default False keeps training side-effect-free.
    train_dir : str, optional
        Checkpoint directory (only used when ``log=True``).

    Notes
    -----
    Swapping the output layer is a deliberate, visible mutation: after a
    ``colorless``/``match_magnitude`` run the model emits ``|H|``; after
    ``match_ir``/``decay`` it emits a time response.
    """
    import torch
    from flamo.optimize.dataset import load_dataset
    from flamo.optimize.trainer import Trainer

    dev = "cpu" if device is None else device
    torch_dtype = torch.float32 if dtype is None else dtype

    seed = _resolve_seed(rng)
    if seed is not None:
        torch.manual_seed(seed)

    # flamo's load_dataset uses an 80/20 split and drops partial batches, so both
    # splits need a full batch -- fail clearly rather than as a deep ZeroDivisionError.
    train_size = int(expand * 0.8)
    valid_size = expand - train_size
    if min(train_size, valid_size) < batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) too large for expand ({expand}): the 80/20 "
            f"split yields train={train_size}, valid={valid_size} and flamo drops "
            "partial batches. Increase expand (>= 5*batch_size) or lower batch_size."
        )

    nfft = int(model.get_inputLayer().nfft)
    n_in, n_out = _io_channels(model)

    dataset, criteria, output_domain = objective.build(
        nfft=nfft, n_in=n_in, n_out=n_out, device=dev, dtype=torch_dtype, expand=expand
    )
    _set_output_domain(model, output_domain, nfft, torch_dtype)

    train_loader, valid_loader = load_dataset(
        dataset, batch_size=batch_size, device=dev
    )
    trainer = Trainer(
        model,
        max_epochs=max_epochs,
        lr=lr,
        patience=patience,
        device=dev,
        train_dir=train_dir,
        log=log,
    )
    for criterion, alpha, requires_model in criteria:
        trainer.register_criterion(criterion, alpha, requires_model)
    trainer.train(train_loader, valid_loader)

    epochs_run = len(trainer.train_loss)
    return TrainLog(
        train_loss=[float(x) for x in trainer.train_loss],
        valid_loss=[float(x) for x in trainer.valid_loss],
        loss_log={
            k: [float(x) for x in v]
            for k, v in getattr(trainer, "train_loss_log", {}).items()
        },
        epochs_run=epochs_run,
        stopped_early=epochs_run < max_epochs,
    )


def _resolve_seed(rng: np.random.Generator | int | None) -> int | None:
    if rng is None:
        return None
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    return int(rng.integers(0, 2**31 - 1))


def _io_channels(model: Any) -> tuple[int, int]:
    """``(n_in, n_out)`` read from the model's input/output gain leaves."""
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    leaves = [
        n for n in flamo_nodes_flat(flamo_model_to_nodes(model)) if n["type"] == "Leaf"
    ]

    def _leaf(name: str) -> Any:
        matches = [n["module"] for n in leaves if n["name"] == name]
        if len(matches) != 1:
            raise ValueError(
                f"model must contain exactly one {name!r} leaf; found {len(matches)}"
            )
        return matches[0]

    return int(_leaf("input_gain").param.shape[1]), int(
        _leaf("output_gain").param.shape[0]
    )


def _set_output_domain(model: Any, output_domain: str, nfft: int, dtype: Any) -> None:
    """Reset the Shell's output layer to time (iFFT) or magnitude (``|.|``)."""
    import torch
    from flamo.processor import dsp

    if output_domain == "time":
        layer: Any = dsp.iFFT(nfft, dtype=dtype)
    elif output_domain == "magnitude":
        layer = dsp.Transform(transform=torch.abs, dtype=dtype)
    else:
        raise ValueError(
            f"output_domain must be 'time' or 'magnitude', got {output_domain!r}"
        )
    model.set_outputLayer(layer)
