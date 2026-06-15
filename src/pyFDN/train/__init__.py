"""Training pipeline for FDNs -- an explicit four-step API over flamo.

1. **build** a trainable flamo model from a config (:func:`build_fdn`, or
   :func:`trainable_from_build` from an existing :class:`~pyFDN.FDNBuild`).
2. **define** an objective from a mode + criteria + data (:func:`make_objective`).
3. **train** the model toward the objective (:func:`train_fdn`).
4. **extract** an :class:`~pyFDN.FDNBuild` / params / metadata
   (:func:`extract_build`, :class:`TrainLog`).

The dataclasses and numpy metrics here import without torch; the builder, engine
and criteria import torch/flamo lazily, so ``import pyFDN`` stays torch-free.
"""

from __future__ import annotations

from .build import MatrixParam, Trainable, build_fdn, trainable_from_build, with_decay
from .engine import TrainLog, train_fdn
from .extract import extract_build
from .metrics import (
    edc_l1,
    flatness_from_magnitude,
    magnitude_response,
    mr_stft_distance,
    octave_colouration,
    spectral_flatness,
)
from .objectives import Objective, ObjectiveMode, make_objective

__all__ = [
    # 1. build
    "build_fdn",
    "trainable_from_build",
    "with_decay",
    "Trainable",
    "MatrixParam",
    # 2. objective
    "make_objective",
    "Objective",
    "ObjectiveMode",
    # 3. train
    "train_fdn",
    "TrainLog",
    # 4. extract
    "extract_build",
    # metrics
    "spectral_flatness",
    "flatness_from_magnitude",
    "octave_colouration",
    "edc_l1",
    "mr_stft_distance",
    "magnitude_response",
]
