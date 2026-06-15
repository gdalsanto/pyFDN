"""Step 4 of training: read a trained flamo model back into an FDNBuild."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild


def extract_build(model: Any, *, fs: float | None = None) -> FDNBuild:
    """Extract trained parameters into a plain :class:`~pyFDN.FDNBuild`.

    Walks the trained flamo ``Shell`` (:func:`pyFDN.flamo_model_to_fdn_parameters`),
    applying each module's ``map`` -- so an orthogonal feedback matrix comes back
    as its realized SO(N) matrix and the (always-present) direct path as ``D``.
    The returned build renders / analyzes / decomposes like any other.

    Parameters
    ----------
    model : flamo Shell
        Trained model from :func:`pyFDN.train_fdn`.
    fs : float, optional
        Sampling rate to record when it cannot be read from the model.

    Returns
    -------
    FDNBuild
        ``A``, ``B``, ``C``, ``D``, ``delays`` (rounded ints), ``fs``, plus
        ``filters`` (in-loop SOS) and ``post_eq`` (output SOS) when present.

    For raw, un-converted parameters use
    :func:`pyFDN.flamo_model_to_fdn_parameters` directly.
    """
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_fdn_parameters
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

    p = flamo_model_to_fdn_parameters(model)
    return FDNBuild(
        A=p.A,
        B=p.B,
        C=p.C,
        D=p.D,
        delays=p.delays,
        fs=float(p.fs if p.fs is not None else (fs or 0.0)),
        filters=p.attenuation_sos,
        post_eq=p.post_eq_sos,
    )
