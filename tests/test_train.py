"""Tests for the four-step pyFDN.train API (require torch + flamo)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flamo")

import pyFDN  # noqa: E402
from pyFDN.generate.fdn_matrix_gallery import FDNBuild  # noqa: E402
from pyFDN.train import (  # noqa: E402
    Trainable,
    build_fdn,
    extract_build,
    flatness_from_magnitude,
    make_objective,
    train_fdn,
    trainable_from_build,
    with_decay,
)

# Tiny / CPU / fast optimization settings.
_FAST = {"expand": 64, "batch_size": 8, "lr": 3e-3, "device": "cpu"}


def _magnitude(model, nfft, n_in=1):
    """|H| at DFT bins from a magnitude-output model, summed over channels."""
    import torch

    x = torch.zeros(1, nfft, n_in)
    x[:, 0, :] = 1.0
    with torch.no_grad():
        return np.asarray(model(x).detach())[0].sum(axis=-1)


def _leaf(model, name):
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    for node in flamo_nodes_flat(flamo_model_to_nodes(model)):
        if node["type"] == "Leaf" and node["name"] == name:
            return node["module"]
    raise AssertionError(f"no {name!r} leaf in model")


# --- build + extract -------------------------------------------------------


def test_build_fdn_default_is_so_n_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # default draw must already be in SO(N)
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    b = extract_build(model, fs=48000.0)
    np.testing.assert_allclose(b.A.T @ b.A, np.eye(4), atol=1e-4)
    assert np.linalg.det(b.A) > 0


def test_extract_roundtrip_with_direct_always_present():
    model = build_fdn(N=4, rt=None, nfft=2**11, device="cpu", rng=3)
    b = extract_build(model, fs=48000.0)
    assert isinstance(b, FDNBuild)
    np.testing.assert_allclose(b.A.T @ b.A, np.eye(4), atol=1e-4)
    assert b.B.shape == (4, 1) and b.C.shape == (1, 4)
    # direct path always exists, zero by default
    assert b.D.shape == (1, 1)
    np.testing.assert_allclose(b.D, 0.0, atol=1e-6)
    assert b.filters is None  # lossless (rt=None)


def test_build_rt_sets_absorption_and_renders():
    model = build_fdn(N=6, rt=2.0, nfft=2**12, device="cpu", rng=1)
    ir = np.asarray(pyFDN.flamo_time_response(model, fs=48000)).reshape(-1)
    assert np.all(np.isfinite(ir))
    b = extract_build(model, fs=48000.0)
    assert b.filters is not None and b.filters.shape[1] == 6


def test_extracted_build_renders_through_build_to_flamo():
    model = build_fdn(N=4, rt=None, nfft=2**11, device="cpu", rng=5)
    b = extract_build(model, fs=48000.0)
    ir = pyFDN.flamo_time_response(
        pyFDN.build_to_flamo(b, nfft=2**12, device="cpu"), fs=48000
    )
    assert np.all(np.isfinite(np.asarray(ir)))


def test_trainable_from_build_threads_requires_grad():
    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    build = pyFDN.fdn_build_gallery(N=4, rt=None, rng=0)
    model = trainable_from_build(
        build,
        trainable=Trainable(input_gain=True, output_gain=False),
        nfft=2**10,
        device="cpu",
    )
    assert feedback_matrix_module(model).param.requires_grad is True
    assert _leaf(model, "input_gain").param.requires_grad is True
    assert _leaf(model, "output_gain").param.requires_grad is False


def test_det_negative_orthogonal_warns_and_projects():
    build = pyFDN.fdn_build_gallery(N=4, rt=None, rng=1)
    if np.linalg.det(build.A) > 0:
        build.A[:, -1] *= -1.0  # force det = -1 (not in SO(N))
    with pytest.warns(UserWarning, match="SO"):
        model = trainable_from_build(build, nfft=2**10, device="cpu")
    out = extract_build(model, fs=48000.0)
    np.testing.assert_allclose(out.A.T @ out.A, np.eye(4), atol=1e-4)
    assert np.linalg.det(out.A) > 0


# --- train -----------------------------------------------------------------


def test_colorless_improves_and_preserves_structure():
    nfft = 2**10
    model = build_fdn(N=4, rt=None, nfft=nfft, device="cpu", rng=0)
    init_twin = trainable_from_build(
        extract_build(model, fs=48000.0), nfft=nfft, output="magnitude", device="cpu"
    )
    init = flatness_from_magnitude(_magnitude(init_twin, nfft))
    delays0 = extract_build(model, fs=48000.0).delays

    log = train_fdn(model, make_objective("colorless"), max_epochs=15, rng=0, **_FAST)

    assert log.train_loss[-1] < log.train_loss[0]
    # the model now emits |H| (output-domain swap) -> flatter than init
    assert flatness_from_magnitude(_magnitude(model, nfft)) > init
    out = extract_build(model, fs=48000.0)
    np.testing.assert_allclose(out.A.T @ out.A, np.eye(4), atol=1e-4)
    np.testing.assert_array_equal(out.delays, delays0)  # delays frozen
    assert "_ColorlessSparsity" in log.loss_log and "mse_loss" in log.loss_log
    assert log.epochs_run == len(log.train_loss)


def test_train_is_reproducible():
    def run():
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=2)
        return train_fdn(
            model, make_objective("colorless"), max_epochs=8, rng=0, **_FAST
        )

    np.testing.assert_allclose(run().train_loss, run().train_loss, rtol=1e-6)


def test_train_rejects_oversized_batch():
    model = build_fdn(N=4, rt=None, nfft=2**9, device="cpu", rng=0)
    with pytest.raises(ValueError, match="batch_size"):
        train_fdn(
            model, make_objective("colorless"), expand=16, batch_size=64, device="cpu"
        )


def test_match_ir_runs_and_scores():
    nfft = 2**11
    target = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=7)
    target_ir = np.asarray(pyFDN.flamo_time_response(target, fs=48000)).reshape(-1)
    fresh = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=11)

    log = train_fdn(
        fresh,
        make_objective("match_ir", target=target_ir, mss_nfft=(256, 512)),
        max_epochs=4,
        rng=0,
        **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])
    out_ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(
                extract_build(fresh, fs=48000.0), nfft=nfft, device="cpu"
            ),
            fs=48000,
        )
    ).reshape(-1)
    assert np.isfinite(pyFDN.mr_stft_distance(out_ir, target_ir))


# --- analytic decay (the exact RT path) ------------------------------------


def test_with_decay_realizes_rt():
    build = extract_build(
        build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3), fs=48000.0
    )
    build = with_decay(build, 0.3)
    assert build.filters is not None and build.filters.shape == (1, 6, 6)

    ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(build, nfft=2**16, device="cpu"), fs=48000
        )
    ).reshape(-1)
    rt, _ = pyFDN.estimate_rt_bands(ir, 48000.0)
    assert 0.3 * 0.7 < float(np.nanmean(rt)) < 0.3 * 1.3
