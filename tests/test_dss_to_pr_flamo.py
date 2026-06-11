"""Tests for FLAMO-first dss_to_pr_flamo / flamo_to_pr entrypoints."""

from __future__ import annotations

import numpy as np
import torch

from pyFDN.translate.dss_to_flamo import dss_to_flamo
from pyFDN.translate.dss_to_pr_direct import dss_to_pr_direct
from pyFDN.translate.dss_to_pr_flamo import (
    dss_to_pr_flamo,
    flamo_to_pr,
)


def _sort_by_pole(res, pol, pair):
    """Sort poles/residues/pair-flags canonically; backends order poles differently."""
    order = np.lexsort((np.abs(pol.imag), pol.real))
    return res[order], pol[order], pair[order]


def test_dss_to_pr_flamo_matches_direct_backend():
    delays = np.array([4, 5, 6], dtype=int)
    a = np.array([[0.25, -0.1, 0.05], [0.15, 0.3, -0.2], [0.1, 0.05, 0.2]])
    b = np.eye(3, 1)
    c = np.eye(1, 3)
    d = np.zeros((1, 1))

    res_ref, pol_ref, direct_ref, pair_ref, _ = dss_to_pr_direct(
        delays,
        a,
        b,
        c,
        d,
        mode="roots",
    )
    res_new, pol_new, direct_new, pair_new, _ = dss_to_pr_flamo(
        delays,
        a,
        b,
        c,
        d,
        Fs=1.0,
        nfft=1024,
        verbose=False,
    )

    res_ref, pol_ref, pair_ref = _sort_by_pole(res_ref, pol_ref, pair_ref)
    res_new, pol_new, pair_new = _sort_by_pole(res_new, pol_new, pair_new)

    np.testing.assert_allclose(pol_new, pol_ref, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res_new, res_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(direct_new, direct_ref, rtol=0, atol=0)
    np.testing.assert_array_equal(pair_new, pair_ref)


def test_flamo_to_pr_matches_dss_to_pr_flamo_wrapper():
    delays = np.array([2, 3], dtype=int)
    a = np.array([[0.25, -0.1], [0.15, 0.3]])
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    model = dss_to_flamo(
        A=a,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=1.0,
        nfft=1024,
        shell=False,
        dtype=torch.float64,
    )

    res_model, pol_model, direct_model, pair_model, meta_model = flamo_to_pr(
        model,
        verbose=False,
    )
    res_wrap, pol_wrap, direct_wrap, pair_wrap, _ = dss_to_pr_flamo(
        delays,
        a,
        b,
        c,
        d,
        Fs=1.0,
        nfft=1024,
        verbose=False,
    )

    res_model, pol_model, pair_model = _sort_by_pole(res_model, pol_model, pair_model)
    res_wrap, pol_wrap, pair_wrap = _sort_by_pole(res_wrap, pol_wrap, pair_wrap)

    np.testing.assert_allclose(pol_model, pol_wrap, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res_model, res_wrap, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(direct_model, direct_wrap, rtol=0, atol=0)
    np.testing.assert_array_equal(pair_model, pair_wrap)
    assert {"P", "B", "C", "D"}.issubset(set(meta_model["decomposition"].keys()))
