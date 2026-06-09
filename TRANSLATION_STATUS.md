# pyFDN — Translation & Modernisation Status

This file tracks the translation of [fdnToolbox (MATLAB)](https://github.com/SebastianJiroSchlecht/fdnToolbox)
to pyFDN (Python).  The goal is modernisation, not a 1-to-1 port: architecture has
shifted to NumPy/SciPy primitives, FLAMO for the differentiable DSP graph, and
marimo for interactive examples.

**Environment:** `conda activate seb312` · Python 3.12  
**Run tests:** `/opt/anaconda3/envs/seb312/bin/python -m pytest tests/ --ignore=tests/test_marimo_examples_run.py`

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — translated and tested |
| 🔜 | Remaining — planned or in-progress |
| ⏭ | Superseded — covered by FLAMO or another module |
| ⛔ | Skip — symbolic math, external tools, or out of scope |

---

## `GraphicEQ/` — 6/6 done ✅

| MATLAB | Python | Notes |
|--------|--------|-------|
| `shelvingFilter.m` | `graphicEQ/shelving_filter.py` | Eqs (18)/(20) Välimäki & Reiss 2016 |
| `bandpassFilter.m` | `graphicEQ/bandpass_filter.py` | Eq (29) Välimäki & Reiss 2016 |
| `graphicEQ.m` | `graphicEQ/graphic_eq.py` | 11 biquad sections: flat + low-shelf + 8 peaking + high-shelf |
| `probeSOS.m` | `graphicEQ/probe_sos.py` | `freqz` returns `(w, h)` order (not `(h, w)`) |
| `designGEQ.m` | `graphicEQ/design_geq.py` | `lsqlin` → `scipy.optimize.lsq_linear` |
| `absorptionGEQ.m` | `graphicEQ/absorption_geq.py` | Output shape `(num_delays, 1, 11, 6)`, `a0` normalised to 1 |

`hertz2rad.m` is covered by `auxiliary/utils.py::hertz_to_rad`.

---

## `Generate/` — 19/21 done, 2 deferred ✅/🔜

| MATLAB | Python | Notes |
|--------|--------|-------|
| `randomOrthogonal.m` | `generate/random_orthogonal.py` | ✅ |
| `householderMatrix.m` | `generate/householder_matrix.py` | ✅ |
| `nearestOrthogonal.m` | `generate/nearest_orthogonal.py` | ✅ SVD Procrustes |
| `nearestSignAgnosticOrthogonal.m` | `generate/nearest_sign_agnostic_orthogonal.py` | ✅ Inline Sinkhorn-Knopp |
| `completeOrthogonal.m` | `generate/complete_orthogonal.py` | ✅ Uses SVD of A for consistency |
| `degreeOneLossless.m` | `generate/degree_one_lossless.py` | ✅ Output shape `(N, N, 2)` |
| `fdnMatrixGallery.m` | `generate/fdn_matrix_gallery.py` | ✅ 10/12 types implemented |
| `AndersonMatrix.m` | `generate/anderson_matrix.py` | ✅ |
| `interpolateOrthogonal.m` | `auxiliary/math.py::interpolate_orthogonal` | ✅ |
| `constructCascadedParaunitaryMatrix.m` | `generate/construct_cascaded_paraunitary_matrix.py` | ✅ |
| `constructVelvetFeedbackMatrix.m` | `generate/construct_velvet_feedback_matrix.py` | ✅ |
| `shiftMatrix.m` | `generate/shift_matrix.py` | ✅ |
| `shiftMatrixDistribute.m` | `generate/shift_matrix_distribute.py` | ✅ |
| `randomMatrixShift.m` | `generate/random_matrix_shift.py` | ✅ |
| `isAlmostZero.m` | `generate/is_almost_zero.py` | ✅ |
| `isAllpass.m` | `auxiliary/allpass.py::is_allpass` | ✅ |
| `isParaunitary.m` | `auxiliary/allpass.py::is_paraunitary` | ✅ |
| `isUniallpass.m` | `auxiliary/allpass.py::is_uniallpass` | ✅ |
| `homogeneousAllpassFDN.m` | `generate/allpass_FDN/homogeneous_allpass_fdn.py` | ✅ |
| `constructDelayMatrix.m` | — | ⏭ Use FLAMO `delay_module()` directly |
| `constructDelayFeedbackMatrix.m` | — | ⏭ Use FLAMO concatenation |

**In-gallery types deferred (raise `NotImplementedError`):**

| Gallery type | Reason |
|---|---|
| `allpassInFDN` | Use `generate/allpass_FDN/` submodule instead |
| `SchroederReverberator` | Use `generate/allpass_FDN/` submodule instead |

**Remaining generate functions:**

| MATLAB | Python target | Priority |
|--------|--------------|---------|
| `diagonallyEquivalent.m` | `generate/diagonally_equivalent.py` | 🔜 P2 |
| `isDiagonallyEquivalentToOrthogonal.m` | `generate/is_diagonally_equivalent_to_orthogonal.py` | 🔜 P2 |
| `isDiagonallySimilarToOrthogonal.m` | `generate/is_diagonally_similar_to_orthogonal.py` | 🔜 P2 |
| `constructParaunitaryFromElementals.m` | extend `construct_cascaded_paraunitary_matrix.py` | 🔜 P2 |
| `mroots.m` | extend `auxiliary/math.py` | 🔜 P2 |
| `minRankChoiceBruteForce.m` | — | ⛔ research-specific |

---

## `Translate/` — 9/11 done, 1 remaining, 1 skip ✅/🔜

| MATLAB | Python | Notes |
|--------|--------|-------|
| `dss2impz.m` | `translate/dss_to_impz.py` | ✅ |
| `dss2pr.m` | `translate/dss_to_pr_flamo.py` | ✅ FLAMO-based path |
| `dss2pr_direct.m` | `translate/dss_to_pr_direct.py` | ✅ |
| `dss2ss.m` | `translate/dss_to_ss.py` | ✅ |
| `dss2tf.m` | `translate/dss_to_tf.py` | ✅ |
| `impz2res.m` | `translate/impz_to_res.py` | ✅ |
| `mtf2impz.m` | `translate/mtf_to_impz.py` | ✅ |
| `pr2impz.m` | `translate/pr_to_impz.py` | ✅ |
| `dss2impzTransposed.m` | — | ⏭ Superseded by FLAMO's flexible graph |
| `dss2res.m` | `translate/dss_to_res.py` | 🔜 **P1** — modal excitation: pre-computed poles → residues (2024 paper) |
| `dss2tfSym.m` | — | ⛔ symbolic, skip |

---

## `Auxiliary/` — partial, ongoing

Key functions mapped:

| MATLAB | Python | Notes |
|--------|--------|-------|
| `absorptionFilters.m` | `auxiliary/acoustics.py::absorption_filters` | ✅ |
| `onePoleAbsorption.m` | `auxiliary/acoustics.py::one_pole_absorption` | ✅ |
| `absorption2T60.m` | `auxiliary/acoustics.py::absorption_to_rt` | ✅ |
| `RT602slope.m` | `auxiliary/acoustics.py::rt_to_slope` | ✅ |
| `EDC.m` | `auxiliary/acoustics.py::edc` | ✅ |
| `processFDN.m` | `process.py::process_fdn` | ✅ |
| `matrixConvolution.m` | `auxiliary/math.py::matrix_convolution` | ✅ |
| `matrixDelayApproximation.m` | `auxiliary/delay.py::matrix_delay_approximation` | ✅ |
| `mgrpdelay.m` | `auxiliary/delay.py::mgrpdelay` | ✅ |
| `ms2smp.m` | `auxiliary/delay.py::ms_to_smp` | ✅ |
| `detPolynomial.m` | `auxiliary/math.py::det_polynomial` | ✅ |
| `generalCharPoly.m` | `auxiliary/math.py::general_char_poly` | ✅ |
| `matrix_polyder.m` | `auxiliary/math.py::matrix_polyder` | ✅ |
| `matrix_polyval.m` | `auxiliary/math.py::matrix_polyval` | ✅ |
| `negpolyder.m` | `auxiliary/math.py::negpolyder` | ✅ |
| `outerSumApproximation.m` | `auxiliary/math.py::outer_sum_approximation` | ✅ |
| `polyDegree.m` | `auxiliary/math.py::poly_degree` | ✅ |
| `polydiag.m` | `auxiliary/math.py::polydiag` | ✅ |
| `poleBoundaries.m` | `auxiliary/utils.py::pole_boundaries` | ✅ |
| `skew.m` | `auxiliary/utils.py::skew` | ✅ |
| `hertz2unit.m` / `rad2hertz.m` | `auxiliary/utils.py` | ✅ |
| `plotImpulseResponseMatrix.m` | `auxiliary/plot.py::plot_impulse_response_matrix` | ✅ |
| `plotSystemMatrix.m` | `auxiliary/plot.py::plot_system_matrix` | ✅ |
| `tinyRotationMatrix.m` | `auxiliary/tiny_rotation_matrix.py` | ✅ (torch-based) |

**Remaining auxiliary functions (audit FLAMO coverage first):**

| MATLAB | Python target | Priority | Notes |
|--------|--------------|---------|-------|
| `loopTF.m` | `auxiliary/loop_tf.py` | 🔜 P2 | Check FLAMO frequency-response first |
| `tfMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | Check FLAMO |
| `firMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | Check FLAMO |
| `poleQuality.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | Audit FLAMO optimisation coverage |
| `refinePolePositions.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | As above |
| `toDiagonalSimilarCanonicalForm.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | As above |
| `realLogOfNormalMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | |
| `lowRankApprox.m` | `auxiliary/math.py` (extend) | 🔜 P3 | |
| `adjPoly.m` / `adjugate.m` | `auxiliary/math.py` (extend) | 🔜 P3 | |
| `zFDNloop.m` / `zFDNloopSimple.m` | — | ⏭ FLAMO graph |
| `zSOS.m` / `zFilter.m` / `zDelay.m` etc. | — | ⏭ FLAMO modules |
| `processFDN.m` / `processTransposedFDN.m` | `process.py` | ✅ / ⏭ |
| `decayFitNet2InitialLevel.m` | — | ⛔ Standalone Python package exists |
| `printMatLatex.m` / `printMatSyntax.m` | — | ⛔ skip |
| `generalCharPolySym.m` / `mpoly2sym.m` | — | ⛔ symbolic, skip |

---

## `DSP/` — 1/4 done 🔜

| MATLAB | Python | Priority | Notes |
|--------|--------|---------|-------|
| `feedbackDelay.m` | `dsp/feedback_delay.py` | ✅ | |
| `complexOscillatorBank.m` | `dsp/complex_oscillator_bank.py` | 🔜 P1 | Modulation bank for time-varying FDNs |
| `dfiltMatrix.m` | `dsp/dfilt_matrix.py` | 🔜 P1 | Apply filter bank to matrix signal |
| `timeVaryingMatrix.m` | `examples/example_time_varying_fdn.py` | 🔜 P1 | Processing script, not a class |

---

## `dspGraph/` — superseded ⏭

All 29 files superseded by FLAMO (`auxiliary/flamo.py`, `auxiliary/flamo_graph.py`,
`translate/dss_to_flamo.py`).  No translation needed.

---

## Examples — 17 + 9 allpass done ✅

| Example | Covers |
|---------|--------|
| `example_vanilla_FDN.py` | Basic FDN |
| `example_colorless_FDN.py` | Colorless FDN design |
| `example_process_fdn.py` | `process_fdn`, `dss_to_impz` |
| `example_coupled_rooms.py` | Coupled rooms FDN |
| `example_absorption_filters.py` | FIR / one-pole absorption |
| `example_one_pole_absorption.py` | `one_pole_absorption` |
| `example_delay_matrix_density.py` | Echo density |
| `example_interpolate_matrix.py` | `interpolate_orthogonal` |
| `example_dss_to_pr_direct.py` | DSS → poles/residues (direct) |
| `example_dss_to_pr_flamo.py` | DSS → poles/residues (FLAMO) |
| `example_dss_to_ss.py` | DSS → state-space |
| `example_dss_to_tf.py` | DSS → transfer function |
| `example_sdn_coefficients.py` | Scattering delay network |
| `example_graphic_eq.py` | `design_geq` — ±10 dB target + freq response |
| `example_absorption_geq.py` | `absorption_geq` — FDN simulation + EDC/T60 |
| `example_fdn_matrix_gallery.py` | All 12 gallery types — heatmaps + losslessness |
| `example_nearest_sign_agnostic_orthogonal.py` | Sign-agnostic Procrustes vs. naive |
| `allpass_FDN/` (9 files) | All allpass FDN variants |

**Remaining examples:**

| Example | Priority | Notes |
|---------|---------|-------|
| `example_time_varying_fdn.py` | 🔜 P1 | Depends on `complex_oscillator_bank.py` + `dfilt_matrix.py` |

---

## Priority summary

### P1 — implement next

1. `translate/dss_to_res.py` — modal excitation (pre-computed poles → residues, 2024 paper)
2. `dsp/complex_oscillator_bank.py` — time-varying modulation
3. `dsp/dfilt_matrix.py` — filter-bank matrix application
4. `examples/example_time_varying_fdn.py` — end-to-end time-varying FDN demo

### P2 — after FLAMO audit

- `loopTF`, `tfMatrix`, `firMatrix` — first check how much FLAMO's frequency evaluation covers
- Pole analysis: `poleQuality`, `refinePolePositions`, `toDiagonalSimilarCanonicalForm`
- Matrix property checks: `diagonallyEquivalent`, `isDiagonallyEquivalentToOrthogonal`, `isDiagonallySimilarToOrthogonal`
- `constructParaunitaryFromElementals` — extend `construct_cascaded_paraunitary_matrix.py`

### P3 / skip

- Symbolic math (`dss2tfSym`, `generalCharPolySym`, `mpoly2sym`, `msym2poly`) — skip
- `minRankChoiceBruteForce` — research-specific, skip
- `External/` (DecayFitNet, Violinplot) — standalone Python equivalents exist
- `OPCAR/` — evaluation scripts, out of scope

---

## Architecture decisions

| Decision | Rationale |
|----------|-----------|
| `dspGraph/` → FLAMO | FLAMO provides a more flexible, differentiable graph; no translation needed |
| `constructDelayMatrix` / `constructDelayFeedbackMatrix` → FLAMO | Use `delay_module()` and FLAMO graph concatenation |
| `dss2impzTransposed` → FLAMO | FLAMO's graph handles arbitrary topologies |
| `allpassInFDN` / `SchroederReverberator` in gallery → `NotImplementedError` | Use `generate/allpass_FDN/` submodule |
| `timeVaryingMatrix` → processing script | Not a general class; FDN-specific processing script is cleaner |
| `complete_orthogonal` uses SVD of A | Separate eigendecompositions of `I − A Aᵀ` / `I − AᵀA` are inconsistent when eigenvalues are degenerate |
| SOS convention | `a0 = 1` throughout (scipy convention); `absorption_geq` normalises on output |
| Examples → marimo | Interactive notebooks; plotly for visualisation |
