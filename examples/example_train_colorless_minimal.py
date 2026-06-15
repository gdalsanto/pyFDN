# gallery_category: Getting Started

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train a colorless FDN (minimal)

    The smallest end-to-end training example, in the four explicit steps of
    `pyFDN`'s training API:

    1. **build** a trainable FDN -- `pyFDN.build_fdn` (here a lossless skeleton:
       random orthogonal feedback matrix, fixed coprime delays).
    2. **define** an objective -- `pyFDN.make_objective("colorless")` asks for a
       flat magnitude response (the task from *Dal Santo et al., DAFx23*).
    3. **train** -- `pyFDN.train_fdn` optimizes the model in place.
    4. **extract** -- `pyFDN.extract_build` reads the result back into an
       `FDNBuild` for rendering / analysis.

    Settings are deliberately tiny so it runs in a few seconds; raise `nfft` /
    `max_epochs` for a flatter result.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    import pyFDN

    return np, pyFDN, torch


@app.cell
def _(np, pyFDN, torch):
    nfft = 2**11

    def magnitude_response(model):
        """The model's |H| at DFT bins, summed over output channels."""
        x = torch.zeros(1, nfft, 1)
        x[:, 0, :] = 1.0
        with torch.no_grad():
            return np.asarray(model(x).detach())[0].sum(axis=-1)

    # 1. build a lossless FDN (magnitude output, so we can read flatness off it).
    model = pyFDN.build_fdn(
        N=4, rt=None, nfft=nfft, output="magnitude", device="cpu", rng=0
    )
    mag_init = magnitude_response(model)

    # 2. define the colorless objective, then 3. train in place.
    objective = pyFDN.make_objective("colorless")
    log = pyFDN.train_fdn(
        model,
        objective,
        max_epochs=20,
        expand=64,
        batch_size=8,
        lr=3e-3,
        device="cpu",
        rng=0,
    )
    mag_opt = magnitude_response(model)

    # 4. extract the optimized FDN (composes with render / analyze / decompose).
    fdn = pyFDN.extract_build(model, fs=48000.0)
    print(
        f"spectral flatness  init {pyFDN.flatness_from_magnitude(mag_init):.4f}"
        f"   colorless {pyFDN.flatness_from_magnitude(mag_opt):.4f}"
    )
    return fdn, log, mag_init, mag_opt, nfft


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training loss

    The combined magnitude-MSE + matrix-sparsity loss falls as the response flattens.
    """)
    return


@app.cell
def _(log, mo):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(log.train_loss, "-o", ms=3)
    _ax.set(xlabel="epoch", ylabel="training loss", title="Colorless optimization")
    _ax.set_yscale("log")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.as_html(_fig)
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnitude response: before vs after

    The optimized FDN has a flatter magnitude response (fewer deep notches and
    sharp peaks), quantified by the spectral-flatness value in each label.
    """)
    return


@app.cell
def _(mag_init, mag_opt, mo, nfft, np, plt, pyFDN):
    _freqs = np.fft.rfftfreq(nfft, 1.0 / 48000.0)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(
        _freqs,
        pyFDN.lin_to_db(mag_init),
        alpha=0.5,
        label=f"init (flatness {pyFDN.flatness_from_magnitude(mag_init):.3f})",
    )
    _ax.plot(
        _freqs,
        pyFDN.lin_to_db(mag_opt),
        label=f"colorless (flatness {pyFDN.flatness_from_magnitude(mag_opt):.3f})",
    )
    _ax.set(xlabel="frequency [Hz]", ylabel="magnitude [dB]", xscale="log")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
