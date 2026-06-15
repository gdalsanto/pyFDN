# gallery_category: FDN Design & Analysis

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
    # Colorless FDN, trained in-notebook (DAFx23)

    The companion to **Colorless FDN**, which *loads* pre-optimized parameters
    from `.mat` files. Here we run the optimization ourselves with `pyFDN`'s
    four-step training API, following *"Differentiable Feedback Delay Network for
    Colorless Reverberation", Dal Santo, Prawda, Schlecht, Välimäki, DAFx23*
    (and its "tiny colorless FDN" follow-up):

    1. `pyFDN.build_fdn` -- a small lossless skeleton (random orthogonal feedback
       matrix, coprime delays).
    2. `pyFDN.make_objective("colorless")` -- flat magnitude via magnitude MSE
       plus a sparsity penalty on the feedback matrix.
    3. `pyFDN.train_fdn` -- optimize the feedback matrix and gains in place.
    4. `pyFDN.extract_build` -- read both the initial and optimized FDNs back out.

    Delays stay fixed. We then add homogeneous decay so the result is audible.
    Flatness is measured on the magnitude response at the **training** FFT bins --
    the right colorless measure for a lossless FDN, whose time response never
    decays. Training is kept short so the example runs in a few seconds.
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
    fs = 48000.0
    nfft = 2**11

    def magnitude_response(model):
        x = torch.zeros(1, nfft, 1)
        x[:, 0, :] = 1.0
        with torch.no_grad():
            return np.asarray(model(x).detach())[0].sum(axis=-1)

    # 1. build a small "tiny colorless" lossless skeleton: 8 coprime delays.
    delays = pyFDN.sample_delay_lengths(
        8, (800, 3200), distribution="geometric", coprime=True, rng=0
    )
    model = pyFDN.build_fdn(
        delays=delays, rt=None, nfft=nfft, output="magnitude", device="cpu", rng=0
    )
    init_build = pyFDN.extract_build(model, fs=fs)  # random init, before training
    mag_init = magnitude_response(model)

    # 2. colorless objective; 3. train in place; 4. extract the optimized FDN.
    log = pyFDN.train_fdn(
        model,
        pyFDN.make_objective("colorless"),
        max_epochs=60,
        expand=64,
        batch_size=8,
        lr=3e-3,
        device="cpu",
        rng=0,
    )
    mag_opt = magnitude_response(model)
    opt_build = pyFDN.extract_build(model, fs=fs)

    print(
        f"spectral flatness  init {pyFDN.flatness_from_magnitude(mag_init):.4f}"
        f"   colorless {pyFDN.flatness_from_magnitude(mag_opt):.4f}"
    )
    return delays, fs, init_build, log, mag_init, mag_opt, nfft, opt_build


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnitude response and training loss
    """)
    return


@app.cell
def _(fs, log, mag_init, mag_opt, mo, nfft, np, pyFDN):
    import matplotlib.pyplot as plt

    _freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.2))
    _axes[0].plot(
        _freqs,
        pyFDN.lin_to_db(mag_init),
        alpha=0.5,
        label=f"init ({pyFDN.flatness_from_magnitude(mag_init):.3f})",
    )
    _axes[0].plot(
        _freqs,
        pyFDN.lin_to_db(mag_opt),
        label=f"colorless ({pyFDN.flatness_from_magnitude(mag_opt):.3f})",
    )
    _axes[0].set(
        xlabel="frequency [Hz]",
        ylabel="magnitude [dB]",
        xscale="log",
        title="Magnitude response (flatness in legend)",
    )
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(log.train_loss, "-o", ms=2)
    _axes[1].set(xlabel="epoch", ylabel="loss", yscale="log", title="Training loss")
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listen: random init vs colorless

    Homogeneous decay (target $T_{60}$) is folded into the feedback matrix via
    $\Gamma = \mathrm{diag}(g^{m})$ and rendered with `pyFDN.dss_to_impz`.
    """)
    return


@app.cell
def _(delays, fs, init_build, mo, np, opt_build, pyFDN):
    rt = 1.5
    g = pyFDN.db_to_lin(pyFDN.rt_to_slope(rt, fs))
    gamma = np.diag(g ** delays.astype(float))
    ir_len = int(1.2 * rt * fs)

    def render(build):
        return pyFDN.dss_to_impz(
            ir_len, delays, build.A @ gamma, build.B, build.C, build.D
        ).squeeze()

    ir_init = render(init_build)
    ir_opt = render(opt_build)

    _plot = pyFDN.plot_impulse_response(
        ir_opt, ir_init, fs=fs, labels=["Colorless", "Random init"]
    )
    _audio = mo.vstack(
        [
            mo.Html("Random init").style({"font-size": "1.5em"}),
            mo.audio(np.asarray(ir_init), rate=int(fs)),
            mo.Html("Colorless").style({"font-size": "1.5em"}),
            mo.audio(np.asarray(ir_opt), rate=int(fs)),
        ],
        gap=1,
    )
    mo.vstack([_plot, _audio], gap=3)
    return


if __name__ == "__main__":
    app.run()
