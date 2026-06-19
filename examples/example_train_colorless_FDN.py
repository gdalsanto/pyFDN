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
    # Colorless FDN, trained in-notebook

    The companion to **Colorless FDN**, which *loads* pre-optimized parameters
    from `.mat` files. Here we run the optimization ourselves with `pyFDN`'s
    training API, following *"Differentiable Feedback Delay Network for
    Colorless Reverberation", Dal Santo, Prawda, Schlecht, Välimäki, DAFx23*
    (and its "tiny colorless FDN" follow-up):

    1. `pyFDN.build_fdn` -- a standard FDN skeleton with random orthogonal feedback
       matrix.
    2. `pyFDN.train_fdn(model, "colorless")` -- optimize the feedback matrix and
       gains for a flat magnitude (magnitude MSE + a feedback-matrix sparsity
       penalty), in place.
    3. `pyFDN.extract_build` -- read both the initial and optimized FDNs back out.

    Delays stay fixed. We then add homogeneous decay so the result is audible.
    Flatness is measured on the magnitude response at the **training** FFT bins --
    the right colorless measure for a lossless FDN, whose time response never
    decays. Training is kept short so the example runs in a few seconds.
          
    - pyFDN training pipeline: Jeremy B. Bai, 2026-06-19
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell
def _(np, pyFDN):
    fs = 48000
    nfft = 2**11

    # 1. build a small "tiny colorless" lossless skeleton: 8 coprime delays.
    delays = pyFDN.sample_delay_lengths(
        8, (800, 3200), distribution="uniform", coprime=False, rng=0
    )
    model = pyFDN.build_fdn(
        delays=delays, rt=None, nfft=nfft, output="magnitude", device="cpu", rng=0
    )
    init_build = pyFDN.extract_build(model, fs=fs)  # random init, before training
    mag_init = np.abs(pyFDN.flamo_freq_response(model, fs=fs).squeeze())

    # 2. train in place for a flat ("colorless") magnitude; then extract.
    log = pyFDN.train_fdn(
        model,
        "colorless",
        max_epochs=200,
        expand=10,
        batch_size=1,
        lr=1e-3,
        device="cpu",
        rng=1,
    )
    mag_opt = np.abs(pyFDN.flamo_freq_response(model, fs=fs).squeeze())
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

    Two renderings of each FDN (`pyFDN.dss_to_impz`, peak-normalized so the A/B
    compares *timbre*, not level):

    * **Lossless (no decay)** -- the trained FDN as-is. Its poles sit on the unit
      circle, so it rings without decaying and you hear the colour directly: the
      random init is tonal/metallic (sharp modal resonances), the colorless one
      is noise-like (flat spectrum).
    * **With decay** -- homogeneous $T_{60}$ folded into the feedback matrix via
      $\Gamma = \mathrm{diag}(g^{m})$, giving an audible reverb tail.
    """)
    return


@app.cell
def _(delays, fs, init_build, mo, np, opt_build, pyFDN):
    rt = 2.0
    g = pyFDN.db_to_lin(pyFDN.rt_to_slope(rt, fs))
    gamma = np.diag(g ** delays.astype(float))

    def render(build, feedback, n_samples):
        ir = pyFDN.dss_to_impz(
            n_samples, delays, feedback, build.B, build.C, build.D
        ).squeeze()
        # peak-normalize so the A/B compares timbre at matched level, not loudness.
        return np.asarray(ir) / (np.max(np.abs(ir)) + 1e-12)

    # Lossless (non-decaying): the trained FDN as-is rings without decay, so you
    # hear the colour directly. A short fade-out avoids the click from truncating
    # a non-decaying signal.
    noise_len = int(2.0 * fs)
    _fade = np.ones(noise_len)
    _fade[-2048:] = np.linspace(1.0, 0.0, 2048)
    init_noise = render(init_build, init_build.A, noise_len) * _fade
    opt_noise = render(opt_build, opt_build.A, noise_len) * _fade

    # Decaying reverb: homogeneous T60 folded into the feedback matrix (A @ Gamma).
    ir_len = int(2.0 * fs)
    init_decay = render(init_build, init_build.A @ gamma, ir_len)
    opt_decay = render(opt_build, opt_build.A @ gamma, ir_len)

    def _clip(label, sig):
        return mo.vstack(
            [mo.Html(label).style({"font-size": "1.1em"}), mo.audio(sig, rate=int(fs))],
            gap=0,
        )

    _plot = pyFDN.plot_impulse_response(
        opt_decay, init_decay, fs=fs, labels=["Colorless", "Random init"]
    )
    _audio = mo.hstack(
        [
            mo.vstack(
                [
                    mo.Html("<b>Lossless (no decay)</b>").style({"font-size": "1.2em"}),
                    _clip("Random init", init_noise),
                    _clip("Colorless", opt_noise),
                ],
                gap=1,
            ),
            mo.vstack(
                [
                    mo.Html("<b>With homogeneous decay</b>").style(
                        {"font-size": "1.2em"}
                    ),
                    _clip("Random init", init_decay),
                    _clip("Colorless", opt_decay),
                ],
                gap=1,
            ),
        ],
        gap=2,
    )

    mo.vstack([_plot, _audio], gap=3)
    return


if __name__ == "__main__":
    app.run()
