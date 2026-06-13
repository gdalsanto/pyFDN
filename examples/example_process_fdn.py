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
    # process_fdn — Pure DSS Simulation

    Demonstrates `pyFDN.process_fdn` for time-domain simulation of a feedback delay network with static matrices.
    A dry audio signal is run through the FDN to produce reverberation.

    For FDNs with absorption filters or learnable parameters, use the FLAMO path (`dss_to_flamo`).
    """)
    return


@app.cell
def _():
    from importlib.resources import files

    import numpy as np
    import soundfile as sf

    import pyFDN

    return files, np, pyFDN, sf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load dry audio
    """)
    return


@app.cell
def _(files, mo, sf):
    path = files("pyFDN.audio") / "synth_dry.wav"
    with path.open("rb") as f:
        dry, fs = sf.read(f, dtype="float64")
    dry = dry[:, 0] if dry.ndim > 1 else dry

    print(f"Loaded {len(dry)} samples at {fs} Hz ({len(dry) / fs:.2f} s)")
    mo.vstack([mo.audio(dry, fs)])
    return dry, fs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN parameters
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    delays = pyFDN.ms_to_smp(np.array([20, 27, 31, 37, 43, 53, 61, 71]), fs)
    build = pyFDN.fdn_build_gallery(
        build_type="vanillaBroadband",
        fs=fs,
        delays=delays,
        io_type="normalized",
        direct_gain=0.0,
        rt60=2.0,
        rng=0,
    )
    A, B, C, D = build.A, build.B, build.C, build.D
    delays = build.delays
    g = pyFDN.rt_to_gain_per_sample(2.0, fs)

    print(f"Delays: {delays} samples, gain per sample: {g:.6f}")
    return A, B, C, D, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run dry signal through the FDN
    """)
    return


@app.cell
def _(A, B, C, D, delays, dry, fs, mo, pyFDN):
    wet = pyFDN.process_fdn(dry, delays, A, B, C, D)
    wet = pyFDN.peak_normalize(wet)

    print(f"Output shape: {wet.shape}")
    mo.vstack([mo.audio(wet, fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, pyFDN):
    ir = pyFDN.dss_to_impz(4 * fs, delays, A, B, C, D)[:, 0, 0]
    pyFDN.plot_impulse_response(ir, fs=fs)
    return (ir,)


@app.cell
def _(fs, ir, pyFDN):
    pyFDN.plot_edc(ir, fs=fs)
    return


if __name__ == "__main__":
    app.run()
