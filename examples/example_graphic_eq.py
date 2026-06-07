import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Graphic EQ design

    Demonstrates `pyFDN.design_geq`: a 10-band proportional parametric EQ that
    approximates a target magnitude response in dB.  The EQ has 11 biquad
    sections (flat gain + low-shelf + 8 peaking bands + high-shelf) and the
    per-band gains are found by constrained least squares.

    Reference: *Välimäki and Reiss, "All About Audio Equalization: Solutions and
    Frontiers," Applied Sciences, vol. 6, no. 5, p. 129, 2016.*

    Original MATLAB version: Sebastian J. Schlecht, 7. January 2019.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from scipy.signal import freqz, sosfilt

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, pio, pyFDN, freqz, sosfilt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design parameters

    The target is an alternating ±10 dB pattern across the 10 frequency bands,
    matching the MATLAB demo in `example_graphicEQ.m`.
    """)
    return


@app.cell
def _(np, pyFDN):
    fs = 48000.0
    fft_len = 2**16

    # Alternating ±10 dB target at 10 frequency bands
    target_g = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, 1], dtype=float) * 10  # dB

    sos, target_f = pyFDN.design_geq(target_g, fs=fs)
    print(f"SOS shape: {sos.shape}  — {sos.shape[0]} biquad sections")
    print(f"Target frequencies: {target_f}")
    return fs, fft_len, sos, target_f, target_g


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency response

    Compute the cascaded frequency response of all sections via `scipy.signal.freqz`
    and overlay the target.
    """)
    return


@app.cell
def _(freqz, fs, fft_len, go, np, sos, target_f, target_g):
    # Cascade all sections: multiply responses in frequency domain
    w_hz = np.linspace(0, fs / 2, fft_len // 2 + 1)
    H_total = np.ones(fft_len // 2 + 1, dtype=complex)
    for band in range(sos.shape[0]):
        b = sos[band, :3]
        a = sos[band, 3:]
        w, h = freqz(b, a, worN=fft_len // 2 + 1, fs=fs)
        H_total *= h

    mag_db = 20 * np.log10(np.maximum(np.abs(H_total), 1e-300))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target_f,
            y=target_g,
            mode="markers+lines",
            name="Target",
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_hz,
            y=mag_db,
            mode="lines",
            name="GEQ response",
            line={"width": 1.5},
        )
    )
    fig.update_layout(
        title="Graphic EQ — cascaded magnitude response vs. target",
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [np.log10(10), np.log10(fs / 2)],
        },
        yaxis={"title": "Magnitude (dB)", "range": [-15, 15]},
        template="plotly_white",
        height=420,
    )
    fig.show()
    return fig, mag_db, w_hz


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification

    Check that the designed GEQ approximates the target within ±2 dB at the
    control frequencies.
    """)
    return


@app.cell
def _(np, pyFDN, sos, target_f, target_g):
    from pyFDN.graphicEQ.probe_sos import probe_sos

    ctrl = target_f[1:-1].astype(float)  # 8 centre frequencies
    ctrl_target = target_g[1:-1]

    G, _, _ = probe_sos(sos, ctrl, 2**16, 48000.0)
    actual_db = G.sum(axis=1)
    error_db = actual_db - ctrl_target

    for f, t, a, e in zip(ctrl, ctrl_target, actual_db, error_db, strict=False):
        print(
            f"{f:6.0f} Hz  target={t:+6.1f} dB  actual={a:+6.1f} dB  error={e:+5.1f} dB"
        )

    assert np.all(np.abs(error_db) < 2.0), "GEQ error exceeds ±2 dB"
    print("\nAll bands within ±2 dB ✓")
    return


if __name__ == "__main__":
    app.run()
