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
    # Frequency-dependent pole boundaries

    FDN with frequency-dependent absorption filters, but *not* with homogeneous
    (delay-proportional) decay. Still, boundaries for the pole magnitudes can be
    computed from the singular values of the loop transfer function and tested
    against the actual poles.

    The loop here is $P(z) = \mathrm{diag}(z^{m}) - A\,\mathrm{diag}(h(z))$ with
    a two-tap FIR absorption filter $h(z) = 0.65 + 0.3 z^{-1}$ on every delay
    line and a non-orthogonal feedback matrix $A = Q/1.5$.

    Reference: *Schlecht, S., Habets, E. (2019). Modal Decomposition of Feedback
    Delay Networks. IEEE Transactions on Signal Processing 67(20), 5340-5351.*
    [doi:10.1109/tsp.2019.2937286](https://dx.doi.org/10.1109/tsp.2019.2937286)

    Original MATLAB: `example_poleBoundaries.m`, Sebastian J. Schlecht,
    23 April 2018. Delays are scaled down relative to MATLAB so the
    root-finding stays fast.
    """)
    return


@app.cell
def _():
    from types import SimpleNamespace

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return SimpleNamespace, go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN with FIR absorption
    """)
    return


@app.cell
def _(SimpleNamespace, np, pyFDN):
    np.random.seed(6)
    fs = 48000
    num_delays = 8
    delays = np.random.randint(50, 301, num_delays)
    input_gain = np.eye(num_delays, 1)
    output_gain = np.eye(1, num_delays)
    direct = np.random.randn(1, 1)

    feedback_matrix = pyFDN.random_orthogonal(num_delays) / 1.5

    # two-tap FIR absorption per delay line: h(z) = 0.65 + 0.3 z^{-1}
    absorption = SimpleNamespace(
        b=np.zeros((num_delays, 1, 2)), a=np.zeros((num_delays, 1, 2))
    )
    absorption.a[:, 0, 0] = 1.0
    absorption.b[:, 0, 0] = 0.65
    absorption.b[:, 0, 1] = 0.3

    print(f"Delays: {delays} (sum = {delays.sum()})")
    return (
        absorption,
        delays,
        direct,
        feedback_matrix,
        fs,
        input_gain,
        num_delays,
        output_gain,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pole boundaries and modal decomposition

    `pole_boundaries` combines the singular values of the feedback matrix with
    the absorption magnitude responses and group delays. For the poles, the
    FIR absorption is folded into the loop as a polynomial feedback matrix
    $A(z) = A\,\mathrm{diag}(h(z))$ and `dss_to_pr_direct` is used in
    ``roots`` mode.
    """)
    return


@app.cell
def _(
    absorption,
    delays,
    direct,
    feedback_matrix,
    fs,
    input_gain,
    np,
    output_gain,
    pyFDN,
):
    min_curve, max_curve, f_bounds = pyFDN.pole_boundaries(
        delays, absorption, feedback_matrix[:, :, None], fs
    )

    # fold FIR absorption into a polynomial feedback matrix
    A_poly = np.stack(
        [feedback_matrix @ np.diag(absorption.b[:, 0, k]) for k in range(2)],
        axis=2,
    )
    residues, poles_all, _, _, _ = pyFDN.dss_to_pr_direct(
        delays, A_poly, input_gain, output_gain, direct, mode="roots"
    )

    # The folded loop has N extra near-defective poles clustered at the zero of
    # the absorption FIR (z = -b1/b0). They carry (numerically) zero residue, so
    # they do not contribute to the response and are not covered by the
    # boundary theory; exclude the cluster.
    fir_zero = -absorption.b[0, 0, 1] / absorption.b[0, 0, 0]
    fdn_poles = np.abs(poles_all - fir_zero) > 0.1
    poles = poles_all[fdn_poles]
    print(f"Number of FDN poles: {poles.size} (of {poles_all.size} roots)")
    print(
        "Max |residue| in excluded cluster: "
        f"{np.max(np.abs(residues[~fdn_poles, 0, 0]), initial=0):.2e}"
    )
    return A_poly, f_bounds, max_curve, min_curve, poles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poles between the boundaries

    Pole magnitudes converted to T60 over frequency. All poles lie between the
    minimum and maximum boundary curves.
    """)
    return


@app.cell
def _(f_bounds, fs, go, max_curve, min_curve, np, poles, pyFDN):
    pole_freq = pyFDN.rad_to_hertz(np.angle(poles), fs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pole_freq,
            y=pyFDN.slope_to_rt(pyFDN.lin_to_db(np.abs(poles)), fs),
            mode="markers",
            marker={"size": 4},
            name="Poles",
        )
    )
    for _curve, _name in [
        (min_curve, "Minimum boundary"),
        (max_curve, "Maximum boundary"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=f_bounds,
                y=pyFDN.slope_to_rt(pyFDN.lin_to_db(_curve), fs),
                mode="lines",
                line={"width": 3},
                name=_name,
            )
        )
    fig.update_layout(
        title="Pole T60 and frequency-dependent boundaries",
        xaxis={"title": "Frequency (Hz)"},
        yaxis={"title": "Pole RT60 (s)"},
        template="plotly_white",
        height=460,
    )
    fig.show()
    return (pole_freq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify bounds
    """)
    return


@app.cell
def _(f_bounds, max_curve, min_curve, np, pole_freq, poles, pyFDN):
    upper_ok, _ = pyFDN.is_bounding_curve(
        pole_freq, np.abs(poles), f_bounds, max_curve, "upper"
    )
    lower_ok, _ = pyFDN.is_bounding_curve(
        pole_freq, np.abs(poles), f_bounds, min_curve, "lower"
    )
    print(f"All poles below the maximum boundary: {bool(upper_ok)}")
    print(f"All poles above the minimum boundary: {bool(lower_ok)}")
    assert upper_ok and lower_ok
    return


if __name__ == "__main__":
    app.run()
