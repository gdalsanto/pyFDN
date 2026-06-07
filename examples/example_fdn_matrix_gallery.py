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
    # FDN Matrix Gallery

    Overview of the feedback matrix types available in `pyFDN.fdn_matrix_gallery`.
    Each type has different spectral properties: some are orthogonal (lossless),
    some are only spectrally lossless (eigenvalues on the unit circle).

    Reference: *Schlecht, "FDNTB: The Feedback Delay Network Toolbox," DAFx-20, 2020.*

    Original MATLAB: Sebastian J. Schlecht, 28 December 2019.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, make_subplots, np, pio, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve all matrix types

    Query `fdn_matrix_gallery()` with no arguments to get the list of available types.
    """)
    return


@app.cell
def _(np, pyFDN):
    N = 8
    matrix_types = pyFDN.fdn_matrix_gallery()
    print(f"Available types ({len(matrix_types)}):")
    for t in matrix_types:
        print(f"  {t}")
    return N, matrix_types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build matrices and check losslessness

    A matrix is orthogonal if `A @ A.T ≈ I`.  A matrix is spectrally lossless if
    all eigenvalues lie on the unit circle.  The table below shows both checks.
    """)
    return


@app.cell
def _(N, matrix_types, np, pyFDN):
    results = {}
    for mtype in matrix_types:
        try:
            result = pyFDN.fdn_matrix_gallery(N, mtype)
            A = result[0] if isinstance(result, tuple) else result
            is_orth = bool(np.allclose(A @ A.T, np.eye(N), atol=1e-6))
            eig_mod = np.abs(np.linalg.eigvals(A))
            is_spectral = bool(np.allclose(eig_mod, np.ones(N), atol=1e-6))
            results[mtype] = {
                "matrix": A,
                "orthogonal": is_orth,
                "spectral": is_spectral,
                "notes": "",
            }
        except NotImplementedError:
            results[mtype] = {"matrix": None, "orthogonal": False, "spectral": False}

    print(f"{'Type':<30} {'Orthogonal':>12} {'Spectral lossless':>18}")
    print("-" * 62)
    for mtype, r in results.items():
        status = "n/a (NotImplemented)" if r["matrix"] is None else ""
        print(
            f"{mtype:<30} {str(r['orthogonal']):>12} {str(r['spectral']):>18} {status}"
        )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix heatmaps

    Visualise each matrix as a colour-coded grid.
    """)
    return


@app.cell
def _(go, make_subplots, matrix_types, np, results):
    # Filter to implemented types only
    implemented = [(t, r) for t, r in results.items() if r["matrix"] is not None]
    n_plots = len(implemented)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    subplot_titles = [
        f"{t}<br><sup>orth={r['orthogonal']}, spec={r['spectral']}</sup>"
        for t, r in implemented
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    for idx, (_mtype, r) in enumerate(implemented):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.add_trace(
            go.Heatmap(
                z=r["matrix"],
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=False,
            ),
            row=row,
            col=col,
        )
        N = r["matrix"].shape[0]
        tick_vals = list(range(N))
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=[str(i) for i in tick_vals], row=row, col=col
        )
        fig.update_yaxes(
            tickvals=tick_vals,
            ticktext=[str(i) for i in tick_vals],
            autorange="reversed",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"FDN matrix gallery (N={r['matrix'].shape[0]})",
        height=200 * n_rows + 80,
        template="plotly_white",
    )
    fig.show()
    return (fig,)


if __name__ == "__main__":
    app.run()
