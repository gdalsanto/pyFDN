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
    # Nearest Sign-Agnostic Orthogonal Matrix

    Given a non-negative matrix **B** (e.g., measured energy flow between delay
    lines), find the orthogonal matrix **U** that minimises ``‖B − |U|‖_F`` where
    ``|·|`` is element-wise absolute value.

    The challenge is assigning the right ±1 sign to each element.  A naive
    approach (just solving the ordinary Procrustes problem with `nearest_orthogonal`)
    ignores the freedom in signs.  The sign-agnostic algorithm does:

    1. Normalise **B** to doubly stochastic via Sinkhorn-Knopp.
    2. Initialise with a random sign pattern.
    3. Alternate: (a) solve nearest-orthogonal via SVD, (b) update signs.
    4. Repeat from 2 with new random initialisations; keep best.

    Reference: *Schlecht and Habets, "Sign-Agnostic Matrix Design for Spatial
    Artificial Reverberation with Feedback Delay Networks," AES Conf. on Spatial
    Reproduction, 2018.*

    Original MATLAB: Sebastian J. Schlecht, 29. January 2020.
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
    ## Setup

    Generate a random orthogonal matrix **A** and strip its signs to get **B = |A|**.
    The goal is to recover a matrix close to **A** from **B** alone.
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(42)
    N = 8

    # Ground-truth orthogonal matrix
    A_original = pyFDN.random_orthogonal(N)
    # Target: absolute values only (signs removed)
    B = np.abs(A_original)

    print("Original orthogonal matrix A:")
    print(np.round(A_original, 3))
    print("\nInput |A| (signs stripped):")
    print(np.round(B, 3))
    return A_original, B, N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve

    Compare the naive **nearest_orthogonal** (ignores sign freedom) against the
    **sign-agnostic** solution.
    """)
    return


@app.cell
def _(A_original, B, N, np, pyFDN):
    classic = pyFDN.nearest_orthogonal(B)
    sign_agnostic = pyFDN.nearest_sign_agnostic_orthogonal(B, max_trials=10_000)

    error_classic = np.linalg.norm(np.abs(classic) - B, "fro") / N**2
    error_sign_agnostic = np.linalg.norm(np.abs(sign_agnostic) - B, "fro") / N**2

    print(f"Classic Procrustes error:      {error_classic:.6f}")
    print(f"Sign-agnostic Procrustes error: {error_sign_agnostic:.6f}")
    print(f"Improvement factor: {error_classic / error_sign_agnostic:.1f}×")

    assert pyFDN.is_orthogonal(classic, tol=1e-8), "classic solution not orthogonal"
    assert pyFDN.is_orthogonal(sign_agnostic, tol=1e-8), "sign-agnostic not orthogonal"
    assert error_sign_agnostic <= error_classic + 1e-10, (
        "sign-agnostic should not be worse"
    )
    return classic, error_classic, error_sign_agnostic, sign_agnostic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualise

    Plot the three matrices side by side: original **A**, input **|A|**, and the two solutions.
    """)
    return


@app.cell
def _(A_original, B, classic, go, make_subplots, np, sign_agnostic):
    matrices = [A_original, B, classic, sign_agnostic]
    titles = ["Original A", "|A| (input)", "Nearest orthogonal", "Sign-agnostic"]
    N = A_original.shape[0]

    fig = make_subplots(rows=1, cols=4, subplot_titles=titles)
    for col, (mat, _title) in enumerate(zip(matrices, titles, strict=False), start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=(col == 4),
            ),
            row=1,
            col=col,
        )

    tick_vals = list(range(N))
    for col in range(1, 5):
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=[str(i) for i in tick_vals], row=1, col=col
        )
        fig.update_yaxes(
            tickvals=tick_vals,
            ticktext=[str(i) for i in tick_vals],
            autorange="reversed",
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Sign-agnostic vs. classic nearest orthogonal",
        height=360,
        template="plotly_white",
    )
    fig.show()
    return (fig,)


if __name__ == "__main__":
    app.run()
