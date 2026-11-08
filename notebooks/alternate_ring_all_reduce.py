import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from cs336_systems.parallelism import ring_seconds
    return mo, ring_seconds


@app.cell
def _(mo):
    mo.md(r"""# Alternate ring all-reduce

$$T=(N-1)S/W.$$

Each of the $N-1$ rounds transmits a **full** $S$-byte input per device at bandwidth $W$; all ranks transmit simultaneously. This is $N/2$ times the reduce-scatter + all-gather cost $2(N-1)S/(NW)$ for $N>1$ (both cost zero at $N=1$).
""")
    return


@app.cell
def _(mo, ring_seconds):
    mo.ui.table([{ "devices": n, "alternate_ms": 1000 * ring_seconds(1e9, n, 300e9, alternate=True), "standard_ms": 1000 * ring_seconds(1e9, n, 300e9)} for n in (1, 2, 4, 8, 16)])
    return


if __name__ == "__main__":
    app.run()
