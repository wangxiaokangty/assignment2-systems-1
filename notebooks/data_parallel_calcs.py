import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from cs336_systems.parallelism import costs
    return mo, costs


@app.cell
def _(mo):
    mo.md(r"""# Data parallel calculations

All costs are **per device**, with a global batch $B$, FP16 tensors, and ideal bandwidth $W$ bytes/s. We ignore latency and non-matmul operations.

**(a)** Backward FLOPs: $12BDD_{FF}/N_{DP}$. There are six matrix products: one for $dz$, two for $dx$, and three weight gradients, each costing $2BDD_{FF}/N_{DP}$.

**(b)** Backward communication: $12(N_{DP}-1)DD_{FF}/(N_{DP}W)$. The three weight gradients total $6DD_{FF}$ bytes, and ring all-reduce sends $2(N_{DP}-1)/N_{DP}$ times that size.

**(c)** Compute-bound when
$$N_{DP} \le 1 + BW/C.$$
Comparing communication with $12BDD_{FF}/(N_{DP}C)$ and cancelling common factors gives $N_{DP}-1\le BW/C$. Equality is the crossover; larger device counts are communication-bound.
""")
    return


@app.cell
def _(mo, costs):
    mo.ui.table([{"devices": n, **costs(fsdp=n, strategy="dp")} for n in (1, 2, 4, 8, 16, 32)])
    return


if __name__ == "__main__":
    app.run()
