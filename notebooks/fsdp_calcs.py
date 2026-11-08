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
    mo.md(r"""# Fully sharded data parallel calculations

**(a)** Backward: $12BDD_{FF}/N_F$ FLOPs, from six batch-sharded matrix products. Forward: $6BDD_{FF}/N_F$ FLOPs, from three batch-sharded matrix products.

**(b)** Backward: $12(N_F-1)DD_{FF}/(N_FW)$ seconds, because all-gathering the three FP16 weights and reduce-scattering their gradients each costs $6(N_F-1)DD_{FF}/(N_FW)$. Forward: $6(N_F-1)DD_{FF}/(N_FW)$ seconds for one all-gather per weight. This assumes weights are discarded after forward, as in the handout.

**(c)** Both forward and backward are compute-bound exactly when
$$N_F\le 1+BW/C.$$
Backward has twice both the forward FLOPs and communication, so both comparisons reduce to $(N_F-1)C\le BW$. These are ideal overlap bounds, ignoring latency, launch cost and optimizer work.
""")
    return


@app.cell
def _(mo, costs):
    mo.ui.table([{"devices": n, **costs(fsdp=n)} for n in (1, 2, 4, 8, 16, 32)])
    return


if __name__ == "__main__":
    app.run()
