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
    mo.md(r"""# Tensor parallel calculations

**(a)** Given replicated $dy$, the backward pass at TP rank $i$ is
$$dz^{(i)}=dy(W_3^{(i)})^T,$$
$$dx_2^{(i)}=dz^{(i)}\odot f(x_1^{(i)}),\qquad dx_1^{(i)}=dz^{(i)}\odot f'(x_1^{(i)})\odot x_2^{(i)},$$
$$dW_3^{(i)}=(z^{(i)})^Tdy,\quad dW_2^{(i)}=x^Tdx_2^{(i)},\quad dW_1^{(i)}=x^Tdx_1^{(i)},$$
$$dx=\operatorname{allreduce}_i\left(dx_1^{(i)}(W_1^{(i)})^T+dx_2^{(i)}(W_2^{(i)})^T\right).$$
The two partial input gradients can be added locally before a single all-reduce. No weight-gradient collective is needed: each rank owns a distinct weight shard. The replicated $dy$ describes one logical loss, so it is not summed across identical replicas.

**(b)** Forward: $6BDD_{FF}/N_T$ FLOPs (three products). Backward: $12BDD_{FF}/N_T$ FLOPs (six products). All products have one dimension sharded by $N_T$.

**(c)** Each pass communicates $4(N_T-1)BD/(N_TW)$ seconds: one all-reduce of an FP16 $(B,D)$ activation or input gradient.

**(d)** Backward is compute-bound when $N_T\le 1+3D_{FF}W/C$. Forward is compute-bound when $N_T\le 1+3D_{FF}W/(2C)$. Substitute the FLOP counts divided by $C$ and compare to the communication time; backward tolerates twice the extra ranks because it does twice the compute.
""")
    return


@app.cell
def _(mo, costs):
    mo.ui.table([{"devices": n, **costs(tp=n)} for n in (1, 2, 4, 8, 16, 32)])
    return


if __name__ == "__main__":
    app.run()
