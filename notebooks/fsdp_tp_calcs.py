import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from cs336_systems.parallelism import optimal_grid
    return mo, optimal_grid


@app.cell
def _(mo):
    mo.md(r"""# 2D FSDP + tensor parallelism

Write $p=N_F$, $q=N_T$.

**(a)** Forward FLOPs per device: $6BDD_{FF}/(pq)$, since the three matrix products shard both batch and hidden dimensions.

**(b)** Forward communication time:
$$\max\left(\frac{6(p-1)DD_{FF}}{pqW},\frac{4(q-1)BD}{pqW}\right).$$
The first term gathers three TP-sharded FP16 weights along FSDP; the second all-reduces the batch-sharded FP16 output along TP. The maximum follows from the assumed independent, overlapping networks.

**(c)** Define $a=C/(BW)$ and $b=2C/(3D_{FF}W)$. Dividing communication by compute time gives $\max(a(p-1),b(q-1))\le1$. Thus
$$N=pq\le (1+BW/C)(1+3D_{FF}W/(2C)).$$
The continuous optimum sets $p=1+1/a$ and $q=1+1/b$. Actual integer grids must round down and satisfy weight/batch divisibility; these hardware constraints can lower the attainable bound.

**(d)** With shared network resources the constraint is
$$a(p-1)+b(q-1)\le1,\quad p,q\ge1.$$
Set $K=1+a+b$. At an interior optimum $ap=bq=K/2$, from maximizing $p(K-ap)/b$. Therefore
$$N\le\begin{cases}
1+1/b,&a>1+b,\\
1+1/a,&b>1+a,\\
K^2/(4ab),&\text{otherwise}.
\end{cases}$$
The boundary cases choose $p=1$ or $q=1$ because a device-axis size cannot be below one. In the interior, equivalently
$$N\le\frac{3BD_{FF}W^2}{8C^2}\left(1+\frac{C}{BW}+\frac{2C}{3D_{FF}W}\right)^2.$$

The numerical tables below are evaluations of this idealized analytical model, **not GPU measurements**. Defaults: $B=4096,D=4096,D_{FF}=11008,C=312\times10^{12}$ FLOP/s and $W=300\times10^9$ bytes/s.
""")
    return


@app.cell
def _(mo, optimal_grid):
    mo.ui.table([{"axis_overlap": overlap, **optimal_grid(overlap=overlap)} for overlap in (True, False)])
    return


if __name__ == "__main__":
    app.run()
