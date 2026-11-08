import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell
def _(torch):
    values = []
    for accumulator_dtype, input_dtype, explicit_cast in [
        (torch.float32, torch.float32, False), (torch.float16, torch.float16, False),
        (torch.float32, torch.float16, False), (torch.float32, torch.float16, True),
    ]:
        s = torch.tensor(0, dtype=accumulator_dtype)
        for _ in range(1000):
            x = torch.tensor(0.01, dtype=input_dtype)
            s += x.float() if explicit_cast else x
        values.append(float(s))
    values
    return (values,)


@app.cell
def _(mo, values):
    mo.md(f"""# Mixed-precision accumulation

    四段原始代码的实测结果依次为 **{values}**，精确实数答案是 10。
    FP16 累加每一步都舍入，随着累加器变大，0.01 相对 ULP 越来越小，因此误差明显；FP32 累加误差很小。
    后两段完全相同：输入在 FP16 中已舍入到 0.01000213623046875，转回 FP32 不会恢复丢失的精度，但 FP32 累加避免了进一步的大量舍入。
    """)
    return


if __name__ == "__main__":
    app.run()
