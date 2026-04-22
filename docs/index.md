# actuarial-reserving

Open-source Python library for P&C loss reserve estimation.

## Install

```bash
pip install actuarial-reserving
```

## Methods

| Method | Description |
|---|---|
| `ChainLadder` | Volume-weighted ATA factors projected to ultimate |
| `BornhuetterFerguson` | Blends chain-ladder with an a priori expected loss ratio |
| `CapeCod` | Derives the expected loss ratio from the data itself |

All three methods share a consistent API: `fit()`, `ultimates()`, `ibnr()`, `summary()`, and `plot`.

## Quick example

```python
import pandas as pd
import reserving as rv
import reserving.plot as rvplot

data = pd.DataFrame(
    {1: [1000, 1200, 900],
     2: [1100, 1300, None],
     3: [1150, None, None]},
    index=[2021, 2022, 2023]
)
tri = rv.Triangle(data)

cl = rv.ChainLadder(tri).fit()
print(cl.summary())

fig = rvplot.summary_chart(cl)
fig.savefig("reserves.png")
```