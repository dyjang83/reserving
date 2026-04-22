# Quickstart

## Loading data

From a long-format DataFrame (e.g. the CAS Loss Reserve Database):

```python
import pandas as pd
from reserving import Triangle

df = pd.read_csv("wkcomp_pos.csv")
tri = Triangle.from_dataframe(
    df,
    origin="AccidentYear",
    dev="DevelopmentLag",
    values="CumPaidLoss"
)
print(tri)
```

From a pivot-format DataFrame:

```python
import pandas as pd
from reserving import Triangle

data = pd.DataFrame(
    {1: [1000, 1200, 900],
     2: [1100, 1300, None],
     3: [1150, None, None]},
    index=[2021, 2022, 2023]
)
tri = Triangle(data)
```

## Chain-ladder

```python
from reserving import ChainLadder

cl = ChainLadder(tri).fit()
print(cl.factors())       # ATA development factors
print(cl.ultimates())     # projected ultimate losses
print(cl.ibnr())          # IBNR by accident year
print(cl.total_ibnr())    # total reserve
print(cl.summary())       # ultimates + 95% bootstrap CIs
```

## Bornhuetter-Ferguson

```python
from reserving import BornhuetterFerguson

bf = BornhuetterFerguson(tri, apriori=0.65, premium=10000).fit()
print(bf.ultimates())
print(bf.summary())
```

## Cape Cod

```python
from reserving import CapeCod

cc = CapeCod(tri, premium=10000).fit()
print(cc.elr())           # derived expected loss ratio
print(cc.ultimates())
print(cc.summary())
```

## Comparing all three methods

```python
import reserving.plot as rvplot

fig = rvplot.comparison_chart({
    "Chain-ladder": cl,
    "BF": bf,
    "Cape Cod": cc,
})
fig.savefig("comparison.png")
```

## Bootstrap confidence intervals

All three methods support `summary(alpha, n_boot)`:

```python
# 90% CI with 2000 bootstrap resamples
summary = cl.summary(alpha=0.10, n_boot=2000)
print(summary)
```