# actuarial-reserving

Open-source Python library for P&C loss reserve estimation — chain-ladder, Bornhuetter-Ferguson, and Cape Cod with bootstrap confidence intervals.

```bash
pip install actuarial-reserving
```

---

## Methods

| Method | Description |
|---|---|
| `ChainLadder` | Projects losses to ultimate using volume-weighted age-to-age factors |
| `BornhuetterFerguson` | Blends chain-ladder development with an a priori expected loss ratio |
| `CapeCod` | Derives the expected loss ratio from the data — no external assumption required |

All three methods share a consistent API: `fit()`, `ultimates()`, `ibnr()`, `summary()`.

---

## Quickstart

```python
import pandas as pd
import reserving as rv
import reserving.plot as rvplot

# Build a triangle from a pivot DataFrame
data = pd.DataFrame(
    {1: [1000, 1200, 900],
     2: [1100, 1300, None],
     3: [1150, None, None]},
    index=[2021, 2022, 2023]
)
tri = rv.Triangle(data)

# Or load from long-format CSV (e.g. CAS Loss Reserve Database)
tri = rv.Triangle.from_csv(
    "wkcomp_pos.csv",
    origin="AccidentYear",
    dev="DevelopmentLag",
    values="CumPaidLoss"
)
```

### Chain-ladder

```python
cl = rv.ChainLadder(tri).fit()

cl.factors()       # volume-weighted ATA development factors
cl.ultimates()     # projected ultimate losses by accident year
cl.ibnr()          # IBNR reserve by accident year
cl.total_ibnr()    # total reserve
cl.summary()       # ultimates + 95% bootstrap confidence intervals
```

### Bornhuetter-Ferguson

```python
bf = rv.BornhuetterFerguson(tri, apriori=0.65, premium=10000).fit()

bf.pct_reported()  # % of ultimate losses already reported
bf.ultimates()
bf.summary()
```

### Cape Cod

```python
cc = rv.CapeCod(tri, premium=10000).fit()

cc.elr()           # expected loss ratio derived from the data
cc.ultimates()
cc.summary()
```

### Bootstrap confidence intervals

All methods support `summary(alpha, n_boot)`:

```python
# 95% CI with 999 bootstrap resamples (default)
print(cl.summary())

# 90% CI with 2000 resamples
print(cl.summary(alpha=0.10, n_boot=2000))
```

### Visualization

```python
rvplot.development_chart(tri)        # loss development curves by accident year
rvplot.ultimates_chart(cl)           # ultimate vs latest diagonal
rvplot.ibnr_chart(cc)                # IBNR reserves with totals
rvplot.summary_chart(bf)             # ultimates with CI bands
rvplot.comparison_chart({            # all three methods side by side
    "Chain-ladder": cl,
    "BF": bf,
    "Cape Cod": cc,
})
```

---

## Installation

```bash
pip install actuarial-reserving
```

Requires Python ≥ 3.9. Dependencies: `pandas`, `numpy`, `matplotlib`.

---

## Documentation

Full API reference and quickstart guide: [dyjang83.github.io/reserving](https://dyjang83.github.io/reserving)

---

## Project structure

```
reserving/
├── reserving/
│   ├── __init__.py
│   ├── triangle.py              # Triangle data structure
│   ├── bootstrap.py             # CI engine
│   ├── plot.py                  # visualization module
│   └── methods/
│       ├── chain_ladder.py
│       ├── bornhuetter_ferguson.py
│       └── cape_cod.py
├── tests/                       # 103 passing tests
├── docs/                        # MkDocs documentation source
├── pyproject.toml
└── README.md
```

---

## Development

```bash
git clone https://github.com/dyjang83/reserving.git
cd reserving
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
