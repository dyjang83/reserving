import pandas as pd
import numpy as np
from reserving.triangle import Triangle


class ChainLadder:
    """
    Chain-ladder reserve estimator with bootstrap confidence intervals.

    The chain-ladder method projects each accident year's losses to ultimate
    by applying volume-weighted age-to-age development factors. It is the
    most widely used deterministic reserving method in P&C insurance.

    Parameters
    ----------
    triangle : Triangle
        A Triangle object containing cumulative paid or incurred losses.

    Examples
    --------
    >>> import pandas as pd
    >>> from reserving import Triangle, ChainLadder
    >>> data = pd.DataFrame(
    ...     {1: [1000, 1200, 900], 2: [1100, 1300, None], 3: [1150, None, None]},
    ...     index=[2021, 2022, 2023]
    ... )
    >>> tri = Triangle(data)
    >>> cl = ChainLadder(tri).fit()
    >>> cl.ultimates()
    >>> cl.ibnr()
    >>> cl.summary()
    """

    def __init__(self, triangle: Triangle) -> None:
        if not isinstance(triangle, Triangle):
            raise TypeError(f"Expected Triangle, got {type(triangle).__name__}")
        self._triangle = triangle
        self._factors = None
        self._ultimates = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self) -> "ChainLadder":
        """
        Fit the chain-ladder model by computing volume-weighted ATA factors.

        Volume-weighted factors are:
            f(k) = sum(loss[k+1]) / sum(loss[k])
        summed over all accident years with data at both lags k and k+1.

        Returns
        -------
        self : ChainLadder (for method chaining)
        """
        self._factors = self._triangle.volume_weighted_factors()
        self._ultimates = self._project()
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")

    def _project(self) -> pd.Series:
        """
        Project each accident year to ultimate by chaining factors forward
        from the latest observed development lag.
        """
        ultimates = {}
        data = self._triangle.data
        factors = self._factors

        for ay in self._triangle.origin_years:
            row = data.loc[ay].dropna()
            if row.empty:
                ultimates[ay] = np.nan
                continue

            latest_lag = row.index[-1]
            loss = row.iloc[-1]

            # Chain multiply from latest_lag through the last factor lag
            for lag in factors.index:
                if lag >= latest_lag:
                    f = factors.loc[lag]
                    if not np.isnan(f):
                        loss *= f

            ultimates[ay] = loss

        return pd.Series(ultimates, name="ultimate")

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #

    def factors(self) -> pd.Series:
        """
        Return the fitted volume-weighted ATA development factors.

        Returns
        -------
        pd.Series indexed by development lag.
        """
        self._check_fitted()
        return self._factors.copy()

    def ultimates(self) -> pd.Series:
        """
        Return the projected ultimate loss for each accident year.

        Returns
        -------
        pd.Series indexed by accident year.
        """
        self._check_fitted()
        return self._ultimates.copy()

    def ibnr(self) -> pd.Series:
        """
        Return the estimated IBNR (incurred but not reported) reserve
        for each accident year.

        IBNR = ultimate - latest diagonal

        Returns
        -------
        pd.Series indexed by accident year.
        """
        self._check_fitted()
        return (self._ultimates - self._triangle.latest_diagonal).rename("ibnr")

    def summary(self, alpha: float = 0.05, n_boot: int = 999) -> pd.DataFrame:
        """
        Return a summary DataFrame with ultimates, IBNR, and bootstrap CIs.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals (default 0.05 → 95% CI).
        n_boot : int
            Number of bootstrap resamples (default 999).

        Returns
        -------
        pd.DataFrame with columns:
            latest, ultimate, ibnr, ci_lower, ci_upper
        """
        self._check_fitted()

        boot_ults = self._bootstrap_ultimates(n_boot=n_boot)
        lo = np.percentile(boot_ults, 100 * alpha / 2, axis=0)
        hi = np.percentile(boot_ults, 100 * (1 - alpha / 2), axis=0)

        return pd.DataFrame({
            "latest":   self._triangle.latest_diagonal,
            "ultimate": self._ultimates,
            "ibnr":     self.ibnr(),
            "ci_lower": pd.Series(lo, index=self._triangle.origin_years),
            "ci_upper": pd.Series(hi, index=self._triangle.origin_years),
        })

    def total_ibnr(self) -> float:
        """Return the total IBNR reserve across all accident years."""
        self._check_fitted()
        return float(self.ibnr().sum())

    # ------------------------------------------------------------------ #
    # Bootstrap                                                           #
    # ------------------------------------------------------------------ #

    def _bootstrap_ultimates(self, n_boot: int = 999) -> np.ndarray:
        """
        Bootstrap the ultimate estimates by resampling accident years
        with replacement and recomputing factors + projections each time.

        Returns
        -------
        np.ndarray of shape (n_boot, n_origins)
        """
        data = self._triangle.data
        origin_years = list(self._triangle.origin_years)
        n = len(origin_years)
        results = np.zeros((n_boot, n))

        for b in range(n_boot):
            # Resample accident year rows with replacement
            sample_idx = np.random.choice(origin_years, size=n, replace=True)
            sample = data.loc[sample_idx].reset_index(drop=True)
            sample.index = origin_years  # restore original index labels

            try:
                boot_tri = Triangle(sample)
                boot_factors = boot_tri.volume_weighted_factors()

                # Project using bootstrap factors
                for i, ay in enumerate(origin_years):
                    row = data.loc[ay].dropna()
                    if row.empty:
                        results[b, i] = np.nan
                        continue
                    latest_lag = row.index[-1]
                    loss = row.iloc[-1]
                    for lag in boot_factors.index:
                        if lag >= latest_lag:
                            f = boot_factors.loc[lag]
                            if not np.isnan(f):
                                loss *= f
                    results[b, i] = loss
            except Exception:
                # If a bootstrap sample is degenerate, use point estimate
                results[b] = self._ultimates.values

        return results

    # ------------------------------------------------------------------ #
    # Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"ChainLadder({status}, "
            f"origins={self._triangle.n_origins}, "
            f"dev_lags={self._triangle.n_devs})"
        )
