import pandas as pd
import numpy as np
from reserving.triangle import Triangle
from reserving.methods.chain_ladder import ChainLadder


class BornhuetterFerguson:
    """
    Bornhuetter-Ferguson reserve estimator with bootstrap confidence intervals.

    The BF method blends the chain-ladder projection with an a priori expected
    loss ratio. Rather than projecting entirely from observed losses (as
    chain-ladder does), BF uses the expected losses as a prior and credibility-
    weights it against the emerged losses. This makes BF more stable than
    chain-ladder for immature accident years where little loss has emerged.

    The BF ultimate for each accident year is:

        ultimate = emerged_losses + expected_unreported

    where:
        emerged_losses    = latest diagonal (what we've observed)
        expected_unreported = a_priori_ultimate × (1 - % reported)
        % reported        = 1 / CDF_to_ultimate

    The a priori ultimate is derived from the premium and the a priori
    expected loss ratio (ELR): a_priori_ultimate = premium × ELR.

    Parameters
    ----------
    triangle : Triangle
        Cumulative paid or incurred loss triangle.
    apriori : float or pd.Series
        A priori expected loss ratio. If float, applied uniformly to all
        accident years. If Series, must be indexed by accident year.
    premium : float or pd.Series
        Earned premium by accident year. If float, applied uniformly.
        If Series, must be indexed by accident year. Defaults to 1.0,
        in which case apriori is interpreted as expected ultimate losses
        directly (not a ratio).

    Examples
    --------
    >>> from reserving import Triangle, BornhuetterFerguson
    >>> bf = BornhuetterFerguson(tri, apriori=0.65, premium=10000).fit()
    >>> bf.ultimates()
    >>> bf.summary()
    """

    def __init__(
        self,
        triangle: Triangle,
        apriori: float | pd.Series,
        premium: float | pd.Series = 1.0,
    ) -> None:
        if not isinstance(triangle, Triangle):
            raise TypeError(f"Expected Triangle, got {type(triangle).__name__}")

        self._triangle = triangle
        self._apriori = self._broadcast(apriori, triangle.origin_years, "apriori")
        self._premium = self._broadcast(premium, triangle.origin_years, "premium")
        self._factors = None
        self._cdfs = None
        self._pct_reported = None
        self._ultimates = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast(
        value: float | pd.Series,
        index: pd.Index,
        name: str,
    ) -> pd.Series:
        """Broadcast a scalar or Series to match the triangle's origin index."""
        if isinstance(value, (int, float)):
            return pd.Series(float(value), index=index, name=name)
        if isinstance(value, pd.Series):
            missing = set(index) - set(value.index)
            if missing:
                raise ValueError(f"{name} is missing values for accident years: {missing}")
            return value.loc[index].rename(name)
        raise TypeError(f"{name} must be float or pd.Series, got {type(value).__name__}")

    def _compute_cdfs(self, factors: pd.Series) -> pd.Series:
        """
        Compute cumulative development factors (CDFs) from tail to each lag.

        CDF at lag k = product of all factors from lag k onward.
        The CDF at the last lag is 1.0 (fully developed).
        """
        lags = list(factors.index)
        cdfs = {}
        cdf = 1.0
        for lag in reversed(lags):
            cdf *= factors.loc[lag]
            cdfs[lag] = cdf

        # Add CDF for the last development lag (fully developed = 1.0)
        last_lag = self._triangle.dev_lags[-1]
        if last_lag not in cdfs:
            cdfs[last_lag] = 1.0

        return pd.Series(cdfs, name="cdf")

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self) -> "BornhuetterFerguson":
        """
        Fit the BF model using chain-ladder factors as development pattern.

        Steps:
        1. Fit chain-ladder to get volume-weighted ATA factors
        2. Compute CDFs (cumulative factors to ultimate)
        3. Compute % reported = 1 / CDF for each accident year's current lag
        4. Compute BF ultimate = emerged + expected_unreported

        Returns
        -------
        self : BornhuetterFerguson (for method chaining)
        """
        cl = ChainLadder(self._triangle).fit()
        self._factors = cl.factors()
        self._cdfs = self._compute_cdfs(self._factors)

        pct_reported = {}
        for ay in self._triangle.origin_years:
            latest_lag = self._triangle.latest_dev_lag.loc[ay]
            if pd.isna(latest_lag):
                pct_reported[ay] = np.nan
            elif latest_lag in self._cdfs.index:
                pct_reported[ay] = 1.0 / self._cdfs.loc[latest_lag]
            else:
                pct_reported[ay] = 1.0
        self._pct_reported = pd.Series(pct_reported, name="pct_reported")

        self._ultimates = self._project()
        self._fitted = True
        return self

    def _project(self) -> pd.Series:
        """
        Compute BF ultimates:
            ultimate = emerged + a_priori_ultimate × (1 - pct_reported)
        """
        emerged = self._triangle.latest_diagonal
        apriori_ultimate = self._premium * self._apriori
        expected_unreported = apriori_ultimate * (1.0 - self._pct_reported)
        return (emerged + expected_unreported).rename("ultimate")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #

    def factors(self) -> pd.Series:
        """Return the chain-ladder ATA factors used as the development pattern."""
        self._check_fitted()
        return self._factors.copy()

    def cdfs(self) -> pd.Series:
        """Return the cumulative development factors (CDF to ultimate) by lag."""
        self._check_fitted()
        return self._cdfs.copy()

    def pct_reported(self) -> pd.Series:
        """
        Return the estimated percentage of ultimate losses already reported,
        for each accident year at its current development lag.
        """
        self._check_fitted()
        return self._pct_reported.copy()

    def ultimates(self) -> pd.Series:
        """Return the BF ultimate loss estimate for each accident year."""
        self._check_fitted()
        return self._ultimates.copy()

    def ibnr(self) -> pd.Series:
        """
        Return the IBNR reserve for each accident year.

        IBNR = ultimate - latest diagonal
        """
        self._check_fitted()
        return (self._ultimates - self._triangle.latest_diagonal).rename("ibnr")

    def total_ibnr(self) -> float:
        """Return the total IBNR reserve across all accident years."""
        self._check_fitted()
        return float(self.ibnr().sum())

    def summary(self, alpha: float = 0.05, n_boot: int = 999) -> pd.DataFrame:
        """
        Return a summary DataFrame with ultimates, IBNR, and bootstrap CIs.

        Bootstrap resamples accident years with replacement and recomputes
        the full BF calculation each time, preserving the a priori assumptions.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 → 95% CI).
        n_boot : int
            Number of bootstrap resamples (default 999).

        Returns
        -------
        pd.DataFrame with columns: latest, ultimate, ibnr, ci_lower, ci_upper
        """
        self._check_fitted()
        boot_ults = self._bootstrap_ultimates(n_boot=n_boot)
        lo = np.nanpercentile(boot_ults, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boot_ults, 100 * (1 - alpha / 2), axis=0)

        return pd.DataFrame({
            "latest":   self._triangle.latest_diagonal,
            "ultimate": self._ultimates,
            "ibnr":     self.ibnr(),
            "ci_lower": pd.Series(lo, index=self._triangle.origin_years),
            "ci_upper": pd.Series(hi, index=self._triangle.origin_years),
        })

    # ------------------------------------------------------------------ #
    # Bootstrap                                                           #
    # ------------------------------------------------------------------ #

    def _bootstrap_ultimates(self, n_boot: int = 999) -> np.ndarray:
        """Bootstrap ultimates by resampling accident year rows."""
        data = self._triangle.data
        origin_years = list(self._triangle.origin_years)
        n = len(origin_years)
        results = np.zeros((n_boot, n))

        for b in range(n_boot):
            sample_idx = np.random.choice(origin_years, size=n, replace=True)
            sample = data.loc[sample_idx].reset_index(drop=True)
            sample.index = origin_years

            try:
                boot_tri = Triangle(sample)
                boot_cl = ChainLadder(boot_tri).fit()
                boot_cdfs = self._compute_cdfs(boot_cl.factors())

                for i, ay in enumerate(origin_years):
                    emerged = data.loc[ay].dropna()
                    if emerged.empty:
                        results[b, i] = np.nan
                        continue
                    latest_lag = emerged.index[-1]
                    latest_loss = emerged.iloc[-1]

                    pct_rep = (
                        1.0 / boot_cdfs.loc[latest_lag]
                        if latest_lag in boot_cdfs.index
                        else 1.0
                    )
                    apriori_ult = self._premium.loc[ay] * self._apriori.loc[ay]
                    results[b, i] = latest_loss + apriori_ult * (1.0 - pct_rep)

            except Exception:
                results[b] = self._ultimates.values

        return results

    # ------------------------------------------------------------------ #
    # Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        apriori_repr = (
            f"{self._apriori.iloc[0]:.3f}"
            if self._apriori.nunique() == 1
            else "variable"
        )
        return (
            f"BornhuetterFerguson({status}, "
            f"apriori={apriori_repr}, "
            f"origins={self._triangle.n_origins})"
        )
