import pandas as pd
import numpy as np
from reserving.triangle import Triangle
from reserving.methods.chain_ladder import ChainLadder
from reserving.methods.bornhuetter_ferguson import BornhuetterFerguson


class CapeCod:
    """
    Cape Cod reserve estimator with bootstrap confidence intervals.

    The Cape Cod method is similar to Bornhuetter-Ferguson but derives
    the expected loss ratio (ELR) from the data itself rather than
    requiring an external a priori assumption. The ELR is computed as:

        ELR = sum(emerged losses) / sum(used-up premium)

    where used-up premium = premium × % reported, and % reported is
    derived from chain-ladder development factors.

    This makes Cape Cod self-contained — it requires premium by accident
    year but no external ELR assumption. Once the ELR is derived, the
    Cape Cod ultimate is computed identically to Bornhuetter-Ferguson:

        ultimate = emerged + ELR × premium × (1 - % reported)

    Parameters
    ----------
    triangle : Triangle
        Cumulative paid or incurred loss triangle.
    premium : float or pd.Series
        Earned premium by accident year. If float, applied uniformly.
        Required — Cape Cod needs premium to compute the ELR.

    Examples
    --------
    >>> from reserving import Triangle, CapeCod
    >>> cc = CapeCod(tri, premium=10000).fit()
    >>> cc.elr()
    >>> cc.ultimates()
    >>> cc.summary()
    """

    def __init__(
        self,
        triangle: Triangle,
        premium: float | pd.Series,
    ) -> None:
        if not isinstance(triangle, Triangle):
            raise TypeError(f"Expected Triangle, got {type(triangle).__name__}")

        self._triangle = triangle
        self._premium = self._broadcast(premium, triangle.origin_years, "premium")
        self._factors = None
        self._cdfs = None
        self._pct_reported = None
        self._elr = None
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
                raise ValueError(
                    f"{name} is missing values for accident years: {missing}"
                )
            return value.loc[index].rename(name)
        raise TypeError(
            f"{name} must be float or pd.Series, got {type(value).__name__}"
        )

    def _compute_cdfs(self, factors: pd.Series) -> pd.Series:
        """Compute cumulative development factors from chain-ladder factors."""
        lags = list(factors.index)
        cdfs = {}
        cdf = 1.0
        for lag in reversed(lags):
            cdf *= factors.loc[lag]
            cdfs[lag] = cdf
        last_lag = self._triangle.dev_lags[-1]
        if last_lag not in cdfs:
            cdfs[last_lag] = 1.0
        return pd.Series(cdfs, name="cdf")

    def _compute_pct_reported(self, cdfs: pd.Series) -> pd.Series:
        """Compute % reported for each accident year at its current lag."""
        pct = {}
        for ay in self._triangle.origin_years:
            latest_lag = self._triangle.latest_dev_lag.loc[ay]
            if pd.isna(latest_lag):
                pct[ay] = np.nan
            elif latest_lag in cdfs.index:
                pct[ay] = 1.0 / cdfs.loc[latest_lag]
            else:
                pct[ay] = 1.0
        return pd.Series(pct, name="pct_reported")

    def _compute_elr(
        self,
        pct_reported: pd.Series,
        premium: pd.Series,
        emerged: pd.Series,
    ) -> float:
        """
        Compute the Cape Cod expected loss ratio.

            ELR = sum(emerged losses) / sum(used-up premium)

        where used-up premium = premium × pct_reported.
        """
        used_up_premium = premium * pct_reported
        total_used_up = used_up_premium.sum()
        if total_used_up == 0:
            raise ValueError(
                "Sum of used-up premium is zero — cannot compute ELR. "
                "Check that premium values are non-zero."
            )
        return float(emerged.sum() / total_used_up)

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self) -> "CapeCod":
        """
        Fit the Cape Cod model.

        Steps:
        1. Fit chain-ladder to get development factors
        2. Compute CDFs and % reported for each accident year
        3. Derive the ELR from the data
        4. Compute ultimates using the BF formula with the derived ELR

        Returns
        -------
        self : CapeCod (for method chaining)
        """
        cl = ChainLadder(self._triangle).fit()
        self._factors = cl.factors()
        self._cdfs = self._compute_cdfs(self._factors)
        self._pct_reported = self._compute_pct_reported(self._cdfs)

        emerged = self._triangle.latest_diagonal
        self._elr = self._compute_elr(self._pct_reported, self._premium, emerged)

        self._ultimates = self._project(self._pct_reported, self._elr)
        self._fitted = True
        return self

    def _project(self, pct_reported: pd.Series, elr: float) -> pd.Series:
        """
        Compute Cape Cod ultimates:
            ultimate = emerged + ELR × premium × (1 - pct_reported)
        """
        emerged = self._triangle.latest_diagonal
        expected_unreported = elr * self._premium * (1.0 - pct_reported)
        return (emerged + expected_unreported).rename("ultimate")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #

    def elr(self) -> float:
        """
        Return the fitted Cape Cod expected loss ratio (ELR).

        The ELR is derived from the data as:
            sum(emerged losses) / sum(used-up premium)
        """
        self._check_fitted()
        return self._elr

    def factors(self) -> pd.Series:
        """Return the chain-ladder ATA factors used as development pattern."""
        self._check_fitted()
        return self._factors.copy()

    def cdfs(self) -> pd.Series:
        """Return cumulative development factors (CDF to ultimate) by lag."""
        self._check_fitted()
        return self._cdfs.copy()

    def pct_reported(self) -> pd.Series:
        """Return % of ultimate losses reported at current lag per accident year."""
        self._check_fitted()
        return self._pct_reported.copy()

    def ultimates(self) -> pd.Series:
        """Return the Cape Cod ultimate loss estimate for each accident year."""
        self._check_fitted()
        return self._ultimates.copy()

    def ibnr(self) -> pd.Series:
        """Return IBNR = ultimate - latest diagonal."""
        self._check_fitted()
        return (self._ultimates - self._triangle.latest_diagonal).rename("ibnr")

    def total_ibnr(self) -> float:
        """Return total IBNR across all accident years."""
        self._check_fitted()
        return float(self.ibnr().sum())

    def summary(self, alpha: float = 0.05, n_boot: int = 999) -> pd.DataFrame:
        """
        Return summary DataFrame with ultimates, IBNR, and bootstrap CIs.

        The bootstrap re-derives the ELR from each resampled triangle,
        preserving the self-contained nature of the Cape Cod method.

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
        """
        Bootstrap ultimates by resampling accident years with replacement
        and re-deriving the ELR from each sample.
        """
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
                boot_pct = self._compute_pct_reported(boot_cdfs)

                boot_emerged = self._triangle.latest_diagonal
                boot_elr = self._compute_elr(
                    boot_pct, self._premium, boot_emerged
                )

                boot_ults = self._project(boot_pct, boot_elr)
                results[b] = boot_ults.values

            except Exception:
                results[b] = self._ultimates.values

        return results

    # ------------------------------------------------------------------ #
    # Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        elr_repr = f"{self._elr:.4f}" if self._fitted else "not computed"
        return (
            f"CapeCod({status}, "
            f"elr={elr_repr}, "
            f"origins={self._triangle.n_origins})"
        )
