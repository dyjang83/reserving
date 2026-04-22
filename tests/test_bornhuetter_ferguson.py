import pytest
import numpy as np
import pandas as pd
from reserving.triangle import Triangle
from reserving.methods.chain_ladder import ChainLadder
from reserving.methods.bornhuetter_ferguson import BornhuetterFerguson


class TestBFConstruction:

    def test_scalar_apriori(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65)
        assert isinstance(bf, BornhuetterFerguson)

    def test_series_apriori(self, simple_triangle):
        apriori = pd.Series(
            [0.60, 0.65, 0.70], index=simple_triangle.origin_years
        )
        bf = BornhuetterFerguson(simple_triangle, apriori=apriori)
        assert isinstance(bf, BornhuetterFerguson)

    def test_rejects_non_triangle(self):
        with pytest.raises(TypeError):
            BornhuetterFerguson("not a triangle", apriori=0.65)

    def test_rejects_bad_apriori_type(self, simple_triangle):
        with pytest.raises(TypeError):
            BornhuetterFerguson(simple_triangle, apriori=[0.6, 0.65, 0.7])

    def test_rejects_missing_apriori_years(self, simple_triangle):
        apriori = pd.Series([0.65], index=[2021])  # missing 2022, 2023
        with pytest.raises(ValueError):
            BornhuetterFerguson(simple_triangle, apriori=apriori)

    def test_raises_before_fit(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65)
        with pytest.raises(RuntimeError):
            bf.ultimates()
        with pytest.raises(RuntimeError):
            bf.ibnr()


class TestBFRepr:

    def test_repr_unfitted(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65)
        assert "not fitted" in repr(bf)

    def test_repr_fitted(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        assert "fitted" in repr(bf)
        assert "0.650" in repr(bf)

    def test_repr_variable_apriori(self, simple_triangle):
        apriori = pd.Series(
            [0.60, 0.65, 0.70], index=simple_triangle.origin_years
        )
        bf = BornhuetterFerguson(simple_triangle, apriori=apriori).fit()
        assert "variable" in repr(bf)


class TestBFFactorsAndCDFs:

    def test_factors_match_chain_ladder(self, simple_triangle):
        """BF uses chain-ladder factors — they must match."""
        cl = ChainLadder(simple_triangle).fit()
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        pd.testing.assert_series_equal(cl.factors(), bf.factors())

    def test_cdfs_decreasing(self, simple_triangle):
        """CDFs should decrease as development lag increases (closer to 1.0)."""
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        cdfs = bf.cdfs().sort_index()
        for i in range(len(cdfs) - 1):
            assert cdfs.iloc[i] >= cdfs.iloc[i + 1] - 1e-9

    def test_pct_reported_between_0_and_1(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        pct = bf.pct_reported()
        assert (pct >= 0.0).all()
        assert (pct <= 1.0 + 1e-9).all()

    def test_last_lag_pct_reported_is_1(self, simple_triangle):
        """AY 2021 is at the last lag — should be 100% reported."""
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        pct = bf.pct_reported()
        assert abs(pct.loc[2021] - 1.0) < 1e-9


class TestBFUltimates:

    def test_ultimates_length(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        assert len(bf.ultimates()) == simple_triangle.n_origins

    def test_ultimates_index(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        assert list(bf.ultimates().index) == list(simple_triangle.origin_years)

    def test_fully_developed_equals_diagonal(self, simple_triangle):
        """
        AY 2021 is at the last lag (100% reported).
        BF ultimate = emerged + 0 = emerged = diagonal.
        """
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        diag = simple_triangle.latest_diagonal
        assert abs(bf.ultimates().loc[2021] - diag.loc[2021]) < 1e-9

    def test_known_answer(self, simple_triangle):
        """
        Manual calculation for AY 2023 (at lag 1):
          f1 = 2400/2200, f2 = 1150/1100
          CDF at lag 1 = f1 * f2
          pct_reported = 1 / CDF
          apriori_ultimate = 1.0 * 0.65 (premium=1, ELR=0.65)
          ultimate = 900 + 0.65 * (1 - pct_reported)
        """
        f1 = 2400 / 2200
        f2 = 1150 / 1100
        cdf_lag1 = f1 * f2
        pct_rep = 1.0 / cdf_lag1
        expected_ult = 900 + 0.65 * (1.0 - pct_rep)

        bf = BornhuetterFerguson(simple_triangle, apriori=0.65, premium=1.0).fit()
        assert abs(bf.ultimates().loc[2023] - expected_ult) < 1e-9

    def test_bf_more_stable_than_cl_for_immature(self, simple_triangle):
        """
        BF should produce a lower IBNR than chain-ladder for very immature
        years (lag 1), because it discounts the chain-ladder projection toward
        the a priori expectation.
        """
        cl = ChainLadder(simple_triangle).fit()
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        # For AY 2023 (lag 1), BF IBNR < CL IBNR
        assert bf.ibnr().loc[2023] <= cl.ibnr().loc[2023] + 1e-9

    def test_with_premium(self, simple_triangle):
        """Premium scales the a priori ultimate."""
        bf_no_prem = BornhuetterFerguson(
            simple_triangle, apriori=1300.0, premium=1.0
        ).fit()
        bf_with_prem = BornhuetterFerguson(
            simple_triangle, apriori=0.65, premium=2000.0
        ).fit()
        # Both should give same result for AY 2021 (100% reported)
        assert abs(
            bf_no_prem.ultimates().loc[2021] -
            bf_with_prem.ultimates().loc[2021]
        ) < 1e-6


class TestBFIBNR:

    def test_ibnr_equals_ultimate_minus_diagonal(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        expected = bf.ultimates() - simple_triangle.latest_diagonal
        pd.testing.assert_series_equal(
            bf.ibnr().round(9), expected.rename("ibnr").round(9)
        )

    def test_ibnr_zero_for_fully_developed(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        assert abs(bf.ibnr().loc[2021]) < 1e-9

    def test_total_ibnr(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        assert abs(bf.total_ibnr() - bf.ibnr().sum()) < 1e-9


class TestBFSummary:

    def test_summary_columns(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        s = bf.summary(n_boot=100)
        for col in ["latest", "ultimate", "ibnr", "ci_lower", "ci_upper"]:
            assert col in s.columns

    def test_ci_bounds_ordered(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        s = bf.summary(n_boot=200)
        assert (s["ci_lower"] <= s["ultimate"] + 1e-6).all()
        assert (s["ci_upper"] >= s["ultimate"] - 1e-6).all()

    def test_summary_shape(self, simple_triangle):
        bf = BornhuetterFerguson(simple_triangle, apriori=0.65).fit()
        s = bf.summary(n_boot=100)
        assert s.shape[0] == simple_triangle.n_origins
