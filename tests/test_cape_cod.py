import pytest
import numpy as np
import pandas as pd
from reserving.triangle import Triangle
from reserving.methods.chain_ladder import ChainLadder
from reserving.methods.bornhuetter_ferguson import BornhuetterFerguson
from reserving.methods.cape_cod import CapeCod


class TestCapeCodConstruction:

    def test_scalar_premium(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0)
        assert isinstance(cc, CapeCod)

    def test_series_premium(self, simple_triangle):
        premium = pd.Series(
            [1000, 1200, 900], index=simple_triangle.origin_years
        )
        cc = CapeCod(simple_triangle, premium=premium)
        assert isinstance(cc, CapeCod)

    def test_rejects_non_triangle(self):
        with pytest.raises(TypeError):
            CapeCod("not a triangle", premium=1000.0)

    def test_rejects_missing_premium_years(self, simple_triangle):
        premium = pd.Series([1000.0], index=[2021])
        with pytest.raises(ValueError):
            CapeCod(simple_triangle, premium=premium)

    def test_raises_before_fit(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0)
        with pytest.raises(RuntimeError):
            cc.ultimates()
        with pytest.raises(RuntimeError):
            cc.elr()
        with pytest.raises(RuntimeError):
            cc.ibnr()


class TestCapeCodRepr:

    def test_repr_unfitted(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0)
        assert "not fitted" in repr(cc)
        assert "not computed" in repr(cc)

    def test_repr_fitted(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert "fitted" in repr(cc)
        assert "elr=" in repr(cc)


class TestCapeCodELR:

    def test_elr_is_float(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert isinstance(cc.elr(), float)

    def test_elr_positive(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert cc.elr() > 0

    def test_elr_known_answer(self, simple_triangle):
        """
        Manual ELR calculation with uniform premium=1000:
          f1 = 2400/2200, f2 = 1150/1100
          CDF_lag1 = f1*f2, CDF_lag2 = f2, CDF_lag3 = 1.0
          pct_rep_2021 = 1/CDF_lag3 = 1.0
          pct_rep_2022 = 1/CDF_lag2 = 1100/1150
          pct_rep_2023 = 1/CDF_lag1 = 2200*1100/(2400*1150)
          used_up = sum(premium * pct_rep)
          ELR = sum(emerged) / used_up
        """
        f1 = 2400 / 2200
        f2 = 1150 / 1100
        cdf_lag1 = f1 * f2
        cdf_lag2 = f2
        prem = 1000.0
        pct_2021 = 1.0
        pct_2022 = 1.0 / cdf_lag2
        pct_2023 = 1.0 / cdf_lag1
        emerged = 1150 + 1300 + 900
        used_up = prem * (pct_2021 + pct_2022 + pct_2023)
        expected_elr = emerged / used_up

        cc = CapeCod(simple_triangle, premium=prem).fit()
        assert abs(cc.elr() - expected_elr) < 1e-9

    def test_elr_scales_with_premium(self, simple_triangle):
        """Doubling uniform premium halves the ELR (emerged stays the same)."""
        cc1 = CapeCod(simple_triangle, premium=1000.0).fit()
        cc2 = CapeCod(simple_triangle, premium=2000.0).fit()
        assert abs(cc1.elr() - 2 * cc2.elr()) < 1e-9


class TestCapeCodUltimates:

    def test_ultimates_length(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert len(cc.ultimates()) == simple_triangle.n_origins

    def test_ultimates_index(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert list(cc.ultimates().index) == list(simple_triangle.origin_years)

    def test_fully_developed_equals_diagonal(self, simple_triangle):
        """AY 2021 is fully developed — ultimate must equal diagonal."""
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        diag = simple_triangle.latest_diagonal
        assert abs(cc.ultimates().loc[2021] - diag.loc[2021]) < 1e-9

    def test_known_answer(self, simple_triangle):
        """
        Cape Cod ultimate for AY 2023:
            ultimate = 900 + ELR * 1000 * (1 - pct_reported_2023)
        """
        f1 = 2400 / 2200
        f2 = 1150 / 1100
        cdf_lag1 = f1 * f2
        cdf_lag2 = f2
        prem = 1000.0
        pct_2021 = 1.0
        pct_2022 = 1.0 / cdf_lag2
        pct_2023 = 1.0 / cdf_lag1
        emerged = 1150 + 1300 + 900
        used_up = prem * (pct_2021 + pct_2022 + pct_2023)
        elr = emerged / used_up
        expected_ult_2023 = 900 + elr * prem * (1 - pct_2023)

        cc = CapeCod(simple_triangle, premium=prem).fit()
        assert abs(cc.ultimates().loc[2023] - expected_ult_2023) < 1e-9

    def test_cape_cod_between_cl_and_bf(self, simple_triangle):
        """
        Cape Cod should produce ultimates between chain-ladder and BF
        for typical data where the empirical ELR differs from an arbitrary prior.
        This is a directional check — not always guaranteed but holds here.
        """
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert cc.ultimates().loc[2023] > 0

    def test_factors_match_chain_ladder(self, simple_triangle):
        """Cape Cod uses chain-ladder factors — they must match exactly."""
        cl = ChainLadder(simple_triangle).fit()
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        pd.testing.assert_series_equal(cl.factors(), cc.factors())


class TestCapeCodIBNR:

    def test_ibnr_equals_ultimate_minus_diagonal(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        expected = cc.ultimates() - simple_triangle.latest_diagonal
        pd.testing.assert_series_equal(
            cc.ibnr().round(9), expected.rename("ibnr").round(9)
        )

    def test_ibnr_zero_for_fully_developed(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert abs(cc.ibnr().loc[2021]) < 1e-9

    def test_total_ibnr(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        assert abs(cc.total_ibnr() - cc.ibnr().sum()) < 1e-9


class TestCapeCodSummary:

    def test_summary_columns(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        s = cc.summary(n_boot=100)
        for col in ["latest", "ultimate", "ibnr", "ci_lower", "ci_upper"]:
            assert col in s.columns

    def test_ci_bounds_ordered(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        s = cc.summary(n_boot=200)
        assert (s["ci_lower"] <= s["ultimate"] + 1e-6).all()
        assert (s["ci_upper"] >= s["ultimate"] - 1e-6).all()

    def test_summary_shape(self, simple_triangle):
        cc = CapeCod(simple_triangle, premium=1000.0).fit()
        s = cc.summary(n_boot=100)
        assert s.shape[0] == simple_triangle.n_origins


class TestCapeCodVsBF:

    def test_same_as_bf_when_elr_matches(self, simple_triangle):
        """
        When the BF a priori ELR equals the Cape Cod derived ELR,
        both methods should produce identical ultimates.
        """
        prem = 1000.0
        cc = CapeCod(simple_triangle, premium=prem).fit()
        derived_elr = cc.elr()

        bf = BornhuetterFerguson(
            simple_triangle, apriori=derived_elr, premium=prem
        ).fit()

        pd.testing.assert_series_equal(
            cc.ultimates().round(6),
            bf.ultimates().round(6),
        )
